//! Range-based reader interface for COG files
//!
//! # Sync readers and async runtimes
//!
//! [`RangeReader`] is the *synchronous* interface: each read blocks the calling thread. It is
//! what plain-thread users and custom readers implement, and what [`LocalRangeReader`] and
//! [`MemoryRangeReader`] provide. Remote sources are natively asynchronous
//! ([`AsyncRangeReader`](crate::AsyncRangeReader), see [`crate::remote`]); the sync remote
//! readers here ([`HttpRangeReader`], [`S3RangeReaderSync`](crate::S3RangeReaderSync)) wrap
//! them with [`AsyncToSync`], which runs the request on a private runtime and blocks the
//! calling thread. That is safe from plain threads and `spawn_blocking` threads and never
//! panics or deadlocks inside a runtime, but on an async worker thread it blocks that worker:
//! from async code use [`CogReader::open_async`](crate::CogReader::open_async) and
//! [`TileExtractor::extract`](crate::TileExtractor::extract), which never block a thread on
//! the network.
//!
//! This module provides a unified interface for reading byte ranges from various sources
//! (local files, S3, HTTP). This is essential for efficient COG reading since COGs are
//! designed to be read via HTTP Range requests.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::async_io::{block_on_io, AsyncToSync};
use crate::remote::ObjectStoreRangeReader;
use crate::tiff_utils::AnyResult;

/// Trait for reading byte ranges from any source
///
/// This abstraction allows the same COG reading code to work with:
/// - Local files (using seek + read)
/// - S3 objects (using `GetObject` with Range header)
/// - HTTP URLs (using Range header)
pub trait RangeReader: Send + Sync {
    /// Read a range of bytes from the source
    ///
    /// # Errors
    /// Returns an error if the read operation fails due to I/O errors, network issues, or invalid ranges.
    fn read_range(&self, offset: u64, length: usize) -> AnyResult<Vec<u8>>;

    /// Get the total size of the source in bytes
    fn size(&self) -> u64;

    /// Get a human-readable identifier for this source (for logging/errors)
    fn identifier(&self) -> &str;

    /// Token that changes when the content changes (see
    /// [`AsyncRangeReader::version`](crate::AsyncRangeReader::version)); `None` if unknown.
    fn version(&self) -> Option<&str> {
        None
    }

    /// Check if this is a local file (fast random access) or remote (expensive reads)
    fn is_local(&self) -> bool {
        let id = self.identifier();
        !id.starts_with("http://") && !id.starts_with("https://") && !id.starts_with("s3://")
    }

    /// True when `read_range` never blocks on I/O (e.g. in-memory data), so async adapters
    /// may call it directly instead of moving it to a blocking thread.
    fn reads_inline(&self) -> bool {
        false
    }
}

/// Local file range reader
pub struct LocalRangeReader {
    path: PathBuf,
    size: u64,
    /// `"{size}:{modification time in ns}"` as of opening
    version: String,
}

impl LocalRangeReader {
    /// # Errors
    /// Returns an error if the file does not exist or metadata cannot be read.
    pub fn new(path: impl AsRef<Path>) -> AnyResult<Self> {
        let path = path.as_ref().to_path_buf();
        let metadata = std::fs::metadata(&path)?;
        let modified_ns = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        Ok(Self {
            path,
            size: metadata.len(),
            version: format!("{}:{modified_ns}", metadata.len()),
        })
    }
}

/// In-memory range reader for bytes already loaded into memory
///
/// This is useful when you have already fetched COG data (e.g., from a cache
/// or network) and want to parse it without writing to disk.
///
/// # Example
///
/// ```rust,no_run
/// use cogrs::{CogReader, MemoryRangeReader};
/// use std::sync::Arc;
///
/// fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
///     let bytes = std::fs::read("path/to/file.tif")?;
///     let reader = MemoryRangeReader::new(bytes, "cached://my-layer.tif".to_string());
///     let cog = CogReader::from_reader(Arc::new(reader))?;
///     Ok(())
/// }
/// ```
pub struct MemoryRangeReader {
    data: Arc<Vec<u8>>,
    identifier: String,
}

impl MemoryRangeReader {
    /// Create a new `MemoryRangeReader` from a byte vector
    ///
    /// # Arguments
    /// * `data` - The COG file bytes
    /// * `identifier` - A human-readable identifier for logging (e.g., "<memory://layer.tif>")
    #[must_use] 
    pub fn new(data: Vec<u8>, identifier: String) -> Self {
        Self {
            data: Arc::new(data),
            identifier,
        }
    }

    /// Create from an `Arc<Vec<u8>>` to avoid cloning large buffers
    #[must_use] 
    pub fn from_arc(data: Arc<Vec<u8>>, identifier: String) -> Self {
        Self { data, identifier }
    }
}

impl RangeReader for MemoryRangeReader {
    fn read_range(&self, offset: u64, length: usize) -> AnyResult<Vec<u8>> {
        // Cast is safe for in-memory data: usize is always sufficient for memory addresses
        #[allow(clippy::cast_possible_truncation)]
        let start = offset as usize;
        let end = (start + length).min(self.data.len());
        if start >= self.data.len() {
            return Ok(vec![]);
        }
        Ok(self.data[start..end].to_vec())
    }

    fn size(&self) -> u64 {
        self.data.len() as u64
    }

    fn identifier(&self) -> &str {
        &self.identifier
    }

    fn is_local(&self) -> bool {
        true // Memory is fast, treat as local
    }

    fn reads_inline(&self) -> bool {
        true
    }
}

impl RangeReader for LocalRangeReader {
    fn read_range(&self, offset: u64, length: usize) -> AnyResult<Vec<u8>> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buffer = vec![0u8; length];
        file.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn identifier(&self) -> &str {
        self.path.to_str().unwrap_or("<invalid path>")
    }

    fn version(&self) -> Option<&str> {
        Some(&self.version)
    }
}

/// Blocking HTTP(S) range reader for remote COG files.
///
/// A synchronous front for [`ObjectStoreRangeReader`] (see the module docs for what that means
/// on async threads). Opening costs one ranged `GET`; the first 16 KiB are kept in memory.
pub struct HttpRangeReader {
    inner: AsyncToSync,
}

impl HttpRangeReader {
    /// Open `url` (`http://` or `https://`).
    ///
    /// # Errors
    /// Returns an error if the URL is invalid, the server does not support range requests, or
    /// the object cannot be read.
    pub fn new(url: &str) -> AnyResult<Self> {
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(format!("Expected an http:// or https:// URL, got: {url}").into());
        }
        let url = url.to_string();
        let reader = block_on_io(async move { ObjectStoreRangeReader::open(&url).await })??;
        Ok(Self { inner: AsyncToSync::new(Arc::new(reader)) })
    }
}

impl RangeReader for HttpRangeReader {
    fn read_range(&self, offset: u64, length: usize) -> AnyResult<Vec<u8>> {
        self.inner.read_range(offset, length)
    }

    fn size(&self) -> u64 {
        self.inner.size()
    }

    fn identifier(&self) -> &str {
        self.inner.identifier()
    }

    fn version(&self) -> Option<&str> {
        self.inner.version()
    }

    fn is_local(&self) -> bool {
        false
    }
}

/// Create a range reader from a path or URL
///
/// # Errors
/// Returns an error if the source cannot be opened or is invalid.
pub fn create_range_reader(source: &str) -> AnyResult<Arc<dyn RangeReader>> {
    if source.starts_with("s3://") {
        // Use the proper S3 reader that supports credentials and custom endpoints
        Ok(Arc::new(crate::s3::S3RangeReaderSync::new(source)?))
    } else if source.starts_with("http://") || source.starts_with("https://") {
        Ok(Arc::new(HttpRangeReader::new(source)?))
    } else {
        Ok(Arc::new(LocalRangeReader::new(source)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_local_range_reader() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"Hello, World!").unwrap();

        let reader = LocalRangeReader::new(file.path()).unwrap();
        assert_eq!(reader.size(), 13);

        let data = reader.read_range(0, 5).unwrap();
        assert_eq!(&data, b"Hello");

        let data = reader.read_range(7, 5).unwrap();
        assert_eq!(&data, b"World");
    }

    #[test]
    fn test_memory_range_reader() {
        let data = b"Hello, World!".to_vec();
        let reader = MemoryRangeReader::new(data, "test://memory".to_string());

        assert_eq!(reader.size(), 13);
        assert_eq!(reader.identifier(), "test://memory");
        assert!(reader.is_local());

        // Test reading ranges
        let range1 = reader.read_range(0, 5).unwrap();
        assert_eq!(&range1, b"Hello");

        let range2 = reader.read_range(7, 5).unwrap();
        assert_eq!(&range2, b"World");

        // Test reading past end (should return partial data)
        let range3 = reader.read_range(10, 10).unwrap();
        assert_eq!(&range3, b"ld!");

        // Test reading from beyond end (should return empty)
        let range4 = reader.read_range(100, 10).unwrap();
        assert!(range4.is_empty());
    }

    #[test]
    fn test_memory_range_reader_from_arc() {
        let data = Arc::new(b"Test data".to_vec());
        let reader = MemoryRangeReader::from_arc(data.clone(), "arc://test".to_string());

        assert_eq!(reader.size(), 9);
        let result = reader.read_range(0, 4).unwrap();
        assert_eq!(&result, b"Test");
    }
}

#[cfg(test)]
mod http_tests {
    use super::*;
    use crate::test_support::serve_bytes;
    use std::time::Duration;

    fn data() -> Vec<u8> {
        (0..50_000usize).map(|i| (i % 253) as u8).collect()
    }

    #[test]
    fn http_range_reader_from_a_plain_thread() {
        let d = data();
        let (base, log) = serve_bytes(d.clone(), Duration::ZERO);
        let reader = HttpRangeReader::new(&format!("{base}/sync.bin")).unwrap();
        assert_eq!((reader.size(), reader.is_local()), (50_000, false));
        assert_eq!(reader.identifier(), format!("{base}/sync.bin"));
        assert_eq!(reader.read_range(10, 100).unwrap(), d[10..110]);
        assert_eq!(reader.read_range(30_000, 500).unwrap(), d[30_000..30_500]);
        // open (prefix) + one request outside the prefix
        assert_eq!(log.lock().len(), 2);
    }

    #[test]
    fn http_range_reader_rejects_other_schemes() {
        assert!(HttpRangeReader::new("s3://bucket/key").is_err());
        assert!(HttpRangeReader::new("/tmp/file").is_err());
    }

    #[test]
    fn create_range_reader_routes_http_sources() {
        let (base, _log) = serve_bytes(data(), Duration::ZERO);
        let reader = create_range_reader(&format!("{base}/x.bin")).unwrap();
        assert!(!reader.is_local());
        assert_eq!(reader.size(), 50_000);
    }

    /// The sync reader inside a single-threaded runtime blocks it but neither panics nor
    /// deadlocks (the request runs on the private I/O runtime).
    #[tokio::test(flavor = "current_thread")]
    async fn http_range_reader_inside_a_current_thread_runtime() {
        let d = data();
        let (base, _log) = serve_bytes(d.clone(), Duration::from_millis(10));
        let reader = HttpRangeReader::new(&format!("{base}/rt.bin")).unwrap();
        assert_eq!(reader.read_range(40_000, 64).unwrap(), d[40_000..40_064]);
    }
}
