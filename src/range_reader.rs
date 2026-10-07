//! Range-based reader interface for COG files
//!
//! This module provides a unified interface for reading byte ranges from various sources
//! (local files, S3, HTTP). This is essential for efficient COG reading since COGs are
//! designed to be read via HTTP Range requests.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

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

    /// Check if this is a local file (fast random access) or remote (expensive reads)
    fn is_local(&self) -> bool {
        let id = self.identifier();
        !id.starts_with("http://") && !id.starts_with("https://") && !id.starts_with("s3://")
    }

    /// Whether this reader already serves the start of the file from memory.
    ///
    /// Used to avoid wrapping a reader in [`PrefixCachedRangeReader`] twice.
    fn has_prefix_cache(&self) -> bool {
        false
    }
}

/// Local file range reader
pub struct LocalRangeReader {
    path: PathBuf,
    size: u64,
}

impl LocalRangeReader {
    /// # Errors
    /// Returns an error if the file does not exist or metadata cannot be read.
    pub fn new(path: impl AsRef<Path>) -> AnyResult<Self> {
        let path = path.as_ref().to_path_buf();
        let metadata = std::fs::metadata(&path)?;
        Ok(Self {
            path,
            size: metadata.len(),
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
}

/// Number of leading bytes [`PrefixCachedRangeReader`] fetches in one request.
///
/// Sized to cover the header, the IFD chain and the out-of-line tag values of a
/// typical COG, which are all placed at the start of the file.
const PREFIX_CACHE_BYTES: u64 = 16 * 1024;

/// Range reader that fetches the first 16 KiB of the source once and serves
/// every read lying fully inside it from memory.
///
/// A COG keeps its header, IFDs and tag values at the start of the file, so
/// opening one through a remote reader would otherwise cost many small
/// sequential requests. Reads that are not fully inside the prefix (including
/// reads straddling its end) are delegated to the inner reader unchanged.
/// The identifier is the inner reader's, so identifier-keyed caches are shared.
pub struct PrefixCachedRangeReader {
    inner: Arc<dyn RangeReader>,
    prefix: Vec<u8>,
}

impl PrefixCachedRangeReader {
    /// Wrap `inner`, reading `min(16 KiB, size)` leading bytes up front.
    ///
    /// # Errors
    /// Returns an error if reading the prefix from `inner` fails.
    pub fn new(inner: Arc<dyn RangeReader>) -> AnyResult<Self> {
        let len = PREFIX_CACHE_BYTES.min(inner.size());
        let prefix = if len == 0 {
            Vec::new()
        } else {
            // Safe cast: len <= PREFIX_CACHE_BYTES
            #[allow(clippy::cast_possible_truncation)]
            inner.read_range(0, len as usize)?
        };
        Ok(Self { inner, prefix })
    }
}

impl RangeReader for PrefixCachedRangeReader {
    fn read_range(&self, offset: u64, length: usize) -> AnyResult<Vec<u8>> {
        let end = offset.checked_add(length as u64);
        if let Some(end) = end
            && end <= self.prefix.len() as u64
        {
            // Safe casts: end <= prefix.len(), which fits in usize
            #[allow(clippy::cast_possible_truncation)]
            return Ok(self.prefix[offset as usize..end as usize].to_vec());
        }
        self.inner.read_range(offset, length)
    }

    fn size(&self) -> u64 {
        self.inner.size()
    }

    fn identifier(&self) -> &str {
        self.inner.identifier()
    }

    fn is_local(&self) -> bool {
        self.inner.is_local()
    }

    fn has_prefix_cache(&self) -> bool {
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
}

/// HTTP range reader for remote COG files
/// Uses reqwest with blocking client for simplicity in sync contexts
pub struct HttpRangeReader {
    url: String,
    size: u64,
    client: reqwest::blocking::Client,
}

impl HttpRangeReader {
    /// # Errors
    /// Returns an error if the HTTP HEAD request fails, the URL is invalid,
    /// or the server does not return a valid Content-Length header.
    pub fn new(url: &str) -> AnyResult<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        // Get file size via HEAD request
        let response = client.head(url).send()?;
        let size = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| format!("HTTP server did not return Content-Length header for {url}"))?;

        Ok(Self {
            url: url.to_string(),
            size,
            client,
        })
    }
}

impl RangeReader for HttpRangeReader {
    fn read_range(&self, offset: u64, length: usize) -> AnyResult<Vec<u8>> {
        let range = format!("bytes={}-{}", offset, offset + length as u64 - 1);
        let response = self.client
            .get(&self.url)
            .header("Range", range)
            .send()?;

        if !response.status().is_success() {
            return Err(format!("HTTP request failed: {}", response.status()).into());
        }

        Ok(response.bytes()?.to_vec())
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn identifier(&self) -> &str {
        &self.url
    }
}

/// S3 range reader for public buckets via HTTPS
///
/// Note: For full S3 support with credentials, use `S3RangeReaderSync` from the `s3` module.
pub struct S3RangeReader {
    size: u64,
    url: String,
}

impl S3RangeReader {
    /// Create from an S3 URL like `s3://bucket/key`
    ///
    /// Validates the URL format but does not fetch size (use `from_https` for that).
    ///
    /// # Errors
    /// Returns an error if the URL is invalid, missing bucket/key, or not using the s3:// scheme.
    pub fn new(url: &str) -> AnyResult<Self> {
        let url_parsed = url::Url::parse(url)?;

        if url_parsed.scheme() != "s3" {
            return Err("URL must use s3:// scheme".into());
        }

        if url_parsed.host_str().is_none() {
            return Err("Missing bucket in S3 URL".into());
        }

        let key = url_parsed.path().trim_start_matches('/');
        if key.is_empty() {
            return Err("Missing key in S3 URL".into());
        }

        Ok(Self {
            size: 0,
            url: url.to_string(),
        })
    }

    /// Create from an HTTPS URL pointing to S3-hosted content
    ///
    /// # Errors
    /// Returns an error if the HTTP HEAD request fails or the URL is invalid.
    pub fn from_https(url: &str) -> AnyResult<Self> {
        let http_reader = HttpRangeReader::new(url)?;

        Ok(Self {
            size: http_reader.size,
            url: url.to_string(),
        })
    }
}

impl RangeReader for S3RangeReader {
    fn read_range(&self, offset: u64, length: usize) -> AnyResult<Vec<u8>> {
        // For public S3 buckets, use HTTP range requests
        // For private buckets, this would use aws-sdk-s3 with credentials
        let client = reqwest::blocking::Client::new();
        let range = format!("bytes={}-{}", offset, offset + length as u64 - 1);

        let response = client
            .get(&self.url)
            .header("Range", range)
            .send()?;

        if !response.status().is_success() {
            return Err(format!("S3 request failed: {}", response.status()).into());
        }

        Ok(response.bytes()?.to_vec())
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn identifier(&self) -> &str {
        &self.url
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

    /// In-memory reader that counts calls to `read_range`.
    struct CountingReader {
        data: Vec<u8>,
        calls: std::sync::Mutex<Vec<(u64, usize)>>,
    }

    impl CountingReader {
        fn new(len: usize) -> Arc<Self> {
            // Non-repeating-ish pattern so offset errors show up
            let data = (0..len).map(|i| (i % 251) as u8).collect();
            Arc::new(Self { data, calls: std::sync::Mutex::new(Vec::new()) })
        }

        fn calls(&self) -> Vec<(u64, usize)> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl RangeReader for CountingReader {
        fn read_range(&self, offset: u64, length: usize) -> AnyResult<Vec<u8>> {
            self.calls.lock().unwrap().push((offset, length));
            let start = offset as usize;
            let end = (start + length).min(self.data.len());
            Ok(self.data[start..end].to_vec())
        }

        fn size(&self) -> u64 {
            self.data.len() as u64
        }

        fn identifier(&self) -> &str {
            "https://example.test/counting.tif"
        }
    }

    #[test]
    fn prefix_reads_inside_prefix_make_no_inner_calls() {
        let inner = CountingReader::new(100_000);
        let reader = PrefixCachedRangeReader::new(inner.clone()).unwrap();
        let prefix_len = PREFIX_CACHE_BYTES as usize;
        assert_eq!(inner.calls(), vec![(0, prefix_len)]);

        assert_eq!(reader.read_range(0, 8).unwrap(), inner.data[0..8]);
        assert_eq!(reader.read_range(192, 4096).unwrap(), inner.data[192..4288]);
        // Last byte of the prefix is still inside
        assert_eq!(
            reader.read_range(prefix_len as u64 - 10, 10).unwrap(),
            inner.data[prefix_len - 10..prefix_len]
        );
        assert_eq!(inner.calls().len(), 1, "only the initial prefix fetch");
        assert_eq!(reader.size(), 100_000);
        assert_eq!(reader.identifier(), inner.identifier());
        assert!(!reader.is_local());
        assert!(reader.has_prefix_cache());
    }

    #[test]
    fn prefix_straddling_and_later_reads_delegate() {
        let inner = CountingReader::new(100_000);
        let reader = PrefixCachedRangeReader::new(inner.clone()).unwrap();
        let p = PREFIX_CACHE_BYTES;

        let straddle = reader.read_range(p - 4, 8).unwrap();
        assert_eq!(straddle, inner.data[(p as usize - 4)..(p as usize + 4)]);

        let after = reader.read_range(50_000, 100).unwrap();
        assert_eq!(after, inner.data[50_000..50_100]);

        assert_eq!(inner.calls()[1..], [(p - 4, 8), (50_000, 100)]);
    }

    #[test]
    fn prefix_file_smaller_than_prefix() {
        let inner = CountingReader::new(1000);
        let reader = PrefixCachedRangeReader::new(inner.clone()).unwrap();
        assert_eq!(inner.calls(), vec![(0, 1000)]);

        assert_eq!(reader.read_range(0, 1000).unwrap(), inner.data);
        assert_eq!(reader.read_range(990, 10).unwrap(), inner.data[990..]);
        assert_eq!(inner.calls().len(), 1);
        assert_eq!(reader.size(), 1000);
    }
}
