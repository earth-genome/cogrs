//! S3 Range Reader implementation using `object_store`
//!
//! This module provides S3-compatible storage access for reading COG files.
//! It supports:
//! - AWS S3
//! - `MinIO`
//! - Any S3-compatible storage (`DigitalOcean` Spaces, Backblaze B2, etc.)
//!
//! # Configuration
//!
//! The reader can be configured via environment variables:
//! - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` - AWS credentials
//! - `AWS_SKIP_SIGNATURE` - Set to "true" for anonymous access to public buckets
//! - `AWS_REGION`, then `AWS_DEFAULT_REGION` - AWS region. If neither is set (and no custom
//!   endpoint is configured) the bucket's region is detected with an unauthenticated
//!   `HEAD https://{bucket}.s3.amazonaws.com` and cached per bucket for the process lifetime
//! - `AWS_ENDPOINT_URL` - Custom endpoint for MinIO/S3-compatible services (disables region
//!   detection; the region then defaults to `us-east-1`)
//! - `AWS_ALLOW_HTTP` - Set to "true" to allow HTTP endpoints (for local `MinIO`)
//!
//! An explicit [`S3Config::region`] always wins over the environment.
//!
//! # Async runtimes
//!
//! [`S3RangeReaderAsync`] is fully async (it implements [`AsyncRangeReader`]).
//! [`S3RangeReaderSync`] (used by `CogReader::open`) blocks the calling thread while the request
//! runs on a private I/O runtime; from async code open COGs with `CogReader::open_async`.
//!
//! # Example
//!
//! ```rust,no_run
//! use cogrs::S3RangeReaderAsync;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//!     // For AWS S3
//!     let reader = S3RangeReaderAsync::new("s3://my-bucket/path/to/file.tif").await?;
//!
//!     // For MinIO (set AWS_ENDPOINT_URL=http://localhost:9000)
//!     // std::env::set_var("AWS_ENDPOINT_URL", "http://localhost:9000");
//!     // std::env::set_var("AWS_ALLOW_HTTP", "true");
//!     // let reader = S3RangeReaderAsync::new("s3://my-bucket/path/to/file.tif").await?;
//!
//!     Ok(())
//! }
//! ```

use crate::async_io::{block_on_io, AsyncRangeReader, AsyncToSync, IoOptions};
use crate::range_reader::RangeReader;
use crate::remote::ObjectStoreRangeReader;
use crate::tiff_utils::AnyResult;
use bytes::Bytes;
use futures::future::BoxFuture;
use object_store::aws::resolve_bucket_region;
use object_store::ClientOptions;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

/// Process-wide cache of detected bucket regions.
static BUCKET_REGIONS: LazyLock<Mutex<HashMap<String, String>>> = LazyLock::new(Mutex::default);

/// Treat unset and empty/whitespace values alike.
fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|v| !v.is_empty())
}

/// Pick the region from explicit config, then `AWS_REGION`, then `AWS_DEFAULT_REGION`.
///
/// Pure (inputs are passed in) so precedence can be tested without touching the process
/// environment. `None` means "not configured": detect it from the bucket, or use the
/// `object_store` default when a custom endpoint is in use.
fn select_region(
    explicit: Option<&str>,
    aws_region: Option<&str>,
    aws_default_region: Option<&str>,
) -> Option<String> {
    non_empty(explicit)
        .or_else(|| non_empty(aws_region))
        .or_else(|| non_empty(aws_default_region))
        .map(str::to_string)
}

/// Region detection is only appropriate for AWS itself: with a custom endpoint
/// (`MinIO`, `LocalStack`, ...) `{bucket}.s3.amazonaws.com` says nothing about the bucket.
fn should_detect_region(region: Option<&str>, endpoint: Option<&str>) -> bool {
    region.is_none() && non_empty(endpoint).is_none()
}

/// Resolve the region to use for `bucket`: explicit > `AWS_REGION` > `AWS_DEFAULT_REGION` >
/// auto-detect (AWS only, cached). `None` means "let `object_store` use its default".
pub(crate) async fn resolve_region(
    bucket: &str,
    explicit: Option<&str>,
    endpoint: Option<&str>,
) -> Option<String> {
    let region = select_region(
        explicit,
        std::env::var("AWS_REGION").ok().as_deref(),
        std::env::var("AWS_DEFAULT_REGION").ok().as_deref(),
    );
    if !should_detect_region(region.as_deref(), endpoint) {
        return region;
    }
    match detect_bucket_region(bucket).await {
        Ok(region) => Some(region),
        Err(e) => {
            tracing::warn!(
                bucket,
                error = %e,
                "Could not detect S3 bucket region; falling back to us-east-1. \
                 Set AWS_REGION to avoid detection"
            );
            None
        }
    }
}

/// Region from the environment only (`AWS_REGION`, then `AWS_DEFAULT_REGION`).
pub(crate) fn region_from_env() -> Option<String> {
    select_region(
        None,
        std::env::var("AWS_REGION").ok().as_deref(),
        std::env::var("AWS_DEFAULT_REGION").ok().as_deref(),
    )
}

/// Resolve a bucket's region, using the process-wide cache.
async fn detect_bucket_region(bucket: &str) -> AnyResult<String> {
    let cache = &*BUCKET_REGIONS;
    if let Some(region) = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(bucket) {
        return Ok(region.clone());
    }

    let options = ClientOptions::new()
        .with_connect_timeout(Duration::from_secs(5))
        .with_timeout(Duration::from_secs(10));
    let region = resolve_bucket_region(bucket, &options).await?;
    tracing::debug!(bucket, %region, "Detected S3 bucket region");

    cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(bucket.to_string(), region.clone());
    Ok(region)
}

/// S3 configuration for connecting to S3-compatible storage
#[derive(Debug, Clone)]
pub struct S3Config {
    /// S3 bucket name
    pub bucket: String,
    /// Object key (path within the bucket)
    pub key: String,
    /// AWS region. `None` means "unset": use `AWS_REGION` / `AWS_DEFAULT_REGION`, else
    /// auto-detect from the bucket (AWS only; custom endpoints default to `us-east-1`)
    pub region: Option<String>,
    /// Custom endpoint URL (for `MinIO`, `LocalStack`, etc.)
    pub endpoint_url: Option<String>,
    /// AWS access key ID
    pub access_key_id: Option<String>,
    /// AWS secret access key
    pub secret_access_key: Option<String>,
    /// Allow HTTP connections (required for local `MinIO` without TLS)
    pub allow_http: bool,
    /// Skip signature verification (for anonymous access to public buckets)
    pub skip_signature: bool,
}

impl S3Config {
    /// Create a new S3 config from an S3 URL
    ///
    /// Parses URLs like `s3://bucket/key/path`
    ///
    /// # Errors
    /// Returns an error if the URL is invalid, missing bucket/key, or not using the s3:// scheme.
    pub fn from_url(url: &str) -> AnyResult<Self> {
        let parsed = url::Url::parse(url)?;

        if parsed.scheme() != "s3" {
            return Err(format!("Expected s3:// URL, got: {}", parsed.scheme()).into());
        }

        let bucket = parsed
            .host_str()
            .ok_or("Missing bucket in S3 URL")?
            .to_string();

        let key = parsed.path().trim_start_matches('/').to_string();

        if key.is_empty() {
            return Err("Missing key in S3 URL".into());
        }

        Ok(Self {
            bucket,
            key,
            // Unset (None) = auto-detect when the reader is opened
            region: region_from_env(),
            endpoint_url: std::env::var("AWS_ENDPOINT_URL").ok(),
            access_key_id: std::env::var("AWS_ACCESS_KEY_ID").ok(),
            secret_access_key: std::env::var("AWS_SECRET_ACCESS_KEY").ok(),
            allow_http: std::env::var("AWS_ALLOW_HTTP")
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(false),
            skip_signature: std::env::var("AWS_SKIP_SIGNATURE")
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(false),
        })
    }

    /// Create a config for `MinIO` with default local settings
    #[must_use] 
    pub fn for_minio(bucket: &str, key: &str, endpoint: &str) -> Self {
        Self {
            bucket: bucket.to_string(),
            key: key.to_string(),
            region: Some("us-east-1".to_string()),
            endpoint_url: Some(endpoint.to_string()),
            access_key_id: std::env::var("AWS_ACCESS_KEY_ID").ok(),
            secret_access_key: std::env::var("AWS_SECRET_ACCESS_KEY").ok(),
            allow_http: true,
            skip_signature: false,
        }
    }
}

/// Async S3 range reader using `object_store`.
///
/// A thin, S3-specific front for [`ObjectStoreRangeReader`]; it implements
/// [`AsyncRangeReader`].
pub struct S3RangeReaderAsync {
    inner: ObjectStoreRangeReader,
}

impl S3RangeReaderAsync {
    /// Create a new S3 range reader from an S3 URL
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cogrs::S3RangeReaderAsync;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    /// let reader = S3RangeReaderAsync::new("s3://my-bucket/data/file.tif").await?;
    /// println!("File size: {} bytes", reader.size());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns an error if the URL is invalid or the S3 object cannot be accessed.
    pub async fn new(url: &str) -> AnyResult<Self> {
        let config = S3Config::from_url(url)?;
        Self::from_config(config).await
    }

    /// Create a new S3 range reader from a config
    ///
    /// # Errors
    /// Returns an error if the S3 configuration is invalid or the object cannot be accessed.
    pub async fn from_config(config: S3Config) -> AnyResult<Self> {
        Self::from_config_with_options(config, &IoOptions::default()).await
    }

    /// Create a new S3 range reader from a config and explicit [`IoOptions`]
    ///
    /// # Errors
    /// Returns an error if the S3 configuration is invalid or the object cannot be accessed.
    pub async fn from_config_with_options(config: S3Config, options: &IoOptions) -> AnyResult<Self> {
        Ok(Self { inner: ObjectStoreRangeReader::open_s3(config, options).await? })
    }

    /// Read a range of bytes asynchronously
    ///
    /// # Errors
    /// Returns an error if the S3 read operation fails due to network issues or invalid ranges.
    pub async fn read_range_async(&self, offset: u64, length: usize) -> AnyResult<Vec<u8>> {
        Ok(self.inner.read_range(offset, length).await?.into())
    }

    /// Get the file size
    #[must_use]
    pub fn size(&self) -> u64 {
        self.inner.size()
    }

    /// Get the S3 URL
    #[must_use]
    pub fn url(&self) -> &str {
        self.inner.identifier()
    }
}

impl AsyncRangeReader for S3RangeReaderAsync {
    fn read_range(&self, offset: u64, len: usize) -> BoxFuture<'_, AnyResult<Bytes>> {
        self.inner.read_range(offset, len)
    }

    fn read_ranges<'a>(&'a self, ranges: &'a [Range<u64>]) -> BoxFuture<'a, AnyResult<Vec<Bytes>>> {
        self.inner.read_ranges(ranges)
    }

    fn size(&self) -> u64 {
        self.inner.size()
    }

    fn identifier(&self) -> &str {
        self.inner.identifier()
    }

    fn is_local(&self) -> bool {
        false
    }

    fn io_options(&self) -> &IoOptions {
        self.inner.io_options()
    }
}

/// Synchronous wrapper for `S3RangeReaderAsync` that implements the `RangeReader` trait
///
/// Reads block the calling thread while the request runs on a private I/O runtime (see
/// [`AsyncToSync`]), so no ambient tokio runtime is needed. That is safe from plain threads and
/// `spawn_blocking` threads; on an async worker thread it blocks that worker, so from async
/// code use `CogReader::open_async`.
pub struct S3RangeReaderSync {
    inner: AsyncToSync,
}

impl S3RangeReaderSync {
    /// Create a new sync S3 range reader
    ///
    /// # Errors
    /// Returns an error if the URL is invalid or the S3 object cannot be accessed.
    pub fn new(url: &str) -> AnyResult<Self> {
        let url = url.to_string();
        let inner = block_on_io(async move { S3RangeReaderAsync::new(&url).await })??;
        Self::from_async(inner)
    }

    /// Create from an existing async reader
    ///
    /// # Errors
    /// Never fails; the `Result` is kept for API compatibility.
    pub fn from_async(inner: S3RangeReaderAsync) -> AnyResult<Self> {
        Ok(Self { inner: AsyncToSync::new(Arc::new(inner)) })
    }
}

impl RangeReader for S3RangeReaderSync {
    fn read_range(&self, offset: u64, length: usize) -> AnyResult<Vec<u8>> {
        self.inner.read_range(offset, length)
    }

    fn size(&self) -> u64 {
        self.inner.size()
    }

    fn identifier(&self) -> &str {
        self.inner.identifier()
    }

    fn is_local(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s3_config_from_url() {
        let config = S3Config::from_url("s3://my-bucket/path/to/file.tif").unwrap();
        assert_eq!(config.bucket, "my-bucket");
        assert_eq!(config.key, "path/to/file.tif");
    }

    #[test]
    fn test_s3_config_from_url_simple() {
        let config = S3Config::from_url("s3://bucket/file.tif").unwrap();
        assert_eq!(config.bucket, "bucket");
        assert_eq!(config.key, "file.tif");
    }

    #[test]
    fn test_s3_config_invalid_scheme() {
        let result = S3Config::from_url("http://bucket/file.tif");
        assert!(result.is_err());
    }

    #[test]
    fn test_s3_config_missing_key() {
        let result = S3Config::from_url("s3://bucket/");
        assert!(result.is_err());
    }

    #[test]
    fn test_minio_config() {
        let config = S3Config::for_minio("test-bucket", "data/test.tif", "http://localhost:9000");
        assert_eq!(config.bucket, "test-bucket");
        assert_eq!(config.key, "data/test.tif");
        assert_eq!(config.endpoint_url, Some("http://localhost:9000".to_string()));
        assert!(config.allow_http);
    }
}

#[cfg(test)]
mod region_tests {
    use super::*;

    #[test]
    fn select_region_precedence() {
        assert_eq!(select_region(Some("eu-west-1"), Some("us-east-2"), Some("us-west-1")).as_deref(), Some("eu-west-1"));
        assert_eq!(select_region(None, Some("us-east-2"), Some("us-west-1")).as_deref(), Some("us-east-2"));
        assert_eq!(select_region(None, None, Some("us-west-1")).as_deref(), Some("us-west-1"));
        assert_eq!(select_region(None, None, None), None);
    }

    #[test]
    fn select_region_ignores_empty_values() {
        assert_eq!(select_region(Some(""), Some("  "), Some("us-west-2")).as_deref(), Some("us-west-2"));
        assert_eq!(select_region(Some(" "), Some(""), Some("")), None);
        assert_eq!(select_region(None, Some(" us-west-2 "), None).as_deref(), Some("us-west-2"));
    }

    #[test]
    fn detection_only_without_region_or_custom_endpoint() {
        assert!(should_detect_region(None, None));
        assert!(should_detect_region(None, Some("")));
        assert!(!should_detect_region(Some("us-west-2"), None));
        assert!(!should_detect_region(None, Some("http://localhost:9000")));
        assert!(!should_detect_region(Some("us-east-1"), Some("http://localhost:9000")));
    }

    #[tokio::test]
    async fn detected_region_is_cached_per_bucket() {
        // A cache hit must return without any network access.
        BUCKET_REGIONS
            .lock()
            .unwrap()
            .insert("cogrs-test-cached-bucket".to_string(), "ap-south-1".to_string());
        let region = detect_bucket_region("cogrs-test-cached-bucket").await.unwrap();
        assert_eq!(region, "ap-south-1");
    }
}
