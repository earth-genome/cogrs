//! S3 COG source.
//!
//! Scans S3 buckets for `GeoTIFF` files and extracts metadata.

use std::collections::HashMap;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use object_store::path::Path as ObjectPath;
use tracing::{debug, warn};

use crate::async_io::IoOptions;
use crate::cog_reader::{CogReader, OverviewQualityHint};
use crate::remote::{s3_object_store, ObjectStoreRangeReader};
use crate::s3::S3Config;

use super::{CogEntry, CogLocation, CogSource};

/// Options for scanning S3 buckets.
#[derive(Debug, Clone)]
pub struct S3ScanOptions {
    /// Prefix to filter objects (e.g., "data/layers/")
    pub prefix: Option<String>,
    /// File extensions to consider as `GeoTIFFs` (case-insensitive)
    pub extensions: Vec<String>,
    /// Maximum number of objects to scan
    pub max_objects: Option<usize>,
    /// Custom endpoint URL (for `MinIO`, `LocalStack`, etc.)
    pub endpoint_url: Option<String>,
    /// AWS region. `None` = `AWS_REGION` / `AWS_DEFAULT_REGION`, else auto-detected from the
    /// bucket (not when a custom endpoint is set)
    pub region: Option<String>,
    /// Allow HTTP connections
    pub allow_http: bool,
    /// Skip request signing (anonymous access to public buckets). Defaults to
    /// `AWS_SKIP_SIGNATURE=true` in the environment.
    pub skip_signature: bool,
    /// Number of objects whose metadata is read concurrently
    pub concurrency: usize,
    /// Network tuning (timeouts, retries, request limits)
    pub io: IoOptions,
}

impl Default for S3ScanOptions {
    fn default() -> Self {
        Self {
            prefix: None,
            extensions: vec![
                "tif".to_string(),
                "tiff".to_string(),
                "geotiff".to_string(),
                "geotif".to_string(),
            ],
            max_objects: None,
            endpoint_url: std::env::var("AWS_ENDPOINT_URL").ok(),
            // None = auto-detect (AWS only) when scanning
            region: crate::s3::region_from_env(),
            allow_http: std::env::var("AWS_ALLOW_HTTP")
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(false),
            skip_signature: std::env::var("AWS_SKIP_SIGNATURE")
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(false),
            concurrency: 16,
            io: IoOptions::default(),
        }
    }
}

impl S3ScanOptions {
    /// Create options with a specific prefix
    #[must_use] 
    pub fn with_prefix(prefix: &str) -> Self {
        Self {
            prefix: Some(prefix.to_string()),
            ..Default::default()
        }
    }

    /// Set the endpoint URL (for `MinIO`, etc.)
    #[must_use] 
    pub fn with_endpoint(mut self, endpoint: &str) -> Self {
        self.endpoint_url = Some(endpoint.to_string());
        self
    }

    /// Set the region
    #[must_use] 
    pub fn with_region(mut self, region: &str) -> Self {
        self.region = Some(region.to_string());
        self
    }

    /// Allow HTTP connections
    #[must_use] 
    pub fn with_allow_http(mut self, allow: bool) -> Self {
        self.allow_http = allow;
        self
    }

    /// Skip request signing (anonymous access to public buckets)
    #[must_use]
    pub fn with_skip_signature(mut self, skip: bool) -> Self {
        self.skip_signature = skip;
        self
    }

    /// Set how many objects are read concurrently
    #[must_use]
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Set maximum number of objects to scan
    #[must_use] 
    pub fn with_max_objects(mut self, max: usize) -> Self {
        self.max_objects = Some(max);
        self
    }
}

/// COG source that scans S3 buckets.
///
/// Discovers `GeoTIFF` files in an S3 bucket and extracts metadata from each.
///
/// # Example
///
/// ```rust,no_run
/// use cogrs::source::{CogSource, S3CogSource, S3ScanOptions};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
///     // Scan with default options
///     let source = S3CogSource::scan("my-bucket", S3ScanOptions::default()).await?;
///
///     // Or scan with a prefix
///     let source = S3CogSource::scan("my-bucket", S3ScanOptions::with_prefix("data/cogs/")).await?;
///
///     for entry in source.entries() {
///         println!("Found: {} ({} bands)", entry.name, entry.bands);
///     }
///     Ok(())
/// }
/// ```
pub struct S3CogSource {
    bucket: String,
    entries: Vec<CogEntry>,
    entries_by_name: HashMap<String, usize>,
}

impl S3CogSource {
    /// Scan an S3 bucket for COG files.
    ///
    /// # Arguments
    /// * `bucket` - S3 bucket name
    /// * `options` - Scanning options (prefix, extensions, etc.)
    ///
    /// # Returns
    /// A `S3CogSource` containing metadata for all discovered COGs.
    ///
    /// # Errors
    /// Returns an error if the S3 bucket cannot be accessed or listed.
    pub async fn scan(
        bucket: &str,
        options: S3ScanOptions,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Credentials come from the environment; the client is shared with the per-object
        // readers below (and with any other COG opened from this bucket).
        let config = S3Config {
            bucket: bucket.to_string(),
            key: String::new(),
            region: options.region.clone(),
            endpoint_url: options.endpoint_url.clone(),
            access_key_id: std::env::var("AWS_ACCESS_KEY_ID").ok(),
            secret_access_key: std::env::var("AWS_SECRET_ACCESS_KEY").ok(),
            allow_http: options.allow_http,
            skip_signature: options.skip_signature,
        };
        let store = s3_object_store(&config, &options.io).await?;

        // List objects in the bucket
        let prefix = options.prefix.as_deref().map(ObjectPath::from);
        let mut items: Vec<_> = store.list(prefix.as_ref()).try_collect().await?;

        // Sort by key for deterministic ordering
        items.sort_by(|a, b| a.location.as_ref().cmp(b.location.as_ref()));

        let candidates: Vec<_> = items
            .into_iter()
            .filter(|meta| {
                meta.location
                    .as_ref()
                    .rsplit('.')
                    .next()
                    .is_some_and(|ext| options.extensions.iter().any(|x| x.eq_ignore_ascii_case(ext)))
            })
            .collect();

        // Read each object's metadata (one ranged request each) with bounded concurrency,
        // keeping key order. Stopping at `max_objects` cancels the reads still in flight.
        let mut reads = futures::stream::iter(candidates.into_iter().map(|meta| {
            let config = S3Config { key: meta.location.as_ref().to_string(), ..config.clone() };
            let io = options.io.clone();
            async move {
                let result = Self::read_cog_entry(config, &meta, &io).await;
                (meta, result)
            }
        }))
        .buffered(options.concurrency.max(1));

        let mut entries = Vec::new();
        let mut entries_by_name = HashMap::new();
        while let Some((meta, read)) = reads.next().await {
            let key = meta.location.as_ref();
            match read {
                Ok(cog_entry) => {
                    let idx = entries.len();
                    entries_by_name.insert(cog_entry.name.clone(), idx);
                    entries.push(cog_entry);
                    debug!(key = %key, "Discovered S3 COG");
                }
                Err(e) => {
                    warn!(key = %key, error = %e, "Failed to read S3 COG metadata");
                }
            }
            if options.max_objects.is_some_and(|max| entries.len() >= max) {
                break;
            }
        }

        Ok(Self {
            bucket: bucket.to_string(),
            entries,
            entries_by_name,
        })
    }

    /// Read metadata from a single S3 COG file.
    async fn read_cog_entry(
        config: S3Config,
        meta: &object_store::ObjectMeta,
        io: &IoOptions,
    ) -> Result<CogEntry, Box<dyn std::error::Error + Send + Sync>> {
        // Extract name from key
        let key = meta.location.as_ref();
        let name = key
            .rsplit('/')
            .next()
            .and_then(|f| f.rsplit('.').next_back())
            .unwrap_or("unknown")
            .to_string();

        let location = CogLocation::S3 { bucket: config.bucket.clone(), key: config.key.clone() };

        // Entry metadata only needs the IFDs, not the overview-quality sampling of tiles
        let reader = ObjectStoreRangeReader::open_s3(config, io).await?;
        let cog_reader = CogReader::from_async_reader_with_hint(Arc::new(reader), OverviewQualityHint::NoneUsable).await?;

        let size_bytes = Some(meta.size);
        let last_modified = Some(meta.last_modified.into());

        CogEntry::from_reader(&cog_reader, name, location, size_bytes, last_modified).map_err(std::convert::Into::into)
    }

    /// Get the bucket name
    #[must_use] 
    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    /// Get statistics about discovered COGs.
    #[must_use] 
    pub fn stats(&self) -> S3SourceStats {
        let total_size: u64 = self.entries.iter().filter_map(|e| e.size_bytes).sum();
        let tiled_count = self.entries.iter().filter(|e| e.is_tiled).count();
        let with_overviews = self.entries.iter().filter(|e| e.has_overviews).count();

        S3SourceStats {
            file_count: self.entries.len(),
            total_size_bytes: total_size,
            tiled_count,
            with_overviews,
        }
    }
}

impl CogSource for S3CogSource {
    fn entries(&self) -> &[CogEntry] {
        &self.entries
    }

    fn get(&self, name: &str) -> Option<&CogEntry> {
        self.entries_by_name.get(name).map(|&idx| &self.entries[idx])
    }
}

/// Statistics about an S3 COG source.
#[derive(Debug, Clone)]
pub struct S3SourceStats {
    /// Number of COG files discovered
    pub file_count: usize,
    /// Total size of all files in bytes
    pub total_size_bytes: u64,
    /// Number of properly tiled COGs
    pub tiled_count: usize,
    /// Number of COGs with overviews
    pub with_overviews: usize,
}

impl S3SourceStats {
    /// Get total size in megabytes
    #[must_use]
    pub fn total_size_mb(&self) -> f64 {
        // Allow cast precision loss: file sizes in MB don't need exact precision
        #[allow(clippy::cast_precision_loss)]
        { self.total_size_bytes as f64 / 1024.0 / 1024.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = S3ScanOptions::default();
        assert!(opts.prefix.is_none());
        assert!(opts.extensions.contains(&"tif".to_string()));
    }

    #[test]
    fn test_with_prefix() {
        let opts = S3ScanOptions::with_prefix("data/layers/");
        assert_eq!(opts.prefix, Some("data/layers/".to_string()));
    }

    #[test]
    fn test_options_builder() {
        let opts = S3ScanOptions::default()
            .with_endpoint("http://localhost:9000")
            .with_region("eu-west-1")
            .with_allow_http(true)
            .with_max_objects(100);

        assert_eq!(opts.endpoint_url, Some("http://localhost:9000".to_string()));
        assert_eq!(opts.region, Some("eu-west-1".to_string()));
        assert!(opts.allow_http);
        assert_eq!(opts.max_objects, Some(100));
    }
}

#[cfg(test)]
mod network_tests {
    use super::*;

    /// Network: anonymous scan of a public bucket (needs `skip_signature`, which `scan` now
    /// honours) with concurrent metadata reads.
    #[tokio::test]
    #[ignore = "needs network access"]
    async fn scan_public_bucket_anonymously() {
        let options = S3ScanOptions::with_prefix("2026-09-01_2026-10-01/18SUJ_2026-09-01_2026-10-01/")
            .with_region("us-west-2")
            .with_skip_signature(true)
            .with_max_objects(3);
        let source = S3CogSource::scan("ei-imagery-sentinel2-prd", options).await.unwrap();
        assert!(!source.entries().is_empty());
        assert!(source.entries().len() <= 3);
        let tci = source.entries().iter().find(|e| e.name == "TCI");
        if let Some(tci) = tci {
            assert!(tci.is_tiled);
            assert_eq!(tci.size_bytes, Some(328_038_946));
        }
    }
}
