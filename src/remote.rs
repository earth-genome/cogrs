//! Remote (S3 / HTTP(S)) async range reader built on `object_store`.
//!
//! One implementation, [`ObjectStoreRangeReader`], serves both `s3://` and `http(s)://`
//! sources. Stores are shared process-wide through a registry keyed by bucket/host plus
//! configuration, so every COG in the same bucket reuses one HTTP connection pool (and its TLS
//! sessions) instead of building a client per open.
//!
//! Opening needs no `HEAD`: a single `GET Range: bytes=0-16383` returns the file prefix (where
//! a COG keeps its header and IFDs) and, through `Content-Range`, the total size, ETag and
//! modification time. That also works with presigned URLs, which are only valid for `GET`.
//! The prefix stays attached to the reader and serves any later read that lies inside it.
//!
//! Every request is spawned on the crate's private I/O runtime and awaited from the caller's
//! runtime. The shared client's pooled connections are therefore driven by tasks that outlive
//! any caller runtime (so runtimes can come and go, e.g. one per test), and dropping a read
//! future aborts the request.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock};

use bytes::Bytes;
use futures::future::BoxFuture;
use object_store::aws::AmazonS3Builder;
use object_store::http::HttpBuilder;
use object_store::path::Path as ObjectPath;
use object_store::{ClientOptions, GetOptions, GetRange, ObjectStore, RetryConfig};
use parking_lot::Mutex;
use tokio::sync::Semaphore;

use crate::async_io::{spawn_io, AsyncRangeReader, IoOptions, SyncToAsync};
use crate::range_reader::LocalRangeReader;
use crate::s3::{resolve_region, S3Config};
use crate::tiff_utils::AnyResult;

/// Number of leading bytes fetched when opening a source.
///
/// Sized to cover the header, the IFD chain and the out-of-line tag values of a typical COG,
/// which are all placed at the start of the file.
pub(crate) const PREFIX_BYTES: u64 = 16 * 1024;

/// A shared `object_store` client plus the in-flight request limiter for it.
struct StoreEntry {
    store: Arc<dyn ObjectStore>,
    limiter: Arc<Semaphore>,
}

static STORES: LazyLock<Mutex<HashMap<String, Arc<StoreEntry>>>> = LazyLock::new(Mutex::default);

fn client_options(options: &IoOptions, allow_http: bool) -> ClientOptions {
    ClientOptions::new()
        .with_connect_timeout(options.connect_timeout)
        .with_timeout(options.request_timeout)
        .with_allow_http(allow_http)
}

fn retry_config(options: &IoOptions) -> RetryConfig {
    RetryConfig {
        max_retries: options.max_retries,
        retry_timeout: options.retry_timeout,
        ..RetryConfig::default()
    }
}

fn fingerprint(value: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

/// Look up `key` in the registry or build and register a store with `build`.
///
/// Every request a store makes is spawned on the crate's private I/O runtime
/// ([`spawn_io`](crate::async_io::spawn_io)), so the store's pooled connections are driven by
/// tasks that outlive any caller runtime. That is what makes one client per bucket or host safe
/// to share process-wide even when callers run on several, possibly short-lived, runtimes.
fn shared_store(
    key: &str,
    options: &IoOptions,
    build: impl FnOnce() -> AnyResult<Arc<dyn ObjectStore>>,
) -> AnyResult<Arc<StoreEntry>> {
    let mut stores = STORES.lock();
    if let Some(entry) = stores.get(key) {
        return Ok(Arc::clone(entry));
    }
    let entry = Arc::new(StoreEntry {
        store: build()?,
        limiter: Arc::new(Semaphore::new(options.max_in_flight_per_store.max(1))),
    });
    stores.insert(key.to_string(), Arc::clone(&entry));
    Ok(entry)
}

/// Number of registered stores whose key starts with `prefix` (tests).
#[cfg(test)]
fn registered_stores_with_prefix(prefix: &str) -> usize {
    STORES.lock().keys().filter(|k| k.starts_with(prefix)).count()
}

/// The shared store (client + request limiter) for the bucket named by `config`, created on first
/// use: explicit config > `AWS_REGION` > `AWS_DEFAULT_REGION` > region detected from the bucket.
async fn s3_store_entry(config: &S3Config, options: &IoOptions) -> AnyResult<Arc<StoreEntry>> {
    let region = resolve_region(&config.bucket, config.region.as_deref(), config.endpoint_url.as_deref()).await;
    let key = format!(
        "s3|{}|{:?}|{:?}|{:?}|{:?}|{}|{}|{:?}",
        config.bucket,
        config.endpoint_url,
        region,
        config.access_key_id,
        config.secret_access_key.as_deref().map(fingerprint),
        config.allow_http,
        config.skip_signature,
        options,
    );
    shared_store(&key, options, || {
        let mut builder = AmazonS3Builder::new()
            .with_bucket_name(&config.bucket)
            .with_client_options(client_options(options, config.allow_http))
            .with_retry(retry_config(options));
        if let Some(region) = &region {
            builder = builder.with_region(region);
        }
        if let Some(endpoint) = &config.endpoint_url {
            builder = builder.with_endpoint(endpoint);
        }
        if let Some(access_key) = &config.access_key_id {
            builder = builder.with_access_key_id(access_key);
        }
        if let Some(secret_key) = &config.secret_access_key {
            builder = builder.with_secret_access_key(secret_key);
        }
        if config.allow_http {
            builder = builder.with_allow_http(true);
        }
        if config.skip_signature {
            builder = builder.with_skip_signature(true);
        }
        Ok(Arc::new(builder.build()?) as Arc<dyn ObjectStore>)
    })
}

/// The shared `object_store` client for `config`'s bucket (used to list objects).
pub(crate) async fn s3_object_store(config: &S3Config, options: &IoOptions) -> AnyResult<Arc<dyn ObjectStore>> {
    Ok(Arc::clone(&s3_store_entry(config, options).await?.store))
}

/// Async range reader over one S3 or HTTP(S) object.
pub struct ObjectStoreRangeReader {
    store: Arc<dyn ObjectStore>,
    limiter: Arc<Semaphore>,
    path: ObjectPath,
    size: u64,
    identifier: String,
    prefix: Bytes,
    etag: Option<String>,
    /// Identity of this version of the object: its ETag, else `"{size}:{last-modified}"`.
    version: String,
    last_modified: Option<i64>,
    options: IoOptions,
}

impl ObjectStoreRangeReader {
    /// Open an `s3://bucket/key` or `http(s)://host/path` object with default [`IoOptions`].
    ///
    /// # Errors
    /// Returns an error if the URL is invalid or the object cannot be read.
    pub async fn open(source: &str) -> AnyResult<Self> {
        Self::open_with_options(source, &IoOptions::default()).await
    }

    /// Like [`open`](Self::open) with explicit [`IoOptions`].
    ///
    /// # Errors
    /// Returns an error if the URL is invalid or the object cannot be read.
    pub async fn open_with_options(source: &str, options: &IoOptions) -> AnyResult<Self> {
        if source.starts_with("s3://") {
            Self::open_s3(S3Config::from_url(source)?, options).await
        } else if source.starts_with("http://") || source.starts_with("https://") {
            Self::open_http(source, options).await
        } else {
            Err(format!("Not a remote URL (expected s3://, http:// or https://): {source}").into())
        }
    }

    /// Open an S3 object from an explicit configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is invalid or the object cannot be read.
    pub async fn open_s3(config: S3Config, options: &IoOptions) -> AnyResult<Self> {
        let entry = s3_store_entry(&config, options).await?;

        let identifier = format!("s3://{}/{}", config.bucket, config.key);
        let path = ObjectPath::from(config.key.as_str());
        Self::open_object(entry, path, identifier, options).await.map_err(|e| {
            let missing_credentials_hint = !config.skip_signature
                && config.access_key_id.is_none()
                && !e.downcast_ref::<object_store::Error>().is_some_and(|e| matches!(e, object_store::Error::NotFound { .. }));
            let hint = if missing_credentials_hint {
                ". No AWS credentials are configured; if this is a public bucket set \
                 AWS_SKIP_SIGNATURE=true (or S3Config::skip_signature) for anonymous access"
            } else {
                ""
            };
            format!("Failed to open s3://{}/{}: {e}{hint}", config.bucket, config.key).into()
        })
    }

    /// Open an `http://` / `https://` object.
    ///
    /// A query string is sent unchanged with every request (presigned or SAS URLs work);
    /// such URLs get a private client rather than a shared one, and the query is left out of
    /// [`identifier`](AsyncRangeReader::identifier).
    async fn open_http(source: &str, options: &IoOptions) -> AnyResult<Self> {
        let url = url::Url::parse(source)?;
        let host = url.host_str().ok_or("Missing host in URL")?.to_string();
        let path = ObjectPath::from_url_path(url.path())?;
        if path.as_ref().is_empty() {
            return Err(format!("Missing object path in URL: {source}").into());
        }
        let mut base = url.clone();
        base.set_path("/");
        base.set_fragment(None);
        let has_query = base.query().is_some();
        let mut identifier = url.clone();
        identifier.set_query(None);
        identifier.set_fragment(None);
        let identifier = identifier.to_string();

        let allow_http = url.scheme() == "http";
        let build = || -> AnyResult<Arc<dyn ObjectStore>> {
            Ok(Arc::new(
                HttpBuilder::new()
                    .with_url(base.as_str())
                    .with_client_options(client_options(options, allow_http))
                    .with_retry(retry_config(options))
                    .build()?,
            ))
        };
        let entry = if has_query {
            Arc::new(StoreEntry {
                store: build()?,
                limiter: Arc::new(Semaphore::new(options.max_in_flight_per_store.max(1))),
            })
        } else {
            let key = format!("http|{}://{}:{:?}|{:?}", url.scheme(), host, url.port(), options);
            shared_store(&key, options, build)?
        };

        Self::open_object(entry, path, identifier.clone(), options)
            .await
            .map_err(|e| format!("Failed to open {identifier}: {e}").into())
    }

    /// Fetch the prefix (and with it size and version info) of `path`.
    async fn open_object(
        entry: Arc<StoreEntry>,
        path: ObjectPath,
        identifier: String,
        options: &IoOptions,
    ) -> AnyResult<Self> {
        let permit = Arc::clone(&entry.limiter).acquire_owned().await?;
        let store = Arc::clone(&entry.store);
        let request_path = path.clone();
        let (meta, prefix) = spawn_io(async move {
            let _permit = permit;
            let result = store
                .get_opts(
                    &request_path,
                    GetOptions { range: Some(GetRange::Bounded(0..PREFIX_BYTES)), ..GetOptions::default() },
                )
                .await?;
            let meta = result.meta.clone();
            let prefix = result.bytes().await?;
            Ok::<_, object_store::Error>((meta, prefix))
        })
        .await??;

        let version = meta.e_tag.clone().unwrap_or_else(|| format!("{}:{}", meta.size, meta.last_modified.timestamp()));
        Ok(Self {
            store: Arc::clone(&entry.store),
            limiter: Arc::clone(&entry.limiter),
            path,
            size: meta.size,
            identifier,
            prefix,
            etag: meta.e_tag,
            version,
            last_modified: Some(meta.last_modified.timestamp()),
            options: options.clone(),
        })
    }

    /// The object's ETag, if the server provided one.
    #[must_use]
    pub fn etag(&self) -> Option<&str> {
        self.etag.as_deref()
    }

    /// The object's last-modified time as seconds since the Unix epoch.
    #[must_use]
    pub fn last_modified_unix(&self) -> Option<i64> {
        self.last_modified
    }
}

impl AsyncRangeReader for ObjectStoreRangeReader {
    fn read_range(&self, offset: u64, len: usize) -> BoxFuture<'_, AnyResult<Bytes>> {
        Box::pin(async move {
            if len == 0 {
                return Ok(Bytes::new());
            }
            let end = offset
                .checked_add(len as u64)
                .filter(|end| *end <= self.size)
                .ok_or_else(|| {
                    format!(
                        "Range {offset}+{len} is outside {} ({} bytes)",
                        self.identifier, self.size
                    )
                })?;
            if end <= self.prefix.len() as u64 {
                // Both bounds are within the prefix, which fits in memory.
                return Ok(self.prefix.slice(offset as usize..end as usize));
            }
            let permit = Arc::clone(&self.limiter).acquire_owned().await?;
            let store = Arc::clone(&self.store);
            let path = self.path.clone();
            let bytes = spawn_io(async move {
                let _permit = permit;
                store.get_range(&path, offset..end).await
            })
            .await?
            .map_err(|e| format!("Reading {offset}+{len} of {}: {e}", self.identifier))?;
            Ok(bytes)
        })
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn identifier(&self) -> &str {
        &self.identifier
    }

    fn version(&self) -> Option<&str> {
        Some(&self.version)
    }

    fn is_local(&self) -> bool {
        false
    }

    fn io_options(&self) -> &IoOptions {
        &self.options
    }
}

/// Create an async reader for `source`: `s3://`, `http(s)://`, or a local path.
///
/// # Errors
/// Returns an error if the source cannot be opened.
pub async fn create_async_range_reader(source: &str, options: &IoOptions) -> AnyResult<Arc<dyn AsyncRangeReader>> {
    if source.starts_with("s3://") || source.starts_with("http://") || source.starts_with("https://") {
        Ok(Arc::new(ObjectStoreRangeReader::open_with_options(source, options).await?))
    } else {
        Ok(Arc::new(SyncToAsync::new(Arc::new(LocalRangeReader::new(source)?))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HTTPS_URL: &str =
        "https://ei-imagery-sentinel2-prd.s3.us-west-2.amazonaws.com/2026-09-01_2026-10-01/18SUJ_2026-09-01_2026-10-01/TCI.tif";
    const S3_URL: &str = "s3://ei-imagery-sentinel2-prd/2026-09-01_2026-10-01/18SUJ_2026-09-01_2026-10-01/TCI.tif";

    fn anonymous_s3() -> S3Config {
        let mut config = S3Config::from_url(S3_URL).unwrap();
        config.skip_signature = true;
        config.region = Some("us-west-2".to_string());
        config
    }

    #[test]
    fn rejects_non_remote_and_malformed_sources() {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            assert!(ObjectStoreRangeReader::open("/some/local/file.tif").await.is_err());
            assert!(ObjectStoreRangeReader::open("https://example.com").await.is_err());
        });
    }

    /// Network: verifies `HttpStore` against the real S3 HTTPS URL, with and without a query
    /// string, and that stores are shared per host.
    #[tokio::test]
    #[ignore = "needs network access"]
    async fn http_store_serves_ranges_with_and_without_query_string() {
        let plain = ObjectStoreRangeReader::open(HTTPS_URL).await.unwrap();
        assert_eq!(plain.size(), 328_038_946);
        assert_eq!(plain.identifier(), HTTPS_URL);
        assert!(plain.etag().is_some());
        // Served from the prefix (no extra request) ...
        assert_eq!(&plain.read_range(0, 4).await.unwrap()[..], b"II*\0");
        // ... and beyond it (a real range request).
        let far = plain.read_range(200_000_000, 4096).await.unwrap();
        assert_eq!(far.len(), 4096);

        let host_prefix = "http|https://ei-imagery-sentinel2-prd.s3.us-west-2.amazonaws.com";
        let stores_before = registered_stores_with_prefix(host_prefix);
        assert_eq!(stores_before, 1, "the plain HTTPS open registered one store for the host");
        let with_query = ObjectStoreRangeReader::open(&format!("{HTTPS_URL}?x=1")).await.unwrap();
        assert_eq!(with_query.size(), plain.size());
        assert_eq!(with_query.identifier(), HTTPS_URL, "query is not part of the identifier");
        assert_eq!(with_query.read_range(200_000_000, 4096).await.unwrap(), far);
        assert_eq!(registered_stores_with_prefix(host_prefix), stores_before, "query-string URLs get a private client");

        // A second open of the same host reuses the registered store.
        let again = ObjectStoreRangeReader::open(HTTPS_URL).await.unwrap();
        assert_eq!(again.read_range(200_000_000, 4096).await.unwrap(), far);
        assert_eq!(registered_stores_with_prefix(host_prefix), stores_before);

        // Merged multi-range fetch over HTTPS.
        let ranges = [200_000_000..200_000_100, 200_000_100..200_000_300, 250_000_000..250_000_064];
        let parts = plain.read_ranges(&ranges).await.unwrap();
        assert_eq!(&parts[0][..], &far[..100]);
        assert_eq!(&parts[1][..], &far[100..300]);
        assert_eq!(parts[2].len(), 64);

        // Out-of-range read is an error, not a short read.
        assert!(plain.read_range(plain.size() - 2, 10).await.is_err());
        // Missing object gives a readable error.
        let err = ObjectStoreRangeReader::open(&format!("{HTTPS_URL}.does-not-exist")).await.err().unwrap();
        assert!(err.to_string().contains("Failed to open"), "{err}");
    }

    /// Network: the same object through `s3://` (anonymous), compared with the HTTPS reader.
    #[tokio::test]
    #[ignore = "needs network access"]
    async fn s3_store_matches_https_store() {
        let s3 = ObjectStoreRangeReader::open_s3(anonymous_s3(), &IoOptions::default()).await.unwrap();
        let https = ObjectStoreRangeReader::open(HTTPS_URL).await.unwrap();
        assert_eq!(s3.size(), https.size());
        assert_eq!(s3.etag(), https.etag());
        assert_eq!(s3.identifier(), S3_URL);
        assert_eq!(
            s3.read_range(123_456_789, 2048).await.unwrap(),
            https.read_range(123_456_789, 2048).await.unwrap()
        );
    }
}

#[cfg(test)]
mod local_server_tests {
    use super::*;
    use crate::test_support::serve_bytes;
    use std::time::Duration;

    fn serve(data: Vec<u8>) -> (String, Arc<Mutex<Vec<String>>>) {
        serve_bytes(data, Duration::ZERO)
    }

    fn data(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 7 % 256) as u8).collect()
    }

    #[tokio::test]
    async fn http_reader_opens_with_one_get_and_serves_prefix_from_memory() {
        let d = data(100_000);
        let (base, log) = serve(d.clone());
        let r = ObjectStoreRangeReader::open(&format!("{base}/dir/with space/file.tif")).await.unwrap();
        assert_eq!(r.size(), 100_000);
        assert_eq!(r.etag(), Some("\"abc\""));
        assert!(r.last_modified_unix().is_some());
        assert_eq!(log.lock().len(), 1, "open is a single request (no HEAD)");
        assert!(log.lock()[0].starts_with("GET /dir/with%20space/file.tif "), "{:?}", log.lock());

        // Inside the prefix: no request. Outside: one request.
        assert_eq!(&r.read_range(10, 100).await.unwrap()[..], &d[10..110]);
        assert_eq!(log.lock().len(), 1);
        assert_eq!(&r.read_range(50_000, 1000).await.unwrap()[..], &d[50_000..51_000]);
        assert_eq!(log.lock().len(), 2);
        // A read straddling the prefix end is fetched whole from the server.
        assert_eq!(&r.read_range(PREFIX_BYTES - 10, 20).await.unwrap()[..], &d[16374..16394]);
        assert_eq!(log.lock().len(), 3);
    }

    #[tokio::test]
    async fn query_string_is_sent_with_every_request() {
        let d = data(100_000);
        let (base, log) = serve(d.clone());
        let r = ObjectStoreRangeReader::open(&format!("{base}/a/b.tif?X-Amz-Signature=s1g%2Bn&x=1")).await.unwrap();
        assert!(!r.identifier().contains('?'));
        r.read_range(60_000, 10).await.unwrap();
        let log = log.lock();
        assert_eq!(log.len(), 2);
        for line in log.iter() {
            assert!(line.starts_with("GET /a/b.tif?X-Amz-Signature=s1g%2Bn&x=1 "), "{line}");
        }
    }

    #[tokio::test]
    async fn out_of_range_reads_are_errors() {
        let (base, _) = serve(data(1000));
        let r = ObjectStoreRangeReader::open(&format!("{base}/f.tif")).await.unwrap();
        assert!(r.read_range(990, 20).await.is_err());
        assert_eq!(r.read_range(990, 10).await.unwrap().len(), 10);
        assert_eq!(r.read_range(5, 0).await.unwrap().len(), 0);
    }
}

#[cfg(test)]
mod runtime_independence_tests {
    use super::*;
    use crate::test_support::serve_bytes;
    use std::time::Duration;

    fn new_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap()
    }

    /// One client is shared by callers on any number of runtimes. Requests are driven by the
    /// crate's own I/O runtime, so a caller's runtime shutting down must not break requests that
    /// other runtimes have in flight on the same pooled connections.
    #[test]
    fn a_shared_client_survives_caller_runtimes_shutting_down() {
        let (base, _log) = serve_bytes(vec![3u8; 200_000], Duration::from_millis(2));
        let url = format!("{base}/stress.bin");

        let steady_url = url.clone();
        let steady = std::thread::spawn(move || {
            new_runtime().block_on(async {
                let reader = ObjectStoreRangeReader::open(&steady_url).await.unwrap();
                for i in 0..300u64 {
                    let bytes = reader.read_range(20_000 + i * 10, 100).await.unwrap();
                    assert_eq!(bytes.len(), 100);
                }
            });
        });

        for _ in 0..30 {
            let rt = new_runtime();
            rt.block_on(async {
                let reader = ObjectStoreRangeReader::open(&url).await.unwrap();
                assert_eq!(reader.read_range(50_000, 1000).await.unwrap().len(), 1000);
            });
            drop(rt);
        }
        steady.join().unwrap();
    }

    /// Dropping a request future cancels the request (the task on the I/O runtime is aborted).
    #[test]
    fn cancelling_a_read_is_safe_and_leaves_the_reader_usable() {
        let (base, _log) = serve_bytes(vec![5u8; 200_000], Duration::from_millis(50));
        new_runtime().block_on(async {
            let reader = ObjectStoreRangeReader::open(&format!("{base}/cancel.bin")).await.unwrap();
            let slow = tokio::time::timeout(Duration::from_millis(5), reader.read_range(100_000, 1000)).await;
            assert!(slow.is_err(), "expected the read to be cancelled");
            assert_eq!(reader.read_range(100_000, 1000).await.unwrap().len(), 1000);
        });
    }
}
