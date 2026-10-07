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
use object_store::{ClientOptions, GetOptions, GetRange, ObjectMeta, ObjectStore, RetryConfig};
use parking_lot::Mutex;
use tokio::sync::Semaphore;

use crate::async_io::{spawn_io, AsyncRangeReader, IoOptions, SourceChanged, SyncToAsync};
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

/// How reads of a remote object are tied to the version that was opened.
///
/// With [`IfMatch`](Self::IfMatch) every range request after the open carries the object's
/// strong ETag as `If-Match` (or, when the server gives no usable ETag but a `Last-Modified`
/// time, `If-Unmodified-Since`). An overwritten object then answers `412 Precondition Failed`,
/// which surfaces as [`SourceChanged`] instead of bytes of the wrong version. The condition
/// travels with the request, so it costs no extra round trip. Weak ETags (`W/"..."`) are never
/// sent: `If-Match` compares strongly and would always fail.
///
/// The other modes send no conditions; a replaced object is then only noticed when a cached
/// header expires ([`Ttl`](Self::Ttl)), or never ([`Immutable`](Self::Immutable), for objects
/// that are never overwritten, such as date-versioned keys).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Validation {
    /// Conditional requests (default).
    #[default]
    IfMatch,
    /// No conditional requests; cached headers expire after their time to live.
    Ttl,
    /// No conditional requests and cached headers never expire.
    Immutable,
}

/// What a server reported about an object when it was opened: enough to rebuild a reader for it
/// without sending a request (see the header cache).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ObjectIdentity {
    pub size: u64,
    pub etag: Option<String>,
    /// `None` when the server sent no `Last-Modified` (`object_store` reports the epoch then).
    pub last_modified: Option<chrono::DateTime<chrono::Utc>>,
}

impl ObjectIdentity {
    fn from_meta(meta: &ObjectMeta) -> Self {
        Self {
            size: meta.size,
            etag: meta.e_tag.clone(),
            last_modified: (meta.last_modified.timestamp() > 0).then_some(meta.last_modified),
        }
    }

    /// Token that changes with the object's content: its ETag, else `"{size}:{last-modified}"`.
    pub(crate) fn version(&self) -> String {
        self.etag.clone().unwrap_or_else(|| {
            format!("{}:{}", self.size, self.last_modified.map_or(0, |t| t.timestamp()))
        })
    }

    fn precondition(&self, validation: Validation) -> Precondition {
        if validation != Validation::IfMatch {
            return Precondition::None;
        }
        match (&self.etag, self.last_modified) {
            (Some(etag), _) if !etag.starts_with("W/") => Precondition::IfMatch(etag.clone()),
            (_, Some(time)) => Precondition::UnmodifiedSince(time),
            _ => Precondition::None,
        }
    }

    /// Whether reads under `validation` carry a condition that detects a replaced object.
    pub(crate) fn is_validatable(&self, validation: Validation) -> bool {
        !matches!(self.precondition(validation), Precondition::None)
    }
}

/// The condition attached to every range request after the open.
#[derive(Clone)]
enum Precondition {
    None,
    IfMatch(String),
    UnmodifiedSince(chrono::DateTime<chrono::Utc>),
}

impl Precondition {
    fn apply(&self, options: &mut GetOptions) {
        match self {
            Self::None => {}
            Self::IfMatch(etag) => options.if_match = Some(etag.clone()),
            Self::UnmodifiedSince(time) => options.if_unmodified_since = Some(*time),
        }
    }
}

/// The object does not exist (`404`). Returned (instead of a plain message) by the open
/// functions so the header cache can remember it for a short time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceNotFound {
    message: String,
}

impl std::fmt::Display for SourceNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SourceNotFound {}

impl SourceNotFound {
    pub(crate) fn new(message: String) -> Self {
        Self { message }
    }
}

/// `message` as an error: [`SourceNotFound`] if `cause` is a `404`, else a plain message.
fn open_failure(cause: &(dyn std::error::Error + 'static), message: String) -> Box<dyn std::error::Error + Send + Sync> {
    if cause.downcast_ref::<object_store::Error>().is_some_and(|e| matches!(e, object_store::Error::NotFound { .. })) {
        Box::new(SourceNotFound { message })
    } else {
        message.into()
    }
}

/// Where an object lives: the shared client and limiter, the path in the store and the
/// identifier (URL without query).
struct Target {
    entry: Arc<StoreEntry>,
    path: ObjectPath,
    identifier: String,
}

async fn s3_target(config: &S3Config, options: &IoOptions) -> AnyResult<Target> {
    Ok(Target {
        entry: s3_store_entry(config, options).await?,
        path: ObjectPath::from(config.key.as_str()),
        identifier: format!("s3://{}/{}", config.bucket, config.key),
    })
}

/// A query string is sent unchanged with every request (presigned or SAS URLs work); such URLs
/// get a private client rather than a shared one, and the query is left out of the identifier.
fn http_target(source: &str, options: &IoOptions) -> AnyResult<Target> {
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
    Ok(Target { entry, path, identifier })
}

async fn resolve_target(source: &str, options: &IoOptions) -> AnyResult<Target> {
    if source.starts_with("s3://") {
        s3_target(&S3Config::from_url(source)?, options).await
    } else if source.starts_with("http://") || source.starts_with("https://") {
        http_target(source, options)
    } else {
        Err(format!("Not a remote URL (expected s3://, http:// or https://): {source}").into())
    }
}

/// The parts of a remote source's identity that are known without any I/O.
pub(crate) struct SourceKeyParts {
    /// Which server and credentials the object is read through. Excludes the query string.
    pub origin: String,
    /// Object path within the origin
    pub path: String,
    /// URL without query or fragment ([`AsyncRangeReader::identifier`])
    pub identifier: String,
}

pub(crate) fn source_key_parts(source: &str) -> AnyResult<SourceKeyParts> {
    if source.starts_with("s3://") {
        let config = S3Config::from_url(source)?;
        Ok(SourceKeyParts {
            origin: format!(
                "s3|{}|{:?}|{:?}|{:?}|{}|{}",
                config.bucket,
                config.endpoint_url,
                config.access_key_id,
                config.secret_access_key.as_deref().map(fingerprint),
                config.allow_http,
                config.skip_signature,
            ),
            identifier: format!("s3://{}/{}", config.bucket, config.key),
            path: config.key,
        })
    } else if source.starts_with("http://") || source.starts_with("https://") {
        let mut url = url::Url::parse(source)?;
        let host = url.host_str().ok_or("Missing host in URL")?.to_string();
        let origin = format!("http|{}://{}:{:?}", url.scheme(), host, url.port());
        url.set_query(None);
        url.set_fragment(None);
        Ok(SourceKeyParts { origin, path: url.path().to_string(), identifier: url.to_string() })
    } else {
        Err(format!("Not a remote URL (expected s3://, http:// or https://): {source}").into())
    }
}

/// Async range reader over one S3 or HTTP(S) object.
pub struct ObjectStoreRangeReader {
    store: Arc<dyn ObjectStore>,
    limiter: Arc<Semaphore>,
    path: ObjectPath,
    identifier: String,
    identity: ObjectIdentity,
    /// Version token ([`ObjectIdentity::version`])
    version: String,
    /// First bytes of the object, kept from the open (empty for readers rebuilt from a cached
    /// identity); reads inside it are served from memory.
    prefix: Bytes,
    options: IoOptions,
    validation: Validation,
    precondition: Precondition,
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
        Self::open_with_validation(source, options, Validation::default()).await
    }

    /// Like [`open_with_options`](Self::open_with_options) with an explicit [`Validation`] mode.
    ///
    /// # Errors
    /// Returns an error if the URL is invalid or the object cannot be read.
    pub async fn open_with_validation(source: &str, options: &IoOptions, validation: Validation) -> AnyResult<Self> {
        if source.starts_with("s3://") {
            Self::open_s3_with(S3Config::from_url(source)?, options, validation).await
        } else if source.starts_with("http://") || source.starts_with("https://") {
            Self::open_http(source, options, validation).await
        } else {
            Err(format!("Not a remote URL (expected s3://, http:// or https://): {source}").into())
        }
    }

    /// Open an S3 object from an explicit configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is invalid or the object cannot be read.
    pub async fn open_s3(config: S3Config, options: &IoOptions) -> AnyResult<Self> {
        Self::open_s3_with(config, options, Validation::default()).await
    }

    /// Like [`open_s3`](Self::open_s3) with an explicit [`Validation`] mode.
    ///
    /// # Errors
    /// Returns an error if the configuration is invalid or the object cannot be read.
    pub async fn open_s3_with(config: S3Config, options: &IoOptions, validation: Validation) -> AnyResult<Self> {
        let target = s3_target(&config, options).await?;
        Self::open_object(target, options, validation).await.map_err(|e| {
            let not_found = e.downcast_ref::<object_store::Error>().is_some_and(|e| matches!(e, object_store::Error::NotFound { .. }));
            let missing_credentials_hint = !config.skip_signature && config.access_key_id.is_none() && !not_found;
            let hint = if missing_credentials_hint {
                ". No AWS credentials are configured; if this is a public bucket set \
                 AWS_SKIP_SIGNATURE=true (or S3Config::skip_signature) for anonymous access"
            } else {
                ""
            };
            open_failure(&*e, format!("Failed to open s3://{}/{}: {e}{hint}", config.bucket, config.key))
        })
    }

    /// Open an `http://` / `https://` object (see [`http_target`] for query strings).
    async fn open_http(source: &str, options: &IoOptions, validation: Validation) -> AnyResult<Self> {
        let target = http_target(source, options)?;
        let identifier = target.identifier.clone();
        Self::open_object(target, options, validation)
            .await
            .map_err(|e| open_failure(&*e, format!("Failed to open {identifier}: {e}")))
    }

    /// Fetch the prefix (and with it size and version info) of the object.
    async fn open_object(target: Target, options: &IoOptions, validation: Validation) -> AnyResult<Self> {
        let permit = Arc::clone(&target.entry.limiter).acquire_owned().await?;
        let store = Arc::clone(&target.entry.store);
        let request_path = target.path.clone();
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
        Ok(Self::from_identity(target, options, validation, ObjectIdentity::from_meta(&meta), prefix))
    }

    /// A reader for an object whose identity was learned earlier: no request is sent, the
    /// prefix is empty (reads inside it go to the network) and every read is conditioned on the
    /// known version, so a changed object is detected on the first read.
    pub(crate) async fn open_known(
        source: &str,
        options: &IoOptions,
        validation: Validation,
        identity: &ObjectIdentity,
    ) -> AnyResult<Self> {
        let target = resolve_target(source, options).await?;
        Ok(Self::from_identity(target, options, validation, identity.clone(), Bytes::new()))
    }

    fn from_identity(
        target: Target,
        options: &IoOptions,
        validation: Validation,
        identity: ObjectIdentity,
        prefix: Bytes,
    ) -> Self {
        Self {
            store: Arc::clone(&target.entry.store),
            limiter: Arc::clone(&target.entry.limiter),
            path: target.path,
            identifier: target.identifier,
            version: identity.version(),
            precondition: identity.precondition(validation),
            identity,
            prefix,
            options: options.clone(),
            validation,
        }
    }

    /// What the server reported about the object when it was opened.
    pub(crate) fn object_identity(&self) -> &ObjectIdentity {
        &self.identity
    }

    /// The object's ETag, if the server provided one.
    #[must_use]
    pub fn etag(&self) -> Option<&str> {
        self.identity.etag.as_deref()
    }

    /// The object's last-modified time as seconds since the Unix epoch, if the server sent one.
    #[must_use]
    pub fn last_modified_unix(&self) -> Option<i64> {
        self.identity.last_modified.map(|t| t.timestamp())
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
                .filter(|end| *end <= self.identity.size)
                .ok_or_else(|| {
                    format!(
                        "Range {offset}+{len} is outside {} ({} bytes)",
                        self.identifier, self.identity.size
                    )
                })?;
            if end <= self.prefix.len() as u64 {
                // Both bounds are within the prefix, which fits in memory.
                return Ok(self.prefix.slice(offset as usize..end as usize));
            }
            let permit = Arc::clone(&self.limiter).acquire_owned().await?;
            let store = Arc::clone(&self.store);
            let path = self.path.clone();
            let mut get = GetOptions { range: Some(GetRange::Bounded(offset..end)), ..GetOptions::default() };
            self.precondition.apply(&mut get);
            let bytes = spawn_io(async move {
                let _permit = permit;
                store.get_opts(&path, get).await?.bytes().await
            })
            .await?
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                match e {
                    object_store::Error::Precondition { .. } => {
                        Box::new(SourceChanged { identifier: self.identifier.clone() })
                    }
                    e => format!("Reading {offset}+{len} of {}: {e}", self.identifier).into(),
                }
            })?;
            Ok(bytes)
        })
    }

    fn size(&self) -> u64 {
        self.identity.size
    }

    fn identifier(&self) -> &str {
        &self.identifier
    }

    fn version(&self) -> Option<&str> {
        Some(&self.version)
    }

    fn reopen(&self) -> BoxFuture<'_, AnyResult<Arc<dyn AsyncRangeReader>>> {
        Box::pin(async move {
            let target = Target {
                entry: Arc::new(StoreEntry { store: Arc::clone(&self.store), limiter: Arc::clone(&self.limiter) }),
                path: self.path.clone(),
                identifier: self.identifier.clone(),
            };
            let fresh = Self::open_object(target, &self.options, self.validation)
                .await
                .map_err(|e| format!("Failed to reopen {}: {e}", self.identifier))?;
            Ok(Arc::new(fresh) as Arc<dyn AsyncRangeReader>)
        })
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
    create_async_range_reader_with(source, options, Validation::default()).await
}

/// Like [`create_async_range_reader`] with an explicit [`Validation`] mode for remote sources.
///
/// # Errors
/// Returns an error if the source cannot be opened.
pub async fn create_async_range_reader_with(
    source: &str,
    options: &IoOptions,
    validation: Validation,
) -> AnyResult<Arc<dyn AsyncRangeReader>> {
    if source.starts_with("s3://") || source.starts_with("http://") || source.starts_with("https://") {
        Ok(Arc::new(ObjectStoreRangeReader::open_with_validation(source, options, validation).await?))
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

#[cfg(test)]
mod validation_tests {
    use super::*;
    use crate::async_io::is_source_changed;
    use crate::test_support::{ObjectServer, ServedObject};
    use std::time::Duration;

    fn data(n: usize, mul: usize) -> Vec<u8> {
        (0..n).map(|i| (i * mul % 256) as u8).collect()
    }

    fn server(object: ServedObject) -> ObjectServer {
        ObjectServer::start(Some(object), Duration::ZERO)
    }

    async fn open(server: &ObjectServer, validation: Validation) -> ObjectStoreRangeReader {
        let url = format!("{}/obj.bin", server.base());
        ObjectStoreRangeReader::open_with_validation(&url, &IoOptions::default(), validation).await.unwrap()
    }

    #[tokio::test]
    async fn reads_after_the_open_carry_if_match_and_a_replaced_object_is_reported() {
        let server = server(ServedObject::new(data(100_000, 7)).etag(Some("\"v1\"")));
        let reader = open(&server, Validation::IfMatch).await;
        assert_eq!(reader.version(), Some("\"v1\""));
        let first = reader.read_range(50_000, 4096).await.unwrap();
        assert_eq!(&first[..], &data(100_000, 7)[50_000..54_096]);

        let requests = server.requests();
        assert_eq!(requests.len(), 2, "open + one read: {requests:?}");
        assert_eq!(requests[0].if_match, None, "the open has no version to pin yet");
        assert_eq!(requests[1].if_match.as_deref(), Some("\"v1\""));
        assert_eq!((requests[1].path.as_str(), requests[1].range), ("/obj.bin", Some((50_000, 54_095))));

        server.replace_default(Some(ServedObject::new(data(100_000, 11)).etag(Some("\"v2\""))));
        let err = reader.read_range(60_000, 100).await.unwrap_err();
        assert!(is_source_changed(&*err), "{err}");
        let changed = err.downcast_ref::<SourceChanged>().unwrap();
        assert_eq!(changed.identifier, reader.identifier());
        assert_eq!(server.requests().last().unwrap().status, 412);
    }

    #[tokio::test]
    async fn reopen_returns_the_new_version() {
        let server = server(ServedObject::new(data(100_000, 7)).etag(Some("\"v1\"")));
        let reader = open(&server, Validation::IfMatch).await;
        server.replace_default(Some(ServedObject::new(data(120_000, 11)).etag(Some("\"v2\""))));
        assert!(reader.read_range(60_000, 100).await.is_err());

        let fresh = reader.reopen().await.unwrap();
        assert_eq!((fresh.version(), fresh.size()), (Some("\"v2\""), 120_000));
        assert_eq!(&fresh.read_range(60_000, 100).await.unwrap()[..], &data(120_000, 11)[60_000..60_100]);
    }

    #[tokio::test]
    async fn weak_etags_are_never_sent_and_last_modified_is_used_instead() {
        let server = server(ServedObject::new(data(100_000, 7)).etag(Some("W/\"w1\"")).modified(Some(1_800_000_000)));
        let reader = open(&server, Validation::IfMatch).await;
        reader.read_range(50_000, 100).await.unwrap();
        let read = server.requests().pop().unwrap();
        assert_eq!(read.if_match, None);
        assert!(read.if_unmodified_since.is_some(), "{read:?}");

        server.replace_default(Some(ServedObject::new(data(100_000, 11)).etag(Some("W/\"w2\"")).modified(Some(1_800_000_100))));
        let err = reader.read_range(50_000, 100).await.unwrap_err();
        assert!(is_source_changed(&*err), "{err}");
    }

    #[tokio::test]
    async fn sources_without_validators_get_no_conditions() {
        let server = server(ServedObject::new(data(100_000, 7)).etag(None).modified(None));
        let reader = open(&server, Validation::IfMatch).await;
        reader.read_range(50_000, 100).await.unwrap();
        let read = server.requests().pop().unwrap();
        assert_eq!((read.if_match, read.if_unmodified_since), (None, None));
        // Nothing detects the replacement: this is the TTL-only case.
        server.replace_default(Some(ServedObject::new(data(100_000, 11)).etag(None).modified(None)));
        assert_eq!(&reader.read_range(50_000, 100).await.unwrap()[..], &data(100_000, 11)[50_000..50_100]);
    }

    #[tokio::test]
    async fn ttl_and_immutable_modes_send_no_conditions() {
        for validation in [Validation::Ttl, Validation::Immutable] {
            let server = server(ServedObject::new(data(100_000, 7)).etag(Some("\"v1\"")));
            let reader = open(&server, validation).await;
            reader.read_range(50_000, 100).await.unwrap();
            let read = server.requests().pop().unwrap();
            assert_eq!((read.if_match, read.if_unmodified_since), (None, None), "{validation:?}");
        }
    }

    #[tokio::test]
    async fn a_server_that_ignores_conditions_is_not_an_error() {
        let server = server(ServedObject::new(data(100_000, 7)).etag(Some("\"v1\"")));
        let reader = open(&server, Validation::IfMatch).await;
        server.set_honor_conditionals(false);
        server.replace_default(Some(ServedObject::new(data(100_000, 11)).etag(Some("\"v2\""))));
        assert_eq!(&reader.read_range(50_000, 100).await.unwrap()[..], &data(100_000, 11)[50_000..50_100]);
    }

    const HTTPS_URL: &str =
        "https://ei-imagery-sentinel2-prd.s3.us-west-2.amazonaws.com/2026-09-01_2026-10-01/18SUJ_2026-09-01_2026-10-01/TCI.tif";
    const S3_URL: &str = "s3://ei-imagery-sentinel2-prd/2026-09-01_2026-10-01/18SUJ_2026-09-01_2026-10-01/TCI.tif";

    /// A conditional range read with the right ETag succeeds and with a wrong one fails with
    /// `412` ([`object_store::Error::Precondition`]), through both the S3 and the HTTPS store, and
    /// the reader maps it to [`SourceChanged`].
    async fn check_real_object_honours_if_match(reader: ObjectStoreRangeReader) {
        let etag = reader.etag().expect("S3 sends an ETag").to_string();
        assert!(!etag.starts_with("W/"), "{etag}");
        let range = GetOptions { range: Some(GetRange::Bounded(200_000_000..200_000_064)), ..GetOptions::default() };

        let right = GetOptions { if_match: Some(etag.clone()), ..range.clone() };
        let bytes = reader.store.get_opts(&reader.path, right).await.unwrap().bytes().await.unwrap();
        assert_eq!(bytes.len(), 64);

        let wrong = GetOptions { if_match: Some("\"00000000000000000000000000000000-0\"".into()), ..range };
        let err = reader.store.get_opts(&reader.path, wrong).await.unwrap_err();
        assert!(matches!(err, object_store::Error::Precondition { .. }), "{err}");

        // The reader itself sends the same condition on every read beyond its prefix.
        assert_eq!(reader.read_range(200_000_000, 64).await.unwrap(), bytes);
    }

    #[tokio::test]
    #[ignore = "needs network access"]
    async fn real_s3_store_answers_412_to_a_wrong_etag() {
        let mut config = S3Config::from_url(S3_URL).unwrap();
        config.skip_signature = true;
        config.region = Some("us-west-2".to_string());
        check_real_object_honours_if_match(ObjectStoreRangeReader::open_s3(config, &IoOptions::default()).await.unwrap()).await;
    }

    #[tokio::test]
    #[ignore = "needs network access"]
    async fn real_https_store_answers_412_to_a_wrong_etag() {
        check_real_object_honours_if_match(ObjectStoreRangeReader::open(HTTPS_URL).await.unwrap()).await;
    }
}
