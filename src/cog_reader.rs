//! Pure Rust COG (Cloud Optimized `GeoTIFF`) reader
//!
//! This module implements efficient COG reading following the COG specification:
//! - Reads only the IFD metadata on initialization (typically < 16KB)
//! - Uses range requests for tile data (no full file downloads)
//! - Caches decompressed tiles with LRU eviction
//! - Supports multiple sources: local files, HTTP, S3
//! - Native async I/O: open with [`CogReader::open_async`], read tiles through
//!   `TileExtractor`; the blocking `open`/`read_tile`/`sample` API stays for plain threads
//! - Cheap to clone: parsed metadata is shared behind `Arc`, so it can be cached per source
//!
//! Key optimizations:
//! - Min/max from GDAL statistics tags (no full scan needed)
//! - Data type detection from TIFF tags (no trial-and-error)
//! - CRS detection from `GeoKey` directory
//! - Single transform inversion per tile (not per pixel)
//! - Global LRU tile cache for decompressed data

use crate::async_io::{block_on_io, AsyncRangeReader, AsyncToSync, IoOptions, SyncToAsync};
use crate::range_reader::{create_range_reader, RangeReader};
use crate::remote::create_async_range_reader;
use crate::tile_cache;
use crate::tiff_utils::AnyResult;
use bytes::Bytes;
use std::collections::HashMap;
use std::sync::Arc;

// TIFF tag constants
const TAG_IMAGE_WIDTH: u16 = 256;
const TAG_IMAGE_LENGTH: u16 = 257;
const TAG_BITS_PER_SAMPLE: u16 = 258;
const TAG_COMPRESSION: u16 = 259;
const TAG_SAMPLES_PER_PIXEL: u16 = 277;
const TAG_PREDICTOR: u16 = 317;
const TAG_ROWS_PER_STRIP: u16 = 278;
const TAG_STRIP_OFFSETS: u16 = 273;
const TAG_STRIP_BYTE_COUNTS: u16 = 279;
const TAG_TILE_WIDTH: u16 = 322;
const TAG_TILE_LENGTH: u16 = 323;
const TAG_TILE_OFFSETS: u16 = 324;
const TAG_TILE_BYTE_COUNTS: u16 = 325;
const TAG_SAMPLE_FORMAT: u16 = 339;
const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
const TAG_MODEL_TIEPOINT: u16 = 33922;
const TAG_GEO_KEY_DIRECTORY: u16 = 34735;
const TAG_GDAL_METADATA: u16 = 42112;
const TAG_GDAL_NODATA: u16 = 42113;

// GeoKey constants
const GEO_KEY_GEOGRAPHIC_TYPE: u16 = 2048;
const GEO_KEY_PROJECTED_CRS: u16 = 3072;

// Compression constants
const COMPRESSION_NONE: u16 = 1;
const COMPRESSION_LZW: u16 = 5;
const COMPRESSION_JPEG: u16 = 7;
const COMPRESSION_DEFLATE: u16 = 8;
const COMPRESSION_WEBP: u16 = 50001;
const COMPRESSION_ZSTD: u16 = 50000;

// Sample format constants
const SAMPLE_FORMAT_UINT: u16 = 1;
const SAMPLE_FORMAT_INT: u16 = 2;
const SAMPLE_FORMAT_FLOAT: u16 = 3;

/// Data type detected from TIFF tags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CogDataType {
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
}

impl CogDataType {
    #[must_use] pub fn bytes_per_sample(&self) -> usize {
        match self {
            CogDataType::UInt8 | CogDataType::Int8 => 1,
            CogDataType::UInt16 | CogDataType::Int16 => 2,
            CogDataType::UInt32 | CogDataType::Int32 | CogDataType::Float32 => 4,
            CogDataType::UInt64 | CogDataType::Int64 | CogDataType::Float64 => 8,
        }
    }

    /// Detect data type from TIFF tags
    #[must_use]
    #[allow(clippy::match_same_arms)] // False positive: default fallback patterns have different semantics
    pub fn from_tags(bits_per_sample: u16, sample_format: u16) -> Option<Self> {
        match (sample_format, bits_per_sample) {
            (SAMPLE_FORMAT_UINT, 8) => Some(CogDataType::UInt8),
            (SAMPLE_FORMAT_UINT, 16) => Some(CogDataType::UInt16),
            (SAMPLE_FORMAT_UINT, 32) => Some(CogDataType::UInt32),
            (SAMPLE_FORMAT_UINT, 64) => Some(CogDataType::UInt64),
            (SAMPLE_FORMAT_INT, 8) => Some(CogDataType::Int8),
            (SAMPLE_FORMAT_INT, 16) => Some(CogDataType::Int16),
            (SAMPLE_FORMAT_INT, 32) => Some(CogDataType::Int32),
            (SAMPLE_FORMAT_INT, 64) => Some(CogDataType::Int64),
            (SAMPLE_FORMAT_FLOAT, 32) => Some(CogDataType::Float32),
            (SAMPLE_FORMAT_FLOAT, 64) => Some(CogDataType::Float64),
            // Default to unsigned if sample format not specified
            (_, 8) => Some(CogDataType::UInt8),
            (_, 16) => Some(CogDataType::UInt16),
            (_, 32) => Some(CogDataType::UInt32),
            _ => None,
        }
    }
}

/// Compression method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    None,
    Lzw,
    Jpeg,
    Deflate,
    Zstd,
    Webp,
}

impl Compression {
    #[must_use] pub fn from_tag(value: u16) -> Option<Self> {
        match value {
            COMPRESSION_NONE => Some(Compression::None),
            COMPRESSION_LZW => Some(Compression::Lzw),
            COMPRESSION_JPEG => Some(Compression::Jpeg),
            COMPRESSION_DEFLATE | 32946 => Some(Compression::Deflate), // 32946 is old deflate
            COMPRESSION_ZSTD => Some(Compression::Zstd),
            COMPRESSION_WEBP => Some(Compression::Webp),
            _ => None,
        }
    }
}

/// `GeoTIFF` transform information
#[derive(Debug, Clone)]
pub struct GeoTransform {
    /// Pixel scale (`x_scale`, `y_scale`, `z_scale`)
    pub pixel_scale: Option<[f64; 3]>,
    /// Tiepoint (i, j, k, x, y, z) - maps pixel (i,j,k) to world (x,y,z)
    pub tiepoint: Option<[f64; 6]>,
    /// Whether the dataset uses "Point" registration (pixel centers) vs "Area" (pixel corners)
    /// When true, GDAL applies a half-pixel shift to the geotransform origin
    pub is_point_registered: bool,
}

impl GeoTransform {
    /// Convert pixel coordinates to world coordinates
    #[must_use] pub fn pixel_to_world(&self, px: f64, py: f64) -> Option<(f64, f64)> {
        let scale = self.pixel_scale?;
        let tie = self.tiepoint?;

        // Apply half-pixel shift for Point registration (GDAL convention)
        // When is_point_registered=true, tiepoint refers to pixel center, not corner
        let offset = if self.is_point_registered { 0.5 } else { 0.0 };

        let world_x = tie[3] + (px + offset - tie[0]) * scale[0];
        let world_y = tie[4] - (py + offset - tie[1]) * scale[1]; // Y is typically inverted

        Some((world_x, world_y))
    }

    /// Convert world coordinates to pixel coordinates
    #[must_use] pub fn world_to_pixel(&self, wx: f64, wy: f64) -> Option<(f64, f64)> {
        let scale = self.pixel_scale?;
        let tie = self.tiepoint?;

        if scale[0] == 0.0 || scale[1] == 0.0 {
            return None;
        }

        // Apply half-pixel shift for Point registration (GDAL convention)
        // When is_point_registered=true, we need to shift by +0.5 pixel to match GDAL
        let offset = if self.is_point_registered { 0.5 } else { 0.0 };

        let px = tie[0] + (wx - tie[3]) / scale[0] + offset;
        let py = tie[1] + (tie[4] - wy) / scale[1] + offset; // Y is typically inverted

        Some((px, py))
    }

    /// Get the world extent of the image
    #[must_use] pub fn get_extent(&self, width: usize, height: usize) -> Option<(f64, f64, f64, f64)> {
        let (min_x, max_y) = self.pixel_to_world(0.0, 0.0)?;
        // Safe cast: usize to f64 precision loss is acceptable for image dimensions (typically < 2^24 pixels)
        #[allow(clippy::cast_precision_loss)]
        let (max_x, min_y) = self.pixel_to_world(width as f64, height as f64)?;
        Some((min_x, min_y, max_x, max_y))
    }
}

/// COG metadata - read from IFD without loading tile data
#[derive(Debug, Clone)]
pub struct CogMetadata {
    /// Image dimensions
    pub width: usize,
    pub height: usize,

    /// Tile dimensions (COG requirement)
    pub tile_width: usize,
    pub tile_height: usize,

    /// Number of bands/samples
    pub bands: usize,

    /// Data type
    pub data_type: CogDataType,

    /// Compression method
    pub compression: Compression,

    /// Predictor (1=none, 2=horizontal differencing, 3=floating point)
    pub predictor: u16,

    /// Byte order
    pub little_endian: bool,

    /// Tile byte offsets in the file
    pub tile_offsets: Vec<u64>,

    /// Tile byte counts (compressed sizes)
    pub tile_byte_counts: Vec<u64>,

    /// Number of tiles across
    pub tiles_across: usize,

    /// Number of tiles down
    pub tiles_down: usize,

    /// Whether this is a tiled TIFF (true) or stripped TIFF (false)
    /// Tiled TIFFs are COG-optimized, stripped TIFFs are not
    pub is_tiled: bool,

    /// Geographic transform
    pub geo_transform: GeoTransform,

    /// Detected CRS (EPSG code)
    pub crs_code: Option<i32>,

    /// Min/max values from GDAL statistics (if present)
    pub stats_min: Option<f32>,
    pub stats_max: Option<f32>,

    /// `NoData` value
    pub nodata: Option<f64>,
}

impl CogMetadata {
    /// Check if this appears to be a valid COG (has tiles)
    #[must_use] pub fn is_tiled(&self) -> bool {
        self.tile_width > 0 && self.tile_height > 0
    }

    /// Get tile index for a pixel coordinate
    #[must_use] pub fn tile_index_for_pixel(&self, px: usize, py: usize) -> Option<usize> {
        if px >= self.width || py >= self.height {
            return None;
        }
        let tile_col = px / self.tile_width;
        let tile_row = py / self.tile_height;
        Some(tile_row * self.tiles_across + tile_col)
    }

    /// Get pixel range within a tile
    #[must_use] pub fn pixel_range_in_tile(&self, tile_index: usize) -> (usize, usize, usize, usize) {
        let tile_col = tile_index % self.tiles_across;
        let tile_row = tile_index / self.tiles_across;

        let start_x = tile_col * self.tile_width;
        let start_y = tile_row * self.tile_height;
        let end_x = (start_x + self.tile_width).min(self.width);
        let end_y = (start_y + self.tile_height).min(self.height);

        (start_x, start_y, end_x, end_y)
    }

    /// Get number of valid pixels in a tile (handles edge tiles)
    #[must_use] pub fn tile_pixel_count(&self, tile_index: usize) -> usize {
        let (start_x, start_y, end_x, end_y) = self.pixel_range_in_tile(tile_index);
        (end_x - start_x) * (end_y - start_y) * self.bands
    }
}

/// Overview metadata - subset of `CogMetadata` for overviews
#[derive(Debug, Clone)]
pub struct OverviewMetadata {
    pub width: usize,
    pub height: usize,
    pub tile_width: usize,
    pub tile_height: usize,
    pub tiles_across: usize,
    pub tiles_down: usize,
    pub tile_offsets: Vec<u64>,
    pub tile_byte_counts: Vec<u64>,
    /// Approximate integer scale factor relative to full resolution (2, 4, 8, ...): the x ratio
    /// `full_width / width` rounded to the nearest integer. Informational only; every coordinate
    /// mapping uses the exact [`scale_x`](Self::scale_x) / [`scale_y`](Self::scale_y).
    pub scale: usize,
    /// Exact ratio of full-resolution width to this overview's width (e.g. `10980 / 1373`). An
    /// overview pixel is `scale_x` full-resolution pixels wide, as in GDAL, which derives an
    /// overview's geotransform as `extent / overview size`.
    pub scale_x: f64,
    /// Exact ratio of full-resolution height to this overview's height. Can differ from
    /// `scale_x` when the two dimensions round differently.
    pub scale_y: f64,
}

impl OverviewMetadata {
    /// Get tile index for a pixel coordinate at this overview level
    #[must_use] pub fn tile_index_for_pixel(&self, px: usize, py: usize) -> Option<usize> {
        if px >= self.width || py >= self.height {
            return None;
        }
        let tile_col = px / self.tile_width;
        let tile_row = py / self.tile_height;
        Some(tile_row * self.tiles_across + tile_col)
    }
}

/// Hint for pre-computed overview quality analysis
///
/// This allows callers to skip the expensive runtime analysis by providing
/// a pre-computed value (e.g., from a database).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum OverviewQualityHint {
    /// Compute at runtime (default behavior) - samples tiles to determine quality
    #[default]
    ComputeAtRuntime,
    /// All overviews have sufficient data density
    AllUsable,
    /// No overviews have sufficient data - always use full resolution
    NoneUsable,
    /// Use overviews 0..=n (where n is the minimum usable overview index)
    MinUsable(usize),
}


impl OverviewQualityHint {
    /// Convert from database representation (`Option<i32>`)
    ///
    /// - `None` -> `ComputeAtRuntime` (legacy layers without pre-computed value)
    /// - `Some(-1)` -> `NoneUsable` (force full resolution)
    /// - `Some(-2)` -> `AllUsable` (all overviews are good)
    /// - `Some(n)` where n >= 0 -> MinUsable(n as usize)
    #[must_use]
    pub fn from_db_value(value: Option<i32>) -> Self {
        match value {
            Some(-2) => Self::AllUsable,
            Some(-1) => Self::NoneUsable,
            Some(n) if n >= 0 => {
                // Safe cast: n is validated to be non-negative, and overview indices are always small (<100)
                #[allow(clippy::cast_sign_loss)]
                Self::MinUsable(n as usize)
            }
            None | Some(_) => Self::ComputeAtRuntime, // None or invalid value, fall back to runtime
        }
    }

    /// Convert to database representation (`Option<i32>`)
    #[must_use]
    pub fn to_db_value(&self) -> Option<i32> {
        match self {
            Self::ComputeAtRuntime => None,
            Self::NoneUsable => Some(-1),
            Self::AllUsable => Some(-2),
            Self::MinUsable(n) => {
                // Safe cast: overview indices are always small (<100), well within i32 range
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                Some(*n as i32)
            }
        }
    }
}

/// Identifies one source tile: an overview level (`None` = full resolution) and the tile's
/// index within that level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TileRef {
    pub overview: Option<usize>,
    pub index: usize,
}

/// Where a source tile's compressed bytes live, and its pixel dimensions.
///
/// `len == 0` marks a sparse tile that was never written.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TileSpan {
    pub offset: u64,
    pub len: usize,
    pub width: usize,
    pub height: usize,
}

/// COG Reader - efficient COG access with range requests
///
/// Cloning is cheap: the I/O handles and the parsed metadata are shared (`Arc`), so a clone
/// can be moved into a task or cached per source without copying tile offset tables.
#[derive(Clone)]
pub struct CogReader {
    /// Asynchronous reads (the network path).
    async_io: Arc<dyn AsyncRangeReader>,
    /// Blocking reads for the synchronous API (`read_tile`, `sample`, ...).
    sync_io: Arc<dyn RangeReader>,
    /// Parsed header and full-resolution IFD.
    pub metadata: Arc<CogMetadata>,
    /// Overview levels (sorted by scale factor, smallest to largest)
    pub overviews: Arc<[OverviewMetadata]>,
    /// Minimum usable overview index - overviews beyond this have insufficient data
    /// None means all overviews are usable, Some(n) means only overviews 0..n are usable
    pub min_usable_overview: Option<usize>,
    /// Identity decoded tiles are cached and de-duplicated under: identifier plus version.
    cache_id: Arc<str>,
}

impl CogReader {
    /// Open a COG from any source (local file, HTTP URL, or S3)
    ///
    /// This is the **blocking** entry point, for plain threads and `spawn_blocking`. Remote
    /// sources are driven on a private I/O runtime, so no ambient tokio runtime is needed
    /// (and calling it on an async worker thread blocks that worker but does not deadlock).
    /// From async code use [`CogReader::open_async`].
    ///
    /// # Errors
    /// Returns an error if the source cannot be read, the file is not a valid TIFF/COG,
    /// or required metadata tags are missing or invalid.
    pub fn open(source: &str) -> AnyResult<Self> {
        Self::open_with_hint(source, OverviewQualityHint::ComputeAtRuntime)
    }

    /// Open a COG without blocking the async runtime (local file, HTTP URL, or S3)
    ///
    /// Remote I/O is natively asynchronous: the header and IFDs arrive in one ranged request
    /// and no thread is held while waiting on the network.
    ///
    /// ```rust,no_run
    /// use cogrs::CogReader;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    /// let reader = CogReader::open_async("s3://bucket/path/to/file.tif").await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns an error if the source cannot be read, the file is not a valid TIFF/COG,
    /// or required metadata tags are missing or invalid.
    pub async fn open_async(source: &str) -> AnyResult<Self> {
        Self::open_async_with_hint(source, OverviewQualityHint::ComputeAtRuntime).await
    }

    /// Async counterpart of [`CogReader::open_with_hint`]; see [`CogReader::open_async`].
    ///
    /// # Errors
    /// Returns an error if the source cannot be read, the file is not a valid TIFF/COG,
    /// or required metadata tags are missing or invalid.
    pub async fn open_async_with_hint(source: &str, hint: OverviewQualityHint) -> AnyResult<Self> {
        Self::open_async_with_options(source, hint, &IoOptions::default()).await
    }

    /// Like [`CogReader::open_async_with_hint`] with explicit I/O tuning (concurrency limits,
    /// range coalescing, retries and timeouts); see [`IoOptions`].
    ///
    /// # Errors
    /// Returns an error if the source cannot be read, the file is not a valid TIFF/COG,
    /// or required metadata tags are missing or invalid.
    pub async fn open_async_with_options(
        source: &str,
        hint: OverviewQualityHint,
        options: &IoOptions,
    ) -> AnyResult<Self> {
        let reader = create_async_range_reader(source, options).await?;
        Self::from_async_reader_with_hint(reader, hint).await
    }

    /// Run a synchronous operation on this reader (point queries, tile reads, ...) on tokio's
    /// blocking thread pool, without blocking the async runtime.
    ///
    /// ```rust,no_run
    /// use cogrs::{CogReader, PointQuery};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    /// let reader = CogReader::open_async("s3://bucket/file.tif").await?;
    /// let result = reader.spawn_blocking(|r| r.sample_lonlat(-122.4, 37.8)).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns the operation's error, or an error if the blocking task fails.
    pub async fn spawn_blocking<F, T>(&self, f: F) -> AnyResult<T>
    where
        F: FnOnce(&CogReader) -> AnyResult<T> + Send + 'static,
        T: Send + 'static,
    {
        let reader = self.clone();
        tokio::task::spawn_blocking(move || f(&reader))
            .await
            .map_err(|e| format!("Task join error: {e}"))?
    }

    /// Open a COG with a pre-computed overview quality hint
    ///
    /// Use this when you have pre-computed the overview quality (e.g., stored in a database)
    /// to skip the expensive runtime analysis that samples tiles.
    ///
    /// Blocking, like [`CogReader::open`]; from async code use
    /// [`CogReader::open_async_with_hint`].
    ///
    /// # Errors
    /// Returns an error if the source cannot be read, the file is not a valid TIFF/COG,
    /// or required metadata tags are missing or invalid.
    pub fn open_with_hint(source: &str, hint: OverviewQualityHint) -> AnyResult<Self> {
        if is_remote_source(source) {
            let source = source.to_string();
            block_on_io(async move { Self::open_async_with_hint(&source, hint).await })?
        } else {
            Self::from_reader_with_hint(create_range_reader(source)?, hint)
        }
    }

    /// Open from an existing range reader
    ///
    /// # Errors
    /// Returns an error if the file is not a valid TIFF/COG, or required metadata tags
    /// are missing or invalid.
    pub fn from_reader(reader: Arc<dyn RangeReader>) -> AnyResult<Self> {
        Self::from_reader_with_hint(reader, OverviewQualityHint::ComputeAtRuntime)
    }

    /// Open from an existing asynchronous range reader
    ///
    /// # Errors
    /// Returns an error if the file is not a valid TIFF/COG, or required metadata tags
    /// are missing or invalid, or if reading the IFD data fails.
    pub async fn from_async_reader(reader: Arc<dyn AsyncRangeReader>) -> AnyResult<Self> {
        Self::from_async_reader_with_hint(reader, OverviewQualityHint::ComputeAtRuntime).await
    }

    /// Open from an existing asynchronous range reader with an overview quality hint (see
    /// [`CogReader::from_reader_with_hint`] for the hint values).
    ///
    /// # Errors
    /// Returns an error if the file is not a valid TIFF/COG, or required metadata tags
    /// are missing or invalid, or if reading the IFD data fails.
    pub async fn from_async_reader_with_hint(
        reader: Arc<dyn AsyncRangeReader>,
        hint: OverviewQualityHint,
    ) -> AnyResult<Self> {
        let structure = parse_cog_structure(&*reader).await?;
        let sync_io: Arc<dyn RangeReader> = Arc::new(AsyncToSync::new(Arc::clone(&reader)));
        let mut cog = Self::assemble(reader, sync_io, structure, hint);
        if matches!(hint, OverviewQualityHint::ComputeAtRuntime) {
            cog.min_usable_overview = cog.analyze_overview_quality_async().await;
        }
        Ok(cog)
    }

    /// Build a reader from already-parsed metadata, without any I/O.
    ///
    /// This lets a caller keep the (shared, immutable) `metadata` and `overviews` of a source in
    /// its own cache and attach a fresh reader to them on later requests.
    #[must_use]
    pub fn from_parts(
        reader: Arc<dyn AsyncRangeReader>,
        metadata: Arc<CogMetadata>,
        overviews: Arc<[OverviewMetadata]>,
        min_usable_overview: Option<usize>,
    ) -> Self {
        let sync_io: Arc<dyn RangeReader> = Arc::new(AsyncToSync::new(Arc::clone(&reader)));
        let cache_id = tile_cache::source_id(reader.identifier(), reader.version());
        Self { async_io: reader, sync_io, metadata, overviews, min_usable_overview, cache_id }
    }

    /// Source identifier (path or URL).
    #[must_use]
    pub fn identifier(&self) -> &str {
        self.async_io.identifier()
    }

    /// Identity of the source's decoded tiles: the identifier plus its version (ETag, or size and
    /// modification time), so replacing the object changes it.
    pub(crate) fn cache_id(&self) -> &Arc<str> {
        &self.cache_id
    }

    /// The asynchronous reader tile data is fetched through.
    pub(crate) fn io(&self) -> &Arc<dyn AsyncRangeReader> {
        &self.async_io
    }

    fn assemble(
        async_io: Arc<dyn AsyncRangeReader>,
        sync_io: Arc<dyn RangeReader>,
        structure: CogStructure,
        hint: OverviewQualityHint,
    ) -> Self {
        let CogStructure { metadata, overviews } = structure;
        // Apply the overview quality hint
        // min_usable_overview = Some(n) means overviews 0..=n are usable
        // min_usable_overview = None means NO overviews are usable (force full resolution)
        let min_usable_overview = match hint {
            // All overviews are usable - set to last overview index
            OverviewQualityHint::AllUsable => overviews.len().checked_sub(1),
            // Force full resolution, or computed by the caller afterwards
            OverviewQualityHint::NoneUsable | OverviewQualityHint::ComputeAtRuntime => None,
            OverviewQualityHint::MinUsable(n) => Some(n),
        };
        let cache_id = tile_cache::source_id(async_io.identifier(), async_io.version());
        Self {
            async_io,
            sync_io,
            metadata: Arc::new(metadata),
            overviews: overviews.into(),
            min_usable_overview,
            cache_id,
        }
    }

    /// Open from an existing range reader with a pre-computed overview quality hint
    ///
    /// This is the preferred method when you have overview quality metadata stored
    /// in a database, as it avoids the 100-200ms latency from runtime analysis.
    ///
    /// # Arguments
    /// * `reader` - The range reader for accessing the COG data
    /// * `hint` - Pre-computed overview quality hint:
    ///   - `ComputeAtRuntime`: Analyze overviews at construction (default, ~100-200ms)
    ///   - `AllUsable`: All overviews have sufficient data
    ///   - `NoneUsable`: No overviews are usable, always use full resolution
    ///   - `MinUsable(n)`: Overviews 0..=n are usable
    ///
    /// # Errors
    /// Returns an error if the file is not a valid TIFF/COG, required metadata tags
    /// are missing or invalid, or if reading the IFD data fails.
    pub fn from_reader_with_hint(reader: Arc<dyn RangeReader>, hint: OverviewQualityHint) -> AnyResult<Self> {
        // The parse runs inline on this thread: the reader is synchronous anyway, and the
        // futures it produces are always ready, so no runtime is involved.
        let inline = SyncToAsync::inline(Arc::clone(&reader));
        let structure = futures::executor::block_on(parse_cog_structure(&inline))?;
        let async_io: Arc<dyn AsyncRangeReader> = Arc::new(SyncToAsync::new(Arc::clone(&reader)));
        let mut cog = Self::assemble(async_io, reader, structure, hint);
        if matches!(hint, OverviewQualityHint::ComputeAtRuntime) {
            cog.min_usable_overview = cog.analyze_overview_quality_impl();
        }
        Ok(cog)
    }

    /// Analyze overview quality by sampling tiles to find valid data density
    /// This determines which overviews have enough data to be useful.
    ///
    /// This method is expensive (~100-200ms for S3) because it samples tiles.
    /// Consider using `from_reader_with_hint()` with a pre-computed value instead.
    ///
    /// Returns the result as an `OverviewQualityHint` that can be stored in a database.
    #[must_use]
    pub fn compute_overview_quality_hint(&self) -> OverviewQualityHint {
        if self.overviews.is_empty() {
            return OverviewQualityHint::AllUsable;
        }

        // Run the analysis logic without mutating self
        let result = self.analyze_overview_quality_impl();

        match result {
            None => {
                // No good overview found - check if we have any overviews at all
                // If we do, it means none are usable
                if self.overviews.is_empty() {
                    OverviewQualityHint::AllUsable
                } else {
                    OverviewQualityHint::NoneUsable
                }
            }
            Some(idx) => OverviewQualityHint::MinUsable(idx),
        }
    }

    /// Async counterpart of [`CogReader::compute_overview_quality_hint`]; sample tiles are
    /// fetched concurrently without blocking a thread.
    pub async fn compute_overview_quality_hint_async(&self) -> OverviewQualityHint {
        if self.overviews.is_empty() {
            return OverviewQualityHint::AllUsable;
        }
        match self.analyze_overview_quality_async().await {
            None => OverviewQualityHint::NoneUsable,
            Some(idx) => OverviewQualityHint::MinUsable(idx),
        }
    }

    /// Internal implementation of overview quality analysis
    /// Returns None if all overviews are too sparse, Some(n) for minimum usable index
    fn analyze_overview_quality_impl(&self) -> Option<usize> {
        // For each overview (from smallest/coarsest to largest/finest), check if it has enough
        // data. We sample a few tiles from each overview and check data density.
        for (idx, ovr) in self.overviews.iter().enumerate().rev() {
            let mut total_pixels = 0usize;
            let mut valid_pixels = 0usize;
            for tile_idx in overview_sample_indices(ovr.tile_offsets.len()) {
                if let Ok((data, _)) = self.read_tile_sync(TileRef { overview: Some(idx), index: tile_idx }) {
                    total_pixels += data.len();
                    valid_pixels += valid_sample_count(&data);
                }
            }
            if overview_density(valid_pixels, total_pixels) >= MIN_OVERVIEW_DENSITY {
                // Found a good overview, return it as the minimum usable
                return Some(idx);
            }
        }

        // No good overview found - all are too sparse
        None
    }

    /// Same analysis as [`Self::analyze_overview_quality_impl`], fetching each level's sample
    /// tiles concurrently.
    async fn analyze_overview_quality_async(&self) -> Option<usize> {
        for (idx, ovr) in self.overviews.iter().enumerate().rev() {
            let samples = overview_sample_indices(ovr.tile_offsets.len());
            let reads = futures::future::join_all(
                samples.iter().map(|&i| self.read_tile_async_ref(TileRef { overview: Some(idx), index: i })),
            )
            .await;
            let mut total_pixels = 0usize;
            let mut valid_pixels = 0usize;
            for (data, _) in reads.into_iter().flatten() {
                total_pixels += data.len();
                valid_pixels += valid_sample_count(&data);
            }
            if overview_density(valid_pixels, total_pixels) >= MIN_OVERVIEW_DENSITY {
                return Some(idx);
            }
        }
        None
    }

    /// Find the best overview level for a given source extent size
    ///
    /// Parameters:
    /// - `extent_src_width`: How many source pixels the extent covers at full resolution
    /// - `extent_src_height`: How many source pixels the extent covers at full resolution
    /// - `output_width`: How many pixels we're actually rendering (e.g., 256)
    /// - `output_height`: How many pixels we're actually rendering (e.g., 256)
    ///
    /// Returns None if full resolution should be used
    #[must_use] pub fn best_overview_for_resolution(&self, extent_src_width: usize, extent_src_height: usize) -> Option<usize> {
        // If min_usable_overview is None, ALL overviews are too sparse - always use full resolution
        // This is critical for sparse datasets where even the largest overview has insufficient data
        if self.min_usable_overview.is_none() && !self.overviews.is_empty() {
            return None;
        }

        // Default output tile size
        let output_size = 256.0;

        // Calculate how many source pixels per output pixel we'd need at full res
        // If extent covers 21600 source pixels but we only output 256 pixels, we can use an 84x overview
        // If extent covers 256 source pixels for 256 output, we need full resolution (scale = 1)
        // Safe cast: usize to f64 precision loss acceptable for extent size calculations
        #[allow(clippy::cast_precision_loss)]
        let scale_x = extent_src_width as f64 / output_size;
        #[allow(clippy::cast_precision_loss)]
        let scale_y = extent_src_height as f64 / output_size;
        let needed_scale = scale_x.max(scale_y);

        // If we need close to full resolution (1:1 or less), don't use an overview
        if needed_scale < 1.5 {
            return None;
        }

        // Find the best overview that has enough resolution
        // We want the overview with the largest scale that's still <= needed_scale
        // (i.e., the smallest overview that still has enough detail)
        let mut best_idx = None;
        let mut best_scale = 0.0f64;

        for (idx, ovr) in self.overviews.iter().enumerate() {
            // Skip overviews that have been determined to have insufficient data
            // min_usable_overview = Some(n) means only overviews 0..=n have enough data
            if let Some(min_usable) = self.min_usable_overview
                && idx > min_usable {
                    // This overview is too sparse (beyond the minimum usable level)
                    continue;
                }

            // This overview has 1/scale resolution compared to full
            // We can use it if the overview has at least as many pixels as we need
            // needed_scale = extent_pixels / output_pixels
            // If needed_scale = 84 and overview scale = 64, overview has enough resolution
            // (the coarser axis decides)
            let ovr_scale = ovr.scale_x.max(ovr.scale_y);
            if ovr_scale <= needed_scale && ovr_scale > best_scale {
                best_scale = ovr_scale;
                best_idx = Some(idx);
            }
        }

        best_idx
    }

    /// Location and shape of a source tile. Errors if the overview or tile index is out of
    /// range.
    pub(crate) fn tile_span(&self, tile: TileRef) -> AnyResult<TileSpan> {
        let (offsets, counts, width, height) = if let Some(overview_idx) = tile.overview {
            let ovr = self
                .overviews
                .get(overview_idx)
                .ok_or_else(|| format!("Overview index {overview_idx} out of range"))?;
            (&ovr.tile_offsets, &ovr.tile_byte_counts, ovr.tile_width, ovr.tile_height)
        } else {
            (
                &self.metadata.tile_offsets,
                &self.metadata.tile_byte_counts,
                self.metadata.tile_width,
                self.metadata.tile_height,
            )
        };
        if tile.index >= offsets.len() {
            return Err(format!("Tile index {} out of range (max {})", tile.index, offsets.len()).into());
        }
        // Safe cast: tile byte counts are always < 100MB, well within usize range
        #[allow(clippy::cast_possible_truncation)]
        let len = counts.get(tile.index).copied().unwrap_or(0) as usize;
        Ok(TileSpan { offset: offsets[tile.index], len, width, height })
    }

    /// All-NaN data for a sparse tile.
    pub(crate) fn sparse_tile(&self, span: &TileSpan) -> Arc<Vec<f32>> {
        Arc::new(vec![f32::NAN; span.width * span.height * self.metadata.bands])
    }

    /// Decompress, un-predict and convert a tile's compressed bytes to `f32` samples (CPU only).
    pub(crate) fn decode_tile(&self, span: &TileSpan, compressed: &[u8]) -> AnyResult<Vec<f32>> {
        let meta = &*self.metadata;
        let decompressed = decompress_tile(
            compressed,
            meta.compression,
            span.width,
            span.height,
            meta.bands,
            meta.data_type.bytes_per_sample(),
        )?;
        let unpredicted = apply_predictor(
            &decompressed,
            meta.predictor,
            span.width,
            meta.bands,
            meta.data_type.bytes_per_sample(),
        )?;
        Ok(convert_to_f32(&unpredicted, meta.data_type, meta.little_endian))
    }

    /// Tile from the process-wide decompressed-tile cache.
    pub(crate) fn cached_tile(&self, tile: TileRef) -> Option<Arc<Vec<f32>>> {
        tile_cache::get_shared(&self.cache_id, tile.index, tile.overview)
    }

    pub(crate) fn cache_tile(&self, tile: TileRef, data: Arc<Vec<f32>>) {
        tile_cache::insert_shared(&self.cache_id, tile.index, tile.overview, data);
    }

    /// Read, decode and cache one tile on the calling thread (blocking I/O).
    ///
    /// Returns the data and the compressed bytes fetched (0 for cache hits and sparse tiles).
    pub(crate) fn read_tile_sync(&self, tile: TileRef) -> AnyResult<(Arc<Vec<f32>>, usize)> {
        if let Some(cached) = self.cached_tile(tile) {
            return Ok((cached, 0));
        }
        let span = self.tile_span(tile)?;
        if span.len == 0 {
            return Ok((self.sparse_tile(&span), 0));
        }
        let compressed = self.sync_io.read_range(span.offset, span.len)?;
        let data = Arc::new(self.decode_tile(&span, &compressed)?);
        self.cache_tile(tile, Arc::clone(&data));
        Ok((data, span.len))
    }

    /// Read, decode and cache one tile without blocking: the fetch is awaited (sharing the
    /// request with concurrent callers that need the same tile) and the decode runs on tokio's
    /// blocking pool.
    pub(crate) async fn read_tile_async_ref(&self, tile: TileRef) -> AnyResult<(Arc<Vec<f32>>, usize)> {
        let mut fetched = crate::tile_fetch::fetch_tiles(self, tile.overview, &[tile.index]).await?;
        let data = fetched.tiles.remove(&tile.index).ok_or("tile missing from fetch result")?;
        Ok((data, fetched.bytes_fetched))
    }

    /// Read a tile from a specific overview level
    /// Uses global LRU cache to avoid re-decompressing tiles
    ///
    /// # Errors
    /// Returns an error if the overview or tile index is out of range, if reading tile data fails,
    /// or if decompression fails.
    pub fn read_overview_tile(&self, overview_idx: usize, tile_index: usize) -> AnyResult<Vec<f32>> {
        let (data, _) = self.read_tile_sync(TileRef { overview: Some(overview_idx), index: tile_index })?;
        Ok((*data).clone())
    }

    /// Read a single tile's raw data and decompress
    /// Uses global LRU cache to avoid re-decompressing tiles
    ///
    /// # Errors
    /// Returns an error if the tile index is out of range, if reading tile data fails,
    /// or if decompression fails.
    pub fn read_tile(&self, tile_index: usize) -> AnyResult<Vec<f32>> {
        let (data, _) = self.read_tile_sync(TileRef { overview: None, index: tile_index })?;
        Ok((*data).clone())
    }

    /// Read a single tile and return both data and bytes fetched from source
    /// Returns (`pixel_data`, `bytes_fetched`) where `bytes_fetched` is the compressed size read from source
    /// If tile was cached, `bytes_fetched` is 0 (no network I/O)
    ///
    /// # Errors
    /// Returns an error if the tile index is out of range, if reading tile data fails,
    /// or if decompression fails.
    pub fn read_tile_with_bytes(&self, tile_index: usize) -> AnyResult<(Vec<f32>, usize)> {
        let (data, bytes) = self.read_tile_sync(TileRef { overview: None, index: tile_index })?;
        Ok(((*data).clone(), bytes))
    }

    /// Read an overview tile and return both data and bytes fetched from source
    /// Returns (`pixel_data`, `bytes_fetched`) where `bytes_fetched` is the compressed size read from source
    /// If tile was cached, `bytes_fetched` is 0 (no network I/O)
    ///
    /// # Errors
    /// Returns an error if the overview or tile index is out of range, if reading tile data fails,
    /// or if decompression fails.
    pub fn read_overview_tile_with_bytes(&self, overview_idx: usize, tile_index: usize) -> AnyResult<(Vec<f32>, usize)> {
        let (data, bytes) = self.read_tile_sync(TileRef { overview: Some(overview_idx), index: tile_index })?;
        Ok(((*data).clone(), bytes))
    }

    /// Async counterpart of [`CogReader::read_tile`]: the fetch is awaited and decoding runs on
    /// the blocking pool.
    ///
    /// # Errors
    /// Returns an error if the tile index is out of range, if reading tile data fails,
    /// or if decompression fails.
    pub async fn read_tile_async(&self, tile_index: usize) -> AnyResult<Vec<f32>> {
        let (data, _) = self.read_tile_async_ref(TileRef { overview: None, index: tile_index }).await?;
        Ok((*data).clone())
    }

    /// Async counterpart of [`CogReader::read_overview_tile`].
    ///
    /// # Errors
    /// Returns an error if the overview or tile index is out of range, if reading tile data fails,
    /// or if decompression fails.
    pub async fn read_overview_tile_async(&self, overview_idx: usize, tile_index: usize) -> AnyResult<Vec<f32>> {
        let (data, _) = self
            .read_tile_async_ref(TileRef { overview: Some(overview_idx), index: tile_index })
            .await?;
        Ok((*data).clone())
    }

    /// Sample a single pixel value
    ///
    /// # Errors
    /// Returns an error if reading or decompressing the tile containing the pixel fails.
    pub fn sample(&self, band: usize, x: usize, y: usize) -> AnyResult<Option<f32>> {
        let Some(tile_index) = self.metadata.tile_index_for_pixel(x, y) else {
            return Ok(None);
        };

        let (tile, _) = self.read_tile_sync(TileRef { overview: None, index: tile_index })?;
        Ok(self.value_in_tile(&tile, tile_index, band, x, y))
    }

    /// The sample for `(band, x, y)` within the decoded full-resolution tile `tile_index`
    /// (shared by `sample` and the async point queries).
    pub(crate) fn value_in_tile(&self, tile: &[f32], tile_index: usize, band: usize, x: usize, y: usize) -> Option<f32> {
        let meta = &self.metadata;
        let tile_col = tile_index % meta.tiles_across;
        let tile_row = tile_index / meta.tiles_across;
        let local_x = x - tile_col * meta.tile_width;
        let local_y = y - tile_row * meta.tile_height;
        tile.get((local_y * meta.tile_width + local_x) * meta.bands + band).copied()
    }

    /// Estimate min/max from sampling (when GDAL stats not available)
    ///
    /// For local files: Scans ALL tiles for accurate min/max values
    /// For remote files (S3/HTTP): Samples a few tiles for efficiency
    ///
    /// Use `estimate_min_max_fast()` to always use fast sampling regardless of source.
    ///
    /// # Errors
    /// Returns an error if reading or decompressing tiles fails.
    pub fn estimate_min_max(&self) -> AnyResult<(f32, f32)> {
        // First check for GDAL statistics
        if let (Some(min), Some(max)) = (self.metadata.stats_min, self.metadata.stats_max) {
            return Ok((min, max));
        }

        // For local files, do a full scan for accuracy
        // For remote files, use fast sampling to minimize network requests
        if self.async_io.is_local() {
            self.estimate_min_max_full_scan()
        } else {
            self.estimate_min_max_fast()
        }
    }

    /// Fast min/max estimation - samples only corner and center tiles
    /// Use this for remote files where full scans are expensive
    ///
    /// # Errors
    /// Returns an error if reading or decompressing tiles fails.
    pub fn estimate_min_max_fast(&self) -> AnyResult<(f32, f32)> {
        // First check for GDAL statistics
        if let (Some(min), Some(max)) = (self.metadata.stats_min, self.metadata.stats_max) {
            return Ok((min, max));
        }

        // For files with overviews, sample from the smallest overview (most efficient)
        if !self.overviews.is_empty() {
            let smallest_ovr_idx = self.overviews.len() - 1;
            let ovr = &self.overviews[smallest_ovr_idx];
            let total_tiles = ovr.tile_offsets.len();

            // Sample corner tiles + center tile from smallest overview
            let sample_indices: Vec<usize> = if total_tiles <= 5 {
                (0..total_tiles).collect()
            } else {
                vec![
                    0,
                    ovr.tiles_across.saturating_sub(1),
                    total_tiles / 2,
                    total_tiles.saturating_sub(ovr.tiles_across),
                    total_tiles.saturating_sub(1),
                ]
            };

            return self.scan_tiles_for_minmax(&sample_indices, Some(smallest_ovr_idx));
        }

        // No overviews - sample from full resolution tiles
        let total_tiles = self.metadata.tile_offsets.len();
        let sample_indices: Vec<usize> = if total_tiles <= 5 {
            (0..total_tiles).collect()
        } else {
            vec![
                0,                                          // Top-left
                self.metadata.tiles_across.saturating_sub(1), // Top-right
                total_tiles / 2,                            // Center
                total_tiles.saturating_sub(self.metadata.tiles_across), // Bottom-left
                total_tiles.saturating_sub(1),              // Bottom-right
            ]
        };

        self.scan_tiles_for_minmax(&sample_indices, None)
    }

    /// Full scan min/max estimation - reads ALL tiles
    /// Use this for local files where disk I/O is fast
    fn estimate_min_max_full_scan(&self) -> AnyResult<(f32, f32)> {
        // For files with overviews, scan the smallest overview (much faster)
        if !self.overviews.is_empty() {
            let smallest_ovr_idx = self.overviews.len() - 1;
            let ovr = &self.overviews[smallest_ovr_idx];
            let all_indices: Vec<usize> = (0..ovr.tile_offsets.len()).collect();
            return self.scan_tiles_for_minmax(&all_indices, Some(smallest_ovr_idx));
        }

        // No overviews - must scan full resolution
        let all_indices: Vec<usize> = (0..self.metadata.tile_offsets.len()).collect();
        self.scan_tiles_for_minmax(&all_indices, None)
    }

    /// Helper to scan specific tiles for min/max values
    fn scan_tiles_for_minmax(&self, indices: &[usize], overview_idx: Option<usize>) -> AnyResult<(f32, f32)> {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let nodata = self.metadata.nodata;

        for &tile_idx in indices {
            let tile_data = if let Some(ovr_idx) = overview_idx {
                self.read_overview_tile(ovr_idx, tile_idx)?
            } else {
                self.read_tile(tile_idx)?
            };

            for &val in &tile_data {
                // Skip NaN and nodata values
                if val.is_nan() {
                    continue;
                }
                if let Some(nd) = nodata
                    && (f64::from(val) - nd).abs() < 0.001 {
                        continue;
                    }
                if val < min {
                    min = val;
                }
                if val > max {
                    max = val;
                }
            }
        }

        if min.is_infinite() || max.is_infinite() {
            Ok((0.0, 1.0)) // Fallback
        } else {
            Ok((min, max))
        }
    }

    /// True if the source is a network resource (S3 or HTTP).
    #[must_use]
    pub fn is_remote(&self) -> bool {
        !self.async_io.is_local()
    }
}

// ============================================================================
// Helper functions for reading TIFF data
// ============================================================================

#[inline]
fn read_u16(bytes: &[u8], little_endian: bool) -> u16 {
    if little_endian {
        u16::from_le_bytes([bytes[0], bytes[1]])
    } else {
        u16::from_be_bytes([bytes[0], bytes[1]])
    }
}

#[inline]
fn read_u32(bytes: &[u8], little_endian: bool) -> u32 {
    if little_endian {
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    } else {
        u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

#[inline]
fn read_u64(bytes: &[u8], little_endian: bool) -> u64 {
    if little_endian {
        u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    } else {
        u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }
}

#[inline]
fn read_f64(bytes: &[u8], little_endian: bool) -> f64 {
    if little_endian {
        f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    } else {
        f64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }
}

/// True for sources that are fetched over the network (`s3://`, `http://`, `https://`).
fn is_remote_source(source: &str) -> bool {
    source.starts_with("s3://") || source.starts_with("http://") || source.starts_with("https://")
}

/// Minimum fraction of valid samples for an overview to count as usable.
///
/// 5% is aggressive but ensures good visual results for sparse data: for a file with 6% valid
/// data at full resolution, overviews with <5% are significantly degraded. The trade-off is
/// that sparse datasets read more tiles at low zoom, but the visual quality improvement is
/// dramatic (see barley crop data as example).
const MIN_OVERVIEW_DENSITY: f64 = 0.05;

/// Tiles sampled from an overview to judge its data density: first, middle and last.
fn overview_sample_indices(num_tiles: usize) -> Vec<usize> {
    if num_tiles <= 3 {
        (0..num_tiles).collect()
    } else {
        vec![0, num_tiles / 2, num_tiles - 1]
    }
}

/// Samples that are neither NaN nor zero.
fn valid_sample_count(data: &[f32]) -> usize {
    data.iter().filter(|v| !v.is_nan() && **v != 0.0).count()
}

fn overview_density(valid_pixels: usize, total_pixels: usize) -> f64 {
    if total_pixels > 0 {
        // Safe cast: usize to f64 precision loss acceptable for pixel counts (ratios still accurate)
        #[allow(clippy::cast_precision_loss)]
        let density = valid_pixels as f64 / total_pixels as f64;
        density
    } else {
        0.0
    }
}

/// What `open` learns from the file structure (header and IFD chain), before any overview
/// quality hint is applied.
struct CogStructure {
    metadata: CogMetadata,
    overviews: Vec<OverviewMetadata>,
}

/// One IFD's entries (value bytes not yet fetched) and the offset of the next IFD.
struct Ifd {
    tags: HashMap<u16, IfdEntry>,
    next: u32,
}

/// Read the TIFF header, the full-resolution IFD and the overview IFD chain.
///
/// IFD tables are read first; the (possibly large) tile offset/byte-count arrays and other tag
/// values are then fetched concurrently, so a COG with many overviews costs one round trip
/// for them rather than one per array.
async fn parse_cog_structure(io: &dyn AsyncRangeReader) -> AnyResult<CogStructure> {
    // Read header to get IFD offset and byte order
    let header_bytes = io.read_range(0, 8).await?;

    let little_endian = match &header_bytes[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return Err("Invalid TIFF signature".into()),
    };

    let version = read_u16(&header_bytes[2..4], little_endian);
    if version != 42 {
        return Err(format!("Invalid TIFF version: {version}").into());
    }

    let ifd_offset = read_u32(&header_bytes[4..8], little_endian);
    let file_size = io.size();

    let first = read_ifd(io, u64::from(ifd_offset), file_size, little_endian).await?;

    // Overview IFDs (subsequent IFDs in the chain)
    let walk_chain = async {
        let mut headers: Vec<(OverviewHeader, HashMap<u16, IfdEntry>)> = Vec::new();
        let mut next = first.next;
        while next != 0 {
            let ifd = read_ifd(io, u64::from(next), file_size, little_endian).await?;
            let Ok(header) = parse_overview_header(&ifd.tags, little_endian) else {
                break;
            };
            headers.push((header, ifd.tags));
            next = ifd.next;

            // Safety limit - COGs typically have at most 10 overviews
            if headers.len() > 10 {
                break;
            }
        }
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(headers)
    };
    let (metadata, headers) = futures::try_join!(parse_ifd(&first.tags, io, little_endian), walk_chain)?;

    let arrays = futures::future::join_all(
        headers
            .iter()
            .map(|(header, tags)| read_overview_arrays(io, tags, header.tiles_across * header.tiles_down, little_endian)),
    )
    .await;

    // An overview that cannot be parsed ends the chain (it and the ones after it are dropped).
    let (full_width, full_height) = (metadata.width, metadata.height);
    let mut overviews = Vec::with_capacity(headers.len());
    for ((header, _), arrays) in headers.into_iter().zip(arrays) {
        let Ok((tile_offsets, tile_byte_counts)) = arrays else {
            break;
        };
        // Exact per-axis ratios, as GDAL derives an overview's geotransform (extent / size):
        // 10980 / 1373 is 7.997 (a rounded-up overview of a /8 level), not 7.
        #[allow(clippy::cast_precision_loss)]
        let (scale_x, scale_y) = (full_width as f64 / header.width as f64, full_height as f64 / header.height as f64);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let approx_scale = scale_x.round() as usize;
        overviews.push(OverviewMetadata {
            width: header.width,
            height: header.height,
            tile_width: header.tile_width,
            tile_height: header.tile_height,
            tiles_across: header.tiles_across,
            tiles_down: header.tiles_down,
            tile_offsets,
            tile_byte_counts,
            scale: approx_scale,
            scale_x,
            scale_y,
        });
    }

    Ok(CogStructure { metadata, overviews })
}

/// Read one IFD table (entries and next-IFD offset) at `offset`.
async fn read_ifd(io: &dyn AsyncRangeReader, offset: u64, file_size: u64, little_endian: bool) -> AnyResult<Ifd> {
    // Read IFD entries - estimate size based on typical COG (usually < 4KB)
    // Clamp to available bytes if IFD is near end of file
    // Safe cast: clamped to 4096, well within usize range on all platforms
    #[allow(clippy::cast_possible_truncation)]
    let size = file_size.saturating_sub(offset).min(4096) as usize;
    if size < 2 {
        return Err(format!("IFD offset {offset} lies outside the file ({file_size} bytes)").into());
    }
    let ifd_bytes = io.read_range(offset, size).await?;
    let entry_count = read_u16(&ifd_bytes[0..2], little_endian) as usize;

    // Parse all IFD entries into a map
    let mut tags: HashMap<u16, IfdEntry> = HashMap::new();

    for i in 0..entry_count {
        let entry_offset = 2 + i * 12;
        if entry_offset + 12 > ifd_bytes.len() {
            break;
        }

        let tag = read_u16(&ifd_bytes[entry_offset..entry_offset + 2], little_endian);
        let field_type = read_u16(&ifd_bytes[entry_offset + 2..entry_offset + 4], little_endian);
        let count = read_u32(&ifd_bytes[entry_offset + 4..entry_offset + 8], little_endian);
        let value_offset = read_u32(&ifd_bytes[entry_offset + 8..entry_offset + 12], little_endian);

        tags.insert(
            tag,
            IfdEntry {
                field_type,
                count,
                value_offset,
                raw_bytes: [
                    ifd_bytes[entry_offset + 8],
                    ifd_bytes[entry_offset + 9],
                    ifd_bytes[entry_offset + 10],
                    ifd_bytes[entry_offset + 11],
                ],
            },
        );
    }

    // The next IFD offset is right after all entries
    let next_ifd_pos = 2 + entry_count * 12;
    let next = if next_ifd_pos + 4 <= ifd_bytes.len() {
        read_u32(&ifd_bytes[next_ifd_pos..next_ifd_pos + 4], little_endian)
    } else {
        0
    };

    Ok(Ifd { tags, next })
}

/// Parse tile or strip layout from IFD tags
///
/// Returns (`tile_width`, `tile_height`, `tiles_across`, `tiles_down`, `is_tiled`, `tile_offsets`, `tile_byte_counts`)
#[allow(clippy::type_complexity)] // Return tuple is clear from context and used locally
async fn parse_tile_layout(
    tags: &HashMap<u16, IfdEntry>,
    io: &dyn AsyncRangeReader,
    little_endian: bool,
    width: usize,
    height: usize,
) -> AnyResult<(usize, usize, usize, usize, bool, Vec<u64>, Vec<u64>)> {
    let has_tile_tags = tags.contains_key(&TAG_TILE_OFFSETS);
    let has_strip_tags = tags.contains_key(&TAG_STRIP_OFFSETS);
    let is_tiled = has_tile_tags;

    if is_tiled {
        // Tiled TIFF (COG-optimized)
        // Safe casts: tile dimensions are always < 10000, well within u32/usize range
        #[allow(clippy::cast_possible_truncation)]
        let tw = get_tag_value(tags, TAG_TILE_WIDTH, little_endian).unwrap_or(width as u32) as usize;
        #[allow(clippy::cast_possible_truncation)]
        let th = get_tag_value(tags, TAG_TILE_LENGTH, little_endian).unwrap_or(height as u32) as usize;
        let ta = width.div_ceil(tw);
        let td = height.div_ceil(th);
        let total_tiles = ta * td;

        let (offsets, byte_counts) = futures::try_join!(
            read_tag_array_u64(tags, TAG_TILE_OFFSETS, io, little_endian, total_tiles),
            read_tag_array_u64(tags, TAG_TILE_BYTE_COUNTS, io, little_endian, total_tiles),
        )?;

        Ok((tw, th, ta, td, is_tiled, offsets, byte_counts))
    } else if has_strip_tags {
        // Stripped TIFF (not COG-optimized)
        // Treat strips as "tiles" that span the full image width
        // Safe cast: rows_per_strip is always < image height (<100k), well within usize range
        #[allow(clippy::cast_possible_truncation)]
        let rows_per_strip = get_tag_value(tags, TAG_ROWS_PER_STRIP, little_endian)
            .unwrap_or(height as u32) as usize;
        let tw = width; // Strip width = image width
        let th = rows_per_strip;
        let ta = 1; // Only 1 "tile" across (strips span full width)
        let td = height.div_ceil(rows_per_strip);
        let total_strips = td;

        let (offsets, byte_counts) = futures::try_join!(
            read_tag_array_u64(tags, TAG_STRIP_OFFSETS, io, little_endian, total_strips),
            read_tag_array_u64(tags, TAG_STRIP_BYTE_COUNTS, io, little_endian, total_strips),
        )?;

        Ok((tw, th, ta, td, false, offsets, byte_counts))
    } else {
        Err("TIFF has neither tile nor strip tags".into())
    }
}

/// Parse the full-resolution IFD and extract all COG metadata
async fn parse_ifd(
    tags: &HashMap<u16, IfdEntry>,
    io: &dyn AsyncRangeReader,
    little_endian: bool,
) -> AnyResult<CogMetadata> {
    // Extract required tags
    let width = get_tag_value(tags, TAG_IMAGE_WIDTH, little_endian)
        .ok_or("Missing ImageWidth tag")? as usize;
    let height = get_tag_value(tags, TAG_IMAGE_LENGTH, little_endian)
        .ok_or("Missing ImageLength tag")? as usize;

    // Safe casts: these tag values are small constants (<100), well within u16/usize range
    #[allow(clippy::cast_possible_truncation)]
    let bits_per_sample = get_tag_value(tags, TAG_BITS_PER_SAMPLE, little_endian).unwrap_or(8) as u16;
    #[allow(clippy::cast_possible_truncation)]
    let sample_format = get_tag_value(tags, TAG_SAMPLE_FORMAT, little_endian).unwrap_or(1) as u16;
    #[allow(clippy::cast_possible_truncation)]
    let bands = get_tag_value(tags, TAG_SAMPLES_PER_PIXEL, little_endian).unwrap_or(1) as usize;
    #[allow(clippy::cast_possible_truncation)]
    let compression_val = get_tag_value(tags, TAG_COMPRESSION, little_endian).unwrap_or(1) as u16;
    #[allow(clippy::cast_possible_truncation)]
    let predictor = get_tag_value(tags, TAG_PREDICTOR, little_endian).unwrap_or(1) as u16;

    let data_type = CogDataType::from_tags(bits_per_sample, sample_format)
        .ok_or_else(|| format!("Unsupported data type: bits={bits_per_sample}, format={sample_format}"))?;

    let compression = Compression::from_tag(compression_val)
        .ok_or_else(|| format!("Unsupported compression: {compression_val}"))?;

    // Everything below reads tag values that may live outside the IFD table; fetch them
    // concurrently. Order of the results matches the order of the calls.
    let (layout, pixel_scale, tiepoint, geokeys, gdal_metadata, nodata) = futures::try_join!(
        parse_tile_layout(tags, io, little_endian, width, height),
        read_tag_f64_array(tags, TAG_MODEL_PIXEL_SCALE, io, little_endian, 3),
        read_tag_f64_array(tags, TAG_MODEL_TIEPOINT, io, little_endian, 6),
        read_geokey_directory(tags, io),
        read_gdal_metadata_info(tags, io),
        read_gdal_nodata(tags, io),
    )?;
    let (tile_width, tile_height, tiles_across, tiles_down, is_tiled, tile_offsets, tile_byte_counts) = layout;

    // Read CRS from GeoKey directory
    let crs_code = crs_from_geokeys(geokeys.as_deref(), little_endian);

    // Check if pixels are Point registered (from GTRasterTypeGeoKey or GDAL metadata)
    // GeoKey takes precedence as it's the GeoTIFF standard way
    let is_point_from_geokey = raster_type_is_point(geokeys.as_deref(), little_endian);

    // GDAL metadata (stats and AREA_OR_POINT fallback)
    let (stats_min, stats_max, is_point_from_gdal) = gdal_metadata;

    // Use GeoKey value if available, otherwise fall back to GDAL metadata
    let is_point_registered = is_point_from_geokey || is_point_from_gdal;

    let geo_transform = GeoTransform {
        pixel_scale: pixel_scale.map(|v| [v[0], v[1], v[2]]),
        tiepoint: tiepoint.map(|v| [v[0], v[1], v[2], v[3], v[4], v[5]]),
        is_point_registered,
    };

    Ok(CogMetadata {
        width,
        height,
        tile_width,
        tile_height,
        bands,
        data_type,
        compression,
        predictor,
        little_endian,
        tile_offsets,
        tile_byte_counts,
        tiles_across,
        tiles_down,
        is_tiled,
        geo_transform,
        crs_code,
        stats_min,
        stats_max,
        nodata,
    })
}

/// Dimensions and tiling of an overview IFD (simpler than full IFD parsing)
struct OverviewHeader {
    width: usize,
    height: usize,
    tile_width: usize,
    tile_height: usize,
    tiles_across: usize,
    tiles_down: usize,
}

fn parse_overview_header(tags: &HashMap<u16, IfdEntry>, little_endian: bool) -> AnyResult<OverviewHeader> {
    // Extract dimensions and tile info
    let width = get_tag_value(tags, TAG_IMAGE_WIDTH, little_endian)
        .ok_or("Overview missing ImageWidth tag")? as usize;
    let height = get_tag_value(tags, TAG_IMAGE_LENGTH, little_endian)
        .ok_or("Overview missing ImageLength tag")? as usize;

    let tile_width = get_tag_value(tags, TAG_TILE_WIDTH, little_endian)
        .ok_or("Overview missing TileWidth tag")? as usize;
    let tile_height = get_tag_value(tags, TAG_TILE_LENGTH, little_endian)
        .ok_or("Overview missing TileLength tag")? as usize;
    if width == 0 || height == 0 || tile_width == 0 || tile_height == 0 {
        return Err("Overview has a zero dimension".into());
    }

    Ok(OverviewHeader {
        width,
        height,
        tile_width,
        tile_height,
        tiles_across: width.div_ceil(tile_width),
        tiles_down: height.div_ceil(tile_height),
    })
}

/// Read an overview's tile offset and byte-count arrays.
async fn read_overview_arrays(
    io: &dyn AsyncRangeReader,
    tags: &HashMap<u16, IfdEntry>,
    total_tiles: usize,
    little_endian: bool,
) -> AnyResult<(Vec<u64>, Vec<u64>)> {
    futures::try_join!(
        read_tag_array_u64(tags, TAG_TILE_OFFSETS, io, little_endian, total_tiles),
        read_tag_array_u64(tags, TAG_TILE_BYTE_COUNTS, io, little_endian, total_tiles),
    )
}

struct IfdEntry {
    field_type: u16,
    count: u32,
    value_offset: u32,
    raw_bytes: [u8; 4],
}

fn get_tag_value(tags: &HashMap<u16, IfdEntry>, tag: u16, little_endian: bool) -> Option<u32> {
    let entry = tags.get(&tag)?;
    let type_size = match entry.field_type {
        1 => 1, // BYTE
        3 => 2, // SHORT
        4 => 4, // LONG
        _ => return None,
    };

    if entry.count == 1 && type_size <= 4 {
        // Value is inline
        match entry.field_type {
            1 => Some(u32::from(entry.raw_bytes[0])),
            3 => Some(u32::from(read_u16(&entry.raw_bytes, little_endian))),
            4 => Some(read_u32(&entry.raw_bytes, little_endian)),
            _ => None,
        }
    } else {
        None // Would need to read from offset
    }
}

/// Bytes of a tag value: inline in the IFD entry when it fits in 4 bytes, else fetched.
async fn tag_value_bytes(entry: &IfdEntry, total_bytes: usize, io: &dyn AsyncRangeReader) -> AnyResult<Bytes> {
    if total_bytes <= 4 {
        Ok(Bytes::copy_from_slice(&entry.raw_bytes[..total_bytes]))
    } else {
        io.read_range(u64::from(entry.value_offset), total_bytes).await
    }
}

async fn read_tag_array_u64(
    tags: &HashMap<u16, IfdEntry>,
    tag: u16,
    io: &dyn AsyncRangeReader,
    little_endian: bool,
    expected_count: usize,
) -> AnyResult<Vec<u64>> {
    let entry = tags.get(&tag).ok_or_else(|| format!("Missing tag {tag}"))?;

    let type_size = match entry.field_type {
        3 => 2, // SHORT
        4 => 4, // LONG
        16 => 8, // LONG8
        _ => return Err(format!("Unsupported type {} for tag {}", entry.field_type, tag).into()),
    };

    let total_bytes = entry.count as usize * type_size;
    let raw_bytes = tag_value_bytes(entry, total_bytes, io).await?;

    let mut values = Vec::with_capacity(entry.count as usize);
    for i in 0..entry.count as usize {
        let offset = i * type_size;
        let value = match entry.field_type {
            3 => u64::from(read_u16(&raw_bytes[offset..], little_endian)),
            4 => u64::from(read_u32(&raw_bytes[offset..], little_endian)),
            16 => read_u64(&raw_bytes[offset..], little_endian),
            _ => 0,
        };
        values.push(value);
    }

    // Pad with zeros if we got fewer than expected
    while values.len() < expected_count {
        values.push(0);
    }

    Ok(values)
}

async fn read_tag_f64_array(
    tags: &HashMap<u16, IfdEntry>,
    tag: u16,
    io: &dyn AsyncRangeReader,
    little_endian: bool,
    min_count: usize,
) -> AnyResult<Option<Vec<f64>>> {
    let Some(entry) = tags.get(&tag) else {
        return Ok(None);
    };

    if entry.field_type != 12 {
        // DOUBLE
        return Ok(None);
    }

    if (entry.count as usize) < min_count {
        return Ok(None);
    }

    let total_bytes = entry.count as usize * 8;
    let raw_bytes = io.read_range(u64::from(entry.value_offset), total_bytes).await?;

    let mut values = Vec::with_capacity(entry.count as usize);
    for i in 0..entry.count as usize {
        let offset = i * 8;
        values.push(read_f64(&raw_bytes[offset..], little_endian));
    }

    Ok(Some(values))
}

/// GeoKey constants
const GEO_KEY_RASTER_TYPE: u16 = 1025;  // GTRasterTypeGeoKey: 1=PixelIsArea, 2=PixelIsPoint

/// CRS code from the `GeoKey` directory (ProjectedCSTypeGeoKey or GeographicTypeGeoKey)
fn crs_from_geokeys(raw_bytes: Option<&[u8]>, little_endian: bool) -> Option<i32> {
    let raw_bytes = raw_bytes?;

    if raw_bytes.len() < 8 {
        return None;
    }

    let num_keys = read_u16(&raw_bytes[6..8], little_endian) as usize;

    for i in 0..num_keys {
        let offset = 8 + i * 8;
        if offset + 8 > raw_bytes.len() {
            break;
        }

        let key_id = read_u16(&raw_bytes[offset..], little_endian);
        let _tiff_tag_location = read_u16(&raw_bytes[offset + 2..], little_endian);
        let _count = read_u16(&raw_bytes[offset + 4..], little_endian);
        let value = read_u16(&raw_bytes[offset + 6..], little_endian);

        // Check for ProjectedCSTypeGeoKey (3072) or GeographicTypeGeoKey (2048)
        if key_id == GEO_KEY_PROJECTED_CRS && value > 0 {
            return Some(i32::from(value));
        }
        if key_id == GEO_KEY_GEOGRAPHIC_TYPE && value > 0 {
            return Some(i32::from(value));
        }
    }

    None
}

/// Read GTRasterTypeGeoKey to determine if pixels are Point or Area registered
/// Returns true if PixelIsPoint (value = 2), false otherwise (Area or missing)
fn raster_type_is_point(raw_bytes: Option<&[u8]>, little_endian: bool) -> bool {
    let Some(raw_bytes) = raw_bytes else {
        return false;
    };

    if raw_bytes.len() < 8 {
        return false;
    }

    let num_keys = read_u16(&raw_bytes[6..8], little_endian) as usize;

    for i in 0..num_keys {
        let offset = 8 + i * 8;
        if offset + 8 > raw_bytes.len() {
            break;
        }

        let key_id = read_u16(&raw_bytes[offset..], little_endian);
        let tiff_tag_location = read_u16(&raw_bytes[offset + 2..], little_endian);
        let _count = read_u16(&raw_bytes[offset + 4..], little_endian);
        let value = read_u16(&raw_bytes[offset + 6..], little_endian);

        // GTRasterTypeGeoKey (1025): 1 = PixelIsArea, 2 = PixelIsPoint
        if key_id == GEO_KEY_RASTER_TYPE && tiff_tag_location == 0 {
            return value == 2; // true if PixelIsPoint
        }
    }

    false
}

/// Helper to read raw `GeoKey` directory bytes
async fn read_geokey_directory(
    tags: &HashMap<u16, IfdEntry>,
    io: &dyn AsyncRangeReader,
) -> AnyResult<Option<Bytes>> {
    let Some(entry) = tags.get(&TAG_GEO_KEY_DIRECTORY) else {
        return Ok(None);
    };

    // GeoKey directory is an array of SHORT values
    if entry.field_type != 3 {
        return Ok(None);
    }

    let total_bytes = entry.count as usize * 2;
    Ok(Some(tag_value_bytes(entry, total_bytes, io).await?))
}

/// Read GDAL metadata: stats and AREA_OR_POINT
/// Returns (min, max, is_point_registered)
async fn read_gdal_metadata_info(
    tags: &HashMap<u16, IfdEntry>,
    io: &dyn AsyncRangeReader,
) -> AnyResult<(Option<f32>, Option<f32>, bool)> {
    let Some(entry) = tags.get(&TAG_GDAL_METADATA) else {
        return Ok((None, None, false));
    };

    // GDAL metadata is ASCII/UTF-8 XML
    let raw_bytes = tag_value_bytes(entry, entry.count as usize, io).await?;

    let metadata_str = String::from_utf8_lossy(&raw_bytes);

    // Parse STATISTICS_MINIMUM and STATISTICS_MAXIMUM from XML
    let min = extract_gdal_stat(&metadata_str, "STATISTICS_MINIMUM");
    let max = extract_gdal_stat(&metadata_str, "STATISTICS_MAXIMUM");

    // Parse AREA_OR_POINT - true if "Point" (pixel centers), false if "Area" or missing
    let is_point_registered = extract_gdal_str(&metadata_str, "AREA_OR_POINT")
        .map(|s| s == "Point")
        .unwrap_or(false);

    Ok((min, max, is_point_registered))
}

fn extract_gdal_stat(metadata: &str, key: &str) -> Option<f32> {
    extract_gdal_str(metadata, key)?.parse().ok()
}

fn extract_gdal_str(metadata: &str, key: &str) -> Option<String> {
    let needle = format!("name=\"{key}\"");
    let pos = metadata.find(&needle)?;
    let rest = &metadata[pos..];
    let start = rest.find('>')? + 1;
    let rest = &rest[start..];
    let end = rest.find('<')?;
    Some(rest[..end].trim().to_string())
}

async fn read_gdal_nodata(
    tags: &HashMap<u16, IfdEntry>,
    io: &dyn AsyncRangeReader,
) -> AnyResult<Option<f64>> {
    let Some(entry) = tags.get(&TAG_GDAL_NODATA) else {
        return Ok(None);
    };

    let raw_bytes = tag_value_bytes(entry, entry.count as usize, io).await?;

    let nodata_str = String::from_utf8_lossy(&raw_bytes);
    let nodata_str = nodata_str.trim_end_matches('\0').trim();

    Ok(nodata_str.parse().ok())
}

// ============================================================================
// Decompression and data conversion
// ============================================================================

fn decompress_tile(
    compressed: &[u8],
    compression: Compression,
    tile_width: usize,
    tile_height: usize,
    bands: usize,
    bytes_per_sample: usize,
) -> AnyResult<Vec<u8>> {
    let expected_size = tile_width * tile_height * bands * bytes_per_sample;

    match compression {
        Compression::None => {
            if compressed.len() >= expected_size {
                Ok(compressed[..expected_size].to_vec())
            } else {
                // Pad with zeros
                let mut result = compressed.to_vec();
                result.resize(expected_size, 0);
                Ok(result)
            }
        }
        Compression::Deflate => {
            use std::io::Read;
            let mut decoder = flate2::read::ZlibDecoder::new(compressed);
            let mut decompressed = Vec::with_capacity(expected_size);
            decoder.read_to_end(&mut decompressed)?;
            Ok(decompressed)
        }
        Compression::Lzw => {
            // Use weezl for LZW decompression
            let mut decoder = weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
            let decompressed = decoder.decode(compressed)?;
            Ok(decompressed)
        }
        Compression::Jpeg => {
            // JPEG decompression using the image crate
            use image::ImageReader;
            use std::io::Cursor;

            let cursor = Cursor::new(compressed);
            let reader = ImageReader::with_format(cursor, image::ImageFormat::Jpeg);
            let img = reader.decode()
                .map_err(|e| format!("JPEG decode error: {e}"))?;

            // Convert to raw bytes based on the image type
            let raw = match img {
                image::DynamicImage::ImageRgb8(rgb) => rgb.into_raw(),
                image::DynamicImage::ImageRgba8(rgba) => rgba.into_raw(),
                image::DynamicImage::ImageLuma8(gray) => gray.into_raw(),
                image::DynamicImage::ImageLumaA8(gray_alpha) => gray_alpha.into_raw(),
                other => {
                    // Convert other formats to RGB8
                    other.to_rgb8().into_raw()
                }
            };

            Ok(raw)
        }
        Compression::Zstd => {
            let decompressed = zstd::stream::decode_all(compressed)?;
            Ok(decompressed)
        }
        Compression::Webp => {
            // WebP decompression using the image crate
            use image::ImageReader;
            use std::io::Cursor;

            let cursor = Cursor::new(compressed);
            let reader = ImageReader::with_format(cursor, image::ImageFormat::WebP);
            let img = reader.decode()
                .map_err(|e| format!("WebP decode error: {e}"))?;

            // Convert to raw bytes based on the image type
            let raw = match img {
                image::DynamicImage::ImageRgb8(rgb) => rgb.into_raw(),
                image::DynamicImage::ImageRgba8(rgba) => rgba.into_raw(),
                image::DynamicImage::ImageLuma8(gray) => gray.into_raw(),
                image::DynamicImage::ImageLumaA8(gray_alpha) => gray_alpha.into_raw(),
                other => {
                    // Convert other formats to RGB8
                    other.to_rgb8().into_raw()
                }
            };

            Ok(raw)
        }
    }
}

/// Reverses TIFF predictor encoding to recover original sample values.
///
/// TIFF predictors are a pre-compression step that improves compression ratios by
/// storing differences between adjacent samples rather than absolute values. This
/// function reverses (decodes) that transformation after decompression.
///
/// # TIFF Predictor Types
///
/// - **Predictor 1 (None)**: No prediction, data is stored as-is.
/// - **Predictor 2 (Horizontal Differencing)**: Each sample stores the difference
///   from the previous sample in the same row. Decoding requires cumulative addition.
/// - **Predictor 3 (Floating Point)**: Specialized for IEEE floating-point data;
///   differences are computed per byte position across samples.
///
/// # Critical Implementation Detail: Sample-Level vs Byte-Level Accumulation
///
/// For predictor 2 with multi-byte samples (16-bit, 32-bit, 64-bit), the differencing
/// operates on **whole samples as integers**, not on individual bytes. This is a subtle
/// but critical distinction:
///
/// ## The Problem (Incorrect Byte-Level Approach)
///
/// A naive implementation might accumulate bytes independently:
/// ```text
/// // WRONG: Byte-level accumulation for 16-bit data
/// for i in 1..data.len() {
///     data[i] = data[i].wrapping_add(data[i - 1]);  // Treats each byte separately
/// }
/// ```
///
/// This produces incorrect results because carries between the low and high bytes
/// of a sample are not propagated correctly. The visual symptom is **horizontal
/// stripe artifacts** in rendered images, where every other row appears corrupted.
///
/// ## The Solution (Correct Sample-Level Approach)
///
/// The correct approach interprets bytes as complete samples, performs integer
/// addition with proper carry propagation, then writes back:
/// ```text
/// // CORRECT: Sample-level accumulation for 16-bit data
/// for i in 1..num_samples {
///     let prev = u16::from_le_bytes([data[prev_offset], data[prev_offset + 1]]);
///     let curr = u16::from_le_bytes([data[curr_offset], data[curr_offset + 1]]);
///     let sum = curr.wrapping_add(prev);  // Proper 16-bit addition with carry
///     data[curr_offset..].copy_from_slice(&sum.to_le_bytes());
/// }
/// ```
///
/// # Row Independence
///
/// Each row is processed independently—the first sample of a new row does NOT
/// accumulate from the last sample of the previous row. This is per the TIFF
/// specification and prevents error propagation across rows.
///
/// # Arguments
///
/// * `data` - Decompressed tile data with predictor encoding still applied
/// * `predictor` - TIFF predictor tag value (1=none, 2=horizontal, 3=floating point)
/// * `tile_width` - Width of the tile in pixels
/// * `bands` - Number of bands (samples per pixel)
/// * `bytes_per_sample` - Size of each sample in bytes (1, 2, 4, or 8)
///
/// # Returns
///
/// The decoded data with original sample values restored.
///
/// # References
///
/// - TIFF 6.0 Specification, Section 14: Differencing Predictor
/// - Adobe TIFF Technote 3: Floating-Point Predictor
fn apply_predictor(
    data: &[u8],
    predictor: u16,
    tile_width: usize,
    bands: usize,
    bytes_per_sample: usize,
) -> AnyResult<Vec<u8>> {
    match predictor {
        // Predictor 1: No prediction applied, return data unchanged
        1 => Ok(data.to_vec()),

        // Predictor 2: Horizontal differencing
        // Samples are stored as: sample[i] = original[i] - original[i-1]
        // We reverse this by cumulative addition: original[i] = sample[i] + original[i-1]
        2 => {
            let mut result = data.to_vec();
            let row_bytes = tile_width * bands * bytes_per_sample;
            let samples_per_row = tile_width * bands;

            // Process each row independently (rows don't accumulate across boundaries)
            for row in result.chunks_mut(row_bytes) {
                match bytes_per_sample {
                    1 => {
                        // 8-bit samples: accumulate per-band (component) with stride
                        // For pixel-interleaved RGB: R0 G0 B0 R1 G1 B1 ...
                        // Each band must accumulate independently:
                        // R1 = R0 + diff_R1, G1 = G0 + diff_G1, B1 = B0 + diff_B1
                        for i in bands..row.len() {
                            row[i] = row[i].wrapping_add(row[i - bands]);
                        }
                    }
                    2 => {
                        // 16-bit samples: must accumulate as u16 to handle carries
                        // between low and high bytes correctly. Accumulate per-band.
                        for i in bands..samples_per_row {
                            let prev_offset = (i - bands) * 2;
                            let curr_offset = i * 2;
                            let prev = u16::from_le_bytes([row[prev_offset], row[prev_offset + 1]]);
                            let curr = u16::from_le_bytes([row[curr_offset], row[curr_offset + 1]]);
                            let sum = curr.wrapping_add(prev);
                            row[curr_offset..curr_offset + 2].copy_from_slice(&sum.to_le_bytes());
                        }
                    }
                    4 => {
                        // 32-bit samples (includes Float32): accumulate as u32
                        // The bit pattern is treated as an integer for differencing,
                        // regardless of whether it represents float or int data. Accumulate per-band.
                        for i in bands..samples_per_row {
                            let prev_offset = (i - bands) * 4;
                            let curr_offset = i * 4;
                            let prev = u32::from_le_bytes([
                                row[prev_offset], row[prev_offset + 1],
                                row[prev_offset + 2], row[prev_offset + 3],
                            ]);
                            let curr = u32::from_le_bytes([
                                row[curr_offset], row[curr_offset + 1],
                                row[curr_offset + 2], row[curr_offset + 3],
                            ]);
                            let sum = curr.wrapping_add(prev);
                            row[curr_offset..curr_offset + 4].copy_from_slice(&sum.to_le_bytes());
                        }
                    }
                    8 => {
                        // 64-bit samples (includes Float64): accumulate as u64
                        // This case is critical for scientific raster data which often
                        // uses Float64 for precision (e.g., climate/agricultural models). Accumulate per-band.
                        for i in bands..samples_per_row {
                            let prev_offset = (i - bands) * 8;
                            let curr_offset = i * 8;
                            let prev = u64::from_le_bytes([
                                row[prev_offset], row[prev_offset + 1],
                                row[prev_offset + 2], row[prev_offset + 3],
                                row[prev_offset + 4], row[prev_offset + 5],
                                row[prev_offset + 6], row[prev_offset + 7],
                            ]);
                            let curr = u64::from_le_bytes([
                                row[curr_offset], row[curr_offset + 1],
                                row[curr_offset + 2], row[curr_offset + 3],
                                row[curr_offset + 4], row[curr_offset + 5],
                                row[curr_offset + 6], row[curr_offset + 7],
                            ]);
                            let sum = curr.wrapping_add(prev);
                            row[curr_offset..curr_offset + 8].copy_from_slice(&sum.to_le_bytes());
                        }
                    }
                    _ => {
                        // Fallback for non-standard sample sizes
                        // Uses byte-level accumulation with stride, which may not be
                        // fully correct for all cases but handles uncommon formats
                        for i in bytes_per_sample..row.len() {
                            row[i] = row[i].wrapping_add(row[i - bytes_per_sample]);
                        }
                    }
                }
            }

            Ok(result)
        }

        // Predictor 3: Floating-point horizontal differencing (Adobe TIFF Technote 3)
        //
        // IMPORTANT: The predictor is applied ROW BY ROW, not to the entire tile at once!
        // Each row is processed independently with its own byte-shuffle layout.
        //
        // For each row:
        // 1. Bytes are grouped by position within the float (byte-shuffled):
        //    [f0b0,f1b0,...,fnb0, f0b1,f1b1,...,fnb1, ...]
        // 2. Horizontal differencing is applied with stride = samples_per_pixel (bands)
        // 3. Floats are reassembled by taking bytes from each section
        //
        // The tiff crate does this in fix_endianness_and_predict() called per-row.
        3 => {
            let samples = bands;  // samples_per_pixel, stride for differencing
            let row_bytes = tile_width * bands * bytes_per_sample;
            let floats_per_row = tile_width * bands;
            let tile_height = data.len() / row_bytes;

            let mut output = vec![0u8; data.len()];

            for row_idx in 0..tile_height {
                let row_start = row_idx * row_bytes;
                let row_end = row_start + row_bytes;

                // Copy row to work buffer
                let mut row_data: Vec<u8> = data[row_start..row_end].to_vec();

                // Step 1: Reverse horizontal differencing within this row
                for i in samples..row_data.len() {
                    row_data[i] = row_data[i].wrapping_add(row_data[i - samples]);
                }

                // Step 2: Reassemble floats from quadrant layout within this row
                // Row is divided into bytes_per_sample sections, each of floats_per_row bytes
                let output_row_start = row_idx * floats_per_row * bytes_per_sample;

                match bytes_per_sample {
                    4 => {
                        for i in 0..floats_per_row {
                            let b0 = row_data[i];
                            let b1 = row_data[floats_per_row + i];
                            let b2 = row_data[floats_per_row * 2 + i];
                            let b3 = row_data[floats_per_row * 3 + i];
                            let val = u32::from_be_bytes([b0, b1, b2, b3]);
                            let out_offset = output_row_start + i * 4;
                            output[out_offset..out_offset + 4].copy_from_slice(&val.to_ne_bytes());
                        }
                    }
                    8 => {
                        for i in 0..floats_per_row {
                            let b0 = row_data[i];
                            let b1 = row_data[floats_per_row + i];
                            let b2 = row_data[floats_per_row * 2 + i];
                            let b3 = row_data[floats_per_row * 3 + i];
                            let b4 = row_data[floats_per_row * 4 + i];
                            let b5 = row_data[floats_per_row * 5 + i];
                            let b6 = row_data[floats_per_row * 6 + i];
                            let b7 = row_data[floats_per_row * 7 + i];
                            let val = u64::from_be_bytes([b0, b1, b2, b3, b4, b5, b6, b7]);
                            let out_offset = output_row_start + i * 8;
                            output[out_offset..out_offset + 8].copy_from_slice(&val.to_ne_bytes());
                        }
                    }
                    2 => {
                        for i in 0..floats_per_row {
                            let b0 = row_data[i];
                            let b1 = row_data[floats_per_row + i];
                            let val = u16::from_be_bytes([b0, b1]);
                            let out_offset = output_row_start + i * 2;
                            output[out_offset..out_offset + 2].copy_from_slice(&val.to_ne_bytes());
                        }
                    }
                    _ => {
                        return Err(format!(
                            "Predictor 3 not supported for {}-byte samples",
                            bytes_per_sample
                        ).into());
                    }
                }
            }

            Ok(output)
        }

        _ => Err(format!("Unsupported predictor: {predictor}").into()),
    }
}

fn convert_to_f32(data: &[u8], data_type: CogDataType, little_endian: bool) -> Vec<f32> {
    let bytes_per_sample = data_type.bytes_per_sample();
    let sample_count = data.len() / bytes_per_sample;
    let mut result = Vec::with_capacity(sample_count);

    for i in 0..sample_count {
        let offset = i * bytes_per_sample;
        let bytes = &data[offset..offset + bytes_per_sample];

        let value = match data_type {
            CogDataType::UInt8 => f32::from(bytes[0]),
            CogDataType::Int8 => {
                // Safe cast: reinterpreting u8 bit pattern as i8
                #[allow(clippy::cast_possible_wrap)]
                f32::from(bytes[0] as i8)
            }
            CogDataType::UInt16 => {
                if little_endian {
                    f32::from(u16::from_le_bytes([bytes[0], bytes[1]]))
                } else {
                    f32::from(u16::from_be_bytes([bytes[0], bytes[1]]))
                }
            }
            CogDataType::Int16 => {
                if little_endian {
                    f32::from(i16::from_le_bytes([bytes[0], bytes[1]]))
                } else {
                    f32::from(i16::from_be_bytes([bytes[0], bytes[1]]))
                }
            }
            CogDataType::UInt32 => {
                // Precision loss acceptable: converting 32-bit int to f32 (mantissa 23 bits)
                #[allow(clippy::cast_precision_loss)]
                if little_endian {
                    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32
                } else {
                    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32
                }
            }
            CogDataType::Int32 => {
                // Precision loss acceptable: converting 32-bit int to f32 (mantissa 23 bits)
                #[allow(clippy::cast_precision_loss)]
                if little_endian {
                    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32
                } else {
                    i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32
                }
            }
            CogDataType::Float32 => {
                if little_endian {
                    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                } else {
                    f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                }
            }
            CogDataType::UInt64 => {
                // Precision loss acceptable: converting 64-bit int to f32 (mantissa 23 bits)
                #[allow(clippy::cast_precision_loss)]
                if little_endian {
                    u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                        bytes[4], bytes[5], bytes[6], bytes[7],
                    ]) as f32
                } else {
                    u64::from_be_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                        bytes[4], bytes[5], bytes[6], bytes[7],
                    ]) as f32
                }
            }
            CogDataType::Int64 => {
                // Precision loss acceptable: converting 64-bit int to f32 (mantissa 23 bits)
                #[allow(clippy::cast_precision_loss)]
                if little_endian {
                    i64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                        bytes[4], bytes[5], bytes[6], bytes[7],
                    ]) as f32
                } else {
                    i64::from_be_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                        bytes[4], bytes[5], bytes[6], bytes[7],
                    ]) as f32
                }
            }
            CogDataType::Float64 => {
                // Precision loss acceptable: converting f64 to f32
                #[allow(clippy::cast_possible_truncation)]
                if little_endian {
                    f64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                        bytes[4], bytes[5], bytes[6], bytes[7],
                    ]) as f32
                } else {
                    f64::from_be_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                        bytes[4], bytes[5], bytes[6], bytes[7],
                    ]) as f32
                }
            }
        };

        result.push(value);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Debug test comparing our predictor 3 implementation with tiff crate
    #[test]
    fn test_compare_with_tiff_crate() {
        use std::io::BufReader;
        use std::fs::File;

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/copernicus_dem_san_francisco.tif");
        if !std::path::Path::new(path).exists() {
            println!("Skipping: test file not found");
            return;
        }

        // Read with tiff crate (reference implementation)
        let file = File::open(path).unwrap();
        let mut decoder = tiff::decoder::Decoder::new(BufReader::new(file)).unwrap();
        let dims = decoder.dimensions().unwrap();
        println!("TIFF dimensions: {}x{}", dims.0, dims.1);

        // Check chunk type
        use tiff::tags::Tag;
        let has_tile_offsets = decoder.get_tag(Tag::TileOffsets).is_ok();
        let has_strip_offsets = decoder.get_tag(Tag::StripOffsets).is_ok();
        println!("Has tile offsets: {}, Has strip offsets: {}", has_tile_offsets, has_strip_offsets);

        if has_tile_offsets {
            if let Ok(tw) = decoder.get_tag_unsigned::<u32>(Tag::TileWidth) {
                println!("Tile width from tiff crate: {}", tw);
            }
            if let Ok(th) = decoder.get_tag_unsigned::<u32>(Tag::TileLength) {
                println!("Tile height from tiff crate: {}", th);
            }
        }

        let tiff_data = match decoder.read_image().unwrap() {
            tiff::decoder::DecodingResult::F32(data) => {
                println!("TIFF crate first 8 values: {:?}", &data[0..8]);
                data
            }
            _ => panic!("Expected f32 data"),
        };

        // Read with our CogReader
        let reader = crate::LocalRangeReader::new(path).unwrap();
        let cog = CogReader::from_reader(std::sync::Arc::new(reader)).unwrap();

        println!("COG metadata: {} x {}, {} bands, predictor={}",
            cog.metadata.width, cog.metadata.height,
            cog.metadata.bands, cog.metadata.predictor);
        println!("Tile size: {} x {}", cog.metadata.tile_width, cog.metadata.tile_height);
        println!("Compression: {:?}", cog.metadata.compression);
        println!("is_point_registered: {}", cog.metadata.geo_transform.is_point_registered);
        println!("tiepoint: {:?}", cog.metadata.geo_transform.tiepoint);
        println!("pixel_scale: {:?}", cog.metadata.geo_transform.pixel_scale);

        // Debug Berkeley Hills coordinates
        let lon = -122.24_f64;
        let lat = 37.88_f64;
        let (px, py) = cog.metadata.geo_transform.world_to_pixel(lon, lat).unwrap();
        println!("\nBerkeley Hills ({}, {}):", lon, lat);
        println!("  pixel: ({}, {})", px, py);
        println!("  truncated: ({}, {})", px as usize, py as usize);

        // Read first tile with our implementation
        let our_data = cog.read_tile(0).unwrap();
        println!("Our first 8 values: {:?}", &our_data[0..8]);

        // Compare first 8 values
        let tile_size = cog.metadata.tile_width * cog.metadata.tile_height;
        println!("Our tile has {} values", our_data.len());
        println!("TIFF tile would have {} values", tile_size);

        // Let's also look at bytes to understand what's happening
        let expected_bytes: [u8; 4] = 101.38305_f32.to_ne_bytes();
        let got_bytes: [u8; 4] = our_data[0].to_ne_bytes();
        println!("Expected first float bytes (native): {:02x} {:02x} {:02x} {:02x}",
            expected_bytes[0], expected_bytes[1], expected_bytes[2], expected_bytes[3]);
        println!("Got first float bytes (native): {:02x} {:02x} {:02x} {:02x}",
            got_bytes[0], got_bytes[1], got_bytes[2], got_bytes[3]);

        // Values should be close (accounting for the fact that tiff crate reads entire image
        // while we read just the first tile)
        let tolerance = 0.001;
        let mut mismatches = 0;
        for i in 0..std::cmp::min(8, our_data.len()) {
            let diff = (our_data[i] - tiff_data[i]).abs();
            if diff > tolerance {
                println!("Mismatch at {}: ours={} vs tiff={}", i, our_data[i], tiff_data[i]);
                mismatches += 1;
            }
        }

        assert_eq!(mismatches, 0, "Found {} value mismatches vs tiff crate", mismatches);
    }

    #[test]
    fn test_data_type_detection() {
        assert_eq!(CogDataType::from_tags(8, 1), Some(CogDataType::UInt8));
        assert_eq!(CogDataType::from_tags(16, 1), Some(CogDataType::UInt16));
        assert_eq!(CogDataType::from_tags(32, 3), Some(CogDataType::Float32));
        assert_eq!(CogDataType::from_tags(64, 3), Some(CogDataType::Float64));
    }

    #[test]
    fn test_compression_detection() {
        assert_eq!(Compression::from_tag(1), Some(Compression::None));
        assert_eq!(Compression::from_tag(5), Some(Compression::Lzw));
        assert_eq!(Compression::from_tag(7), Some(Compression::Jpeg));
        assert_eq!(Compression::from_tag(8), Some(Compression::Deflate));
        assert_eq!(Compression::from_tag(50000), Some(Compression::Zstd));
        assert_eq!(Compression::from_tag(50001), Some(Compression::Webp));
        assert_eq!(Compression::from_tag(999), None);
    }

    #[test]
    fn test_geo_transform() {
        // Test Area registration (default, tiepoint = pixel corner)
        let transform = GeoTransform {
            pixel_scale: Some([10.0, 10.0, 0.0]),
            tiepoint: Some([0.0, 0.0, 0.0, 100.0, 200.0, 0.0]),
            is_point_registered: false,
        };

        // Pixel (0,0) should map to (100, 200)
        let (wx, wy) = transform.pixel_to_world(0.0, 0.0).unwrap();
        assert!((wx - 100.0).abs() < 0.001);
        assert!((wy - 200.0).abs() < 0.001);

        // Pixel (10, 5) should map to (200, 150)
        let (wx, wy) = transform.pixel_to_world(10.0, 5.0).unwrap();
        assert!((wx - 200.0).abs() < 0.001);
        assert!((wy - 150.0).abs() < 0.001);
    }

    #[test]
    fn test_geo_transform_point_registered() {
        // Test Point registration (GTRasterTypeGeoKey=PixelIsPoint, tiepoint = pixel center)
        // When PixelIsPoint, the tiepoint (0,0) -> (100,200) means the CENTER of pixel (0,0) is at (100,200)
        // GDAL compensates by shifting the geotransform origin by half a pixel
        let transform = GeoTransform {
            pixel_scale: Some([10.0, 10.0, 0.0]),
            tiepoint: Some([0.0, 0.0, 0.0, 100.0, 200.0, 0.0]),
            is_point_registered: true,
        };

        // world_to_pixel: World (100, 200) is the center of pixel (0,0)
        // With the 0.5 offset, this maps to pixel (0.5, 0.5), which truncates to pixel (0, 0)
        let (px, py) = transform.world_to_pixel(100.0, 200.0).unwrap();
        assert!((px - 0.5).abs() < 0.001, "Expected px=0.5, got {}", px);
        assert!((py - 0.5).abs() < 0.001, "Expected py=0.5, got {}", py);

        // A point at (105, 195) should be at pixel (1, 0.5) with truncation to pixel (1, 0)
        let (px, py) = transform.world_to_pixel(105.0, 195.0).unwrap();
        assert!((px - 1.0).abs() < 0.001, "Expected px=1.0, got {}", px);
        assert!((py - 1.0).abs() < 0.001, "Expected py=1.0, got {}", py);
    }

    #[test]
    fn test_real_cog_file() {
        // Test with a real COG file if it exists
        let path = "data/viridis/output_cog.tif";
        if !std::path::Path::new(path).exists() {
            println!("Skipping test - file not found: {}", path);
            return;
        }

        let reader = CogReader::open(path).expect("Failed to open COG");
        let m = &reader.metadata;

        println!("Testing real COG: {}", path);
        println!("  Width: {}, Height: {}", m.width, m.height);
        println!("  Tile size: {}x{}", m.tile_width, m.tile_height);
        println!("  CRS code: {:?}", m.crs_code);
        println!("  Bands: {}", m.bands);
        println!("  Compression: {:?}", m.compression);
        println!("  Extent: {:?}", m.geo_transform.get_extent(m.width, m.height));
        println!("  Pixel scale: {:?}", m.geo_transform.pixel_scale);
        println!("  Tiepoint: {:?}", m.geo_transform.tiepoint);

        // Verify basic metadata
        assert!(m.width > 0, "Width should be positive");
        assert!(m.height > 0, "Height should be positive");
        assert!(m.is_tiled(), "Should be a tiled TIFF");

        // Test world_to_pixel for known coordinates
        // Center of the image should be at pixel (width/2, height/2)
        if let Some((px, py)) = m.geo_transform.world_to_pixel(0.0, 0.0) {
            println!("  Lon 0, Lat 0 -> pixel ({}, {})", px, py);
            // For a global dataset, (0,0) should be near center
            assert!(px > 0.0 && px < m.width as f64, "X pixel should be in range");
            assert!(py > 0.0 && py < m.height as f64, "Y pixel should be in range");
        }

        // Try to read tile 0
        let tile_data = reader.read_tile(0).expect("Failed to read tile 0");
        assert!(!tile_data.is_empty(), "Tile data should not be empty");

        let non_nan = tile_data.iter().filter(|v| !v.is_nan()).count();
        println!("  Tile 0: {} values, {} non-NaN", tile_data.len(), non_nan);
        assert!(non_nan > 0, "Tile should have some valid pixels");

        // Test min/max estimation
        let (min, max) = reader.estimate_min_max().expect("Failed to estimate min/max");
        println!("  Estimated min: {}, max: {}", min, max);
        // For an RGB image, values should be 0-255
        assert!(min >= 0.0, "Min should be >= 0");
        assert!(max <= 255.0, "Max should be <= 255 for 8-bit data");
    }

    // ============================================================
    // PREDICTOR=2 (HORIZONTAL DIFFERENCING) TESTS
    //
    // These tests validate the implementation of TIFF Predictor=2 for multi-byte
    // data types (16-bit, 32-bit, 64-bit). The correct implementation must perform
    // sample-level accumulation, NOT byte-level accumulation.
    //
    // BACKGROUND:
    // TIFF Predictor=2 stores the first sample of each row verbatim, then stores
    // differences between consecutive samples. To reconstruct, we accumulate:
    //   sample[i] = sample[i] + sample[i-1]  (wrapping on overflow)
    //
    // THE BUG:
    // A naive implementation might iterate over bytes:
    //   data[i] = data[i] + data[i-1]  // WRONG for multi-byte samples!
    //
    // For example, with 16-bit little-endian data [0x00, 0x01] (value 256):
    // - Byte-level: low byte and high byte accumulate separately, corrupting values
    // - Sample-level: the u16 value 256 is accumulated correctly
    //
    // SYMPTOM:
    // Incorrect byte-level accumulation causes "horizontal stripe" artifacts in
    // rendered tiles because carry propagation between bytes is lost.
    //
    // REFERENCES:
    // - TIFF 6.0 Specification, Section 14
    // - libtiff tif_predict.c: horizontalDifferenceN() functions
    // ============================================================

    /// Validates 16-bit sample-level accumulation for predictor=2.
    ///
    /// This test uses values that would produce incorrect results if bytes were
    /// accumulated independently. The input [0x0100, 0x0001, 0x0001, 0x0001]
    /// (256, 1, 1, 1 as u16) should produce [256, 257, 258, 259].
    ///
    /// With incorrect byte-level accumulation, the low and high bytes would
    /// accumulate separately, producing garbage values.
    #[test]
    fn test_predictor2_16bit_samples() {
        // Input: 4 samples of 16-bit data (8 bytes total)
        // Sample values: [0x0100, 0x0001, 0x0001, 0x0001] (little-endian)
        // As bytes: [0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00]
        let input: Vec<u8> = vec![0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00];

        // Expected after predictor reversal (cumulative sum):
        // Sample 0: 0x0100 (256)
        // Sample 1: 0x0100 + 0x0001 = 0x0101 (257)
        // Sample 2: 0x0101 + 0x0001 = 0x0102 (258)
        // Sample 3: 0x0102 + 0x0001 = 0x0103 (259)
        let result = apply_predictor(&input, 2, 4, 1, 2).unwrap();

        // Verify as 16-bit values
        let s0 = u16::from_le_bytes([result[0], result[1]]);
        let s1 = u16::from_le_bytes([result[2], result[3]]);
        let s2 = u16::from_le_bytes([result[4], result[5]]);
        let s3 = u16::from_le_bytes([result[6], result[7]]);

        assert_eq!(s0, 256, "Sample 0 should be 256");
        assert_eq!(s1, 257, "Sample 1 should be 256 + 1 = 257");
        assert_eq!(s2, 258, "Sample 2 should be 257 + 1 = 258");
        assert_eq!(s3, 259, "Sample 3 should be 258 + 1 = 259");
    }

    /// Validates 32-bit sample-level accumulation for predictor=2.
    ///
    /// This is particularly important for Float32 COG files, where the 4-byte
    /// IEEE 754 representation must be treated as a single unit during
    /// accumulation. Byte-level accumulation would corrupt float bit patterns.
    ///
    /// The test uses integer values for simplicity, but the same logic applies
    /// to float bit patterns stored in the TIFF.
    #[test]
    fn test_predictor2_32bit_samples() {
        // 4 samples of 32-bit data
        // First sample: 0x40000000 (2.0 as f32)
        // Differences: 0x00000001 each
        let input: Vec<u8> = vec![
            0x00, 0x00, 0x00, 0x40,  // 2.0f32 as little-endian
            0x01, 0x00, 0x00, 0x00,  // +1
            0x01, 0x00, 0x00, 0x00,  // +1
            0x01, 0x00, 0x00, 0x00,  // +1
        ];

        let result = apply_predictor(&input, 2, 4, 1, 4).unwrap();

        let s0 = u32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        let s1 = u32::from_le_bytes([result[4], result[5], result[6], result[7]]);
        let s2 = u32::from_le_bytes([result[8], result[9], result[10], result[11]]);
        let s3 = u32::from_le_bytes([result[12], result[13], result[14], result[15]]);

        assert_eq!(s0, 0x40000000, "Sample 0 should be 0x40000000");
        assert_eq!(s1, 0x40000001, "Sample 1 should be 0x40000001");
        assert_eq!(s2, 0x40000002, "Sample 2 should be 0x40000002");
        assert_eq!(s3, 0x40000003, "Sample 3 should be 0x40000003");
    }

    /// Validates 64-bit sample-level accumulation for predictor=2.
    ///
    /// This is the critical test case - 64-bit Float64 COG files were the original
    /// source of the "horizontal stripe" rendering bug. The 8-byte IEEE 754 double
    /// representation requires sample-level accumulation.
    ///
    /// When incorrectly implemented with byte-level accumulation, each of the 8 bytes
    /// accumulates independently, destroying the float bit pattern and causing
    /// wildly incorrect pixel values that manifest as horizontal stripes across tiles.
    #[test]
    fn test_predictor2_64bit_samples() {
        // 3 samples of 64-bit data, using simple integer values for clarity
        // Start with 0x0000000000001000, then add 1 each time
        let input: Vec<u8> = vec![
            0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x1000 (4096)
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // +1
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // +1
        ];

        let result = apply_predictor(&input, 2, 3, 1, 8).unwrap();

        // Convert to u64 and verify sample-level accumulation
        let s0 = u64::from_le_bytes([
            result[0], result[1], result[2], result[3],
            result[4], result[5], result[6], result[7],
        ]);
        let s1 = u64::from_le_bytes([
            result[8], result[9], result[10], result[11],
            result[12], result[13], result[14], result[15],
        ]);
        let s2 = u64::from_le_bytes([
            result[16], result[17], result[18], result[19],
            result[20], result[21], result[22], result[23],
        ]);

        assert_eq!(s0, 0x1000, "Sample 0 should be 0x1000 (4096)");
        assert_eq!(s1, 0x1001, "Sample 1 should be 0x1000 + 1 = 0x1001 (4097)");
        assert_eq!(s2, 0x1002, "Sample 2 should be 0x1001 + 1 = 0x1002 (4098)");
    }

    /// Validates wrapping arithmetic for predictor=2 overflow cases.
    ///
    /// TIFF horizontal differencing uses unsigned arithmetic that wraps on overflow.
    /// This is intentional - the encoder produces differences that may be negative
    /// when interpreted as signed, but the unsigned representation wraps correctly.
    ///
    /// For example, encoding the sequence [65535, 0] produces differences [65535, 1]
    /// because 0 - 65535 = 1 in u16 wrapping arithmetic. On decode, 65535 + 1 = 0.
    ///
    /// This test verifies that our implementation uses wrapping_add() correctly.
    #[test]
    fn test_predictor2_wrapping_overflow() {
        // Test that we use wrapping_add correctly for overflow
        // Start with max u16, add 1 should wrap to 0
        let input: Vec<u8> = vec![
            0xFF, 0xFF,  // 65535
            0x01, 0x00,  // +1 should wrap to 0
        ];

        let result = apply_predictor(&input, 2, 2, 1, 2).unwrap();

        let s0 = u16::from_le_bytes([result[0], result[1]]);
        let s1 = u16::from_le_bytes([result[2], result[3]]);

        assert_eq!(s0, 65535, "Sample 0 should be 65535");
        assert_eq!(s1, 0, "Sample 1 should wrap to 0 (65535 + 1)");
    }

    /// Validates row-independent accumulation for predictor=2.
    ///
    /// Per TIFF specification, horizontal differencing resets at row boundaries.
    /// Each row's first sample is stored verbatim, and accumulation starts fresh.
    /// This is critical because:
    ///
    /// 1. Tiles may be decoded in any order (random access)
    /// 2. Rows within a tile must be independently decodable for parallel processing
    /// 3. An error in one row should not propagate to subsequent rows
    ///
    /// This test verifies that row 2's values are NOT affected by row 1's final
    /// accumulated value.
    #[test]
    fn test_predictor2_multiple_rows() {
        // 2 rows of 3 samples each (16-bit)
        let input: Vec<u8> = vec![
            // Row 1: [100, +1, +1]
            0x64, 0x00, 0x01, 0x00, 0x01, 0x00,
            // Row 2: [200, +2, +2] - should NOT continue from row 1
            0xC8, 0x00, 0x02, 0x00, 0x02, 0x00,
        ];

        let result = apply_predictor(&input, 2, 3, 1, 2).unwrap();

        // Row 1
        let r1s0 = u16::from_le_bytes([result[0], result[1]]);
        let r1s1 = u16::from_le_bytes([result[2], result[3]]);
        let r1s2 = u16::from_le_bytes([result[4], result[5]]);

        // Row 2
        let r2s0 = u16::from_le_bytes([result[6], result[7]]);
        let r2s1 = u16::from_le_bytes([result[8], result[9]]);
        let r2s2 = u16::from_le_bytes([result[10], result[11]]);

        assert_eq!(r1s0, 100, "Row 1 Sample 0");
        assert_eq!(r1s1, 101, "Row 1 Sample 1");
        assert_eq!(r1s2, 102, "Row 1 Sample 2");

        assert_eq!(r2s0, 200, "Row 2 Sample 0 - fresh start");
        assert_eq!(r2s1, 202, "Row 2 Sample 1");
        assert_eq!(r2s2, 204, "Row 2 Sample 2");
    }

    /// Validates 8-bit multiband predictor=2 (byte-level accumulation).
    ///
    /// For 8-bit data, sample size equals byte size, so accumulation is naturally
    /// byte-level. This test ensures that multiband 8-bit images (e.g., RGB) are
    /// handled correctly - all bands within a row are accumulated sequentially.
    ///
    /// Layout for 2-band 8-bit: [pixel0_band0, pixel0_band1, pixel1_band0, pixel1_band1]
    /// Accumulation proceeds per-component (band) across pixels in the row.
    /// For pixel-interleaved data: R0 G0 R1 G1 -> R0, G0, R0+R1, G0+G1
    #[test]
    fn test_predictor2_multiband_8bit() {
        // 2 pixels, 2 bands each (8-bit)
        // Layout: [pixel0_band0, pixel0_band1, pixel1_band0, pixel1_band1]
        let input: Vec<u8> = vec![10, 20, 1, 2];

        let result = apply_predictor(&input, 2, 2, 2, 1).unwrap();

        // Per-component accumulation: each band accumulates independently
        // Band 0: result[0] = 10, result[2] = 10 + 1 = 11
        // Band 1: result[1] = 20, result[3] = 20 + 2 = 22
        assert_eq!(result[0], 10, "Pixel 0 Band 0");
        assert_eq!(result[1], 20, "Pixel 0 Band 1");
        assert_eq!(result[2], 11, "Pixel 1 Band 0 = 10 + 1");
        assert_eq!(result[3], 22, "Pixel 1 Band 1 = 20 + 2");
    }

    /// Validates 16-bit multiband predictor=2 (per-component accumulation).
    ///
    /// For 16-bit multiband data, each sample must be accumulated as a u16,
    /// and each band/component accumulates independently (per TIFF Technote 3).
    #[test]
    fn test_predictor2_multiband_16bit() {
        // 2 pixels, 2 bands each (16-bit)
        // Layout: [p0b0_lo, p0b0_hi, p0b1_lo, p0b1_hi, p1b0_lo, p1b0_hi, p1b1_lo, p1b1_hi]
        // Sample values: [100, 200, 1, 2]
        let input: Vec<u8> = vec![
            100, 0,  // pixel 0 band 0 = 100
            200, 0,  // pixel 0 band 1 = 200
            1, 0,    // pixel 1 band 0 = +1
            2, 0,    // pixel 1 band 1 = +2
        ];

        let result = apply_predictor(&input, 2, 2, 2, 2).unwrap();

        // Per-component accumulation: each band accumulates independently
        // Band 0: s[0] = 100, s[2] = 100 + 1 = 101
        // Band 1: s[1] = 200, s[3] = 200 + 2 = 202
        let s0 = u16::from_le_bytes([result[0], result[1]]);
        let s1 = u16::from_le_bytes([result[2], result[3]]);
        let s2 = u16::from_le_bytes([result[4], result[5]]);
        let s3 = u16::from_le_bytes([result[6], result[7]]);

        assert_eq!(s0, 100, "Pixel 0 Band 0");
        assert_eq!(s1, 200, "Pixel 0 Band 1");
        assert_eq!(s2, 101, "Pixel 1 Band 0 = 100 + 1");
        assert_eq!(s3, 202, "Pixel 1 Band 1 = 200 + 2");
    }

    // ============================================================
    // OVERVIEW QUALITY HINT TESTS
    //
    // OverviewQualityHint controls which COG overview levels are considered
    // acceptable quality for tile serving. This is important because:
    //
    // 1. Some COG files have blurry or poorly-resampled overviews
    // 2. Performance vs. quality tradeoffs vary by use case
    // 3. Layer administrators may want to force full-resolution serving
    //
    // The hint is stored in the database as an i32:
    //   - NULL  -> ComputeAtRuntime (analyze at load time)
    //   - -1    -> NoneUsable (always use full resolution)
    //   - -2    -> AllUsable (all overviews are acceptable)
    //   - n>=0  -> MinUsable(n) (overview index n and higher are acceptable)
    // ============================================================

    /// Validates database value to OverviewQualityHint conversion.
    ///
    /// Tests the from_db_value() function which converts nullable i32 database
    /// values to the enum representation used in application code.
    #[test]
    fn test_overview_hint_from_db_value() {
        // None -> ComputeAtRuntime
        assert!(matches!(
            OverviewQualityHint::from_db_value(None),
            OverviewQualityHint::ComputeAtRuntime
        ));

        // -1 -> NoneUsable (force full resolution)
        assert!(matches!(
            OverviewQualityHint::from_db_value(Some(-1)),
            OverviewQualityHint::NoneUsable
        ));

        // -2 -> AllUsable (all overviews are good quality)
        assert!(matches!(
            OverviewQualityHint::from_db_value(Some(-2)),
            OverviewQualityHint::AllUsable
        ));

        // Positive values -> MinUsable(n)
        assert!(matches!(
            OverviewQualityHint::from_db_value(Some(0)),
            OverviewQualityHint::MinUsable(0)
        ));
        assert!(matches!(
            OverviewQualityHint::from_db_value(Some(3)),
            OverviewQualityHint::MinUsable(3)
        ));
    }

    /// Validates OverviewQualityHint to database value conversion.
    ///
    /// Tests the to_db_value() function which converts the enum back to the
    /// nullable i32 representation for database storage. This is the inverse
    /// of from_db_value() and ensures round-trip consistency.
    #[test]
    fn test_overview_hint_to_db_value() {
        // NoneUsable = -1 (force full resolution)
        assert_eq!(OverviewQualityHint::NoneUsable.to_db_value(), Some(-1));
        // AllUsable = -2 (all overviews are good)
        assert_eq!(OverviewQualityHint::AllUsable.to_db_value(), Some(-2));
        assert_eq!(OverviewQualityHint::MinUsable(0).to_db_value(), Some(0));
        assert_eq!(OverviewQualityHint::MinUsable(5).to_db_value(), Some(5));
        // ComputeAtRuntime returns None (no db value)
        assert_eq!(OverviewQualityHint::ComputeAtRuntime.to_db_value(), None);
    }

    /// Validates WebP decompression produces correct raw pixel data.
    ///
    /// WebP is a lossy/lossless image format that GDAL can use for COG tiles
    /// (compression tag 50001). This test encodes a small RGB image as WebP,
    /// then verifies that decompress_tile correctly decodes it back to raw RGB.
    #[test]
    fn test_webp_decompression() {
        use image::{ImageBuffer, ImageFormat, Rgb, DynamicImage};
        use std::io::Cursor;

        // Create a 2x2 RGB test image with known pixel values
        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(2, 2);
        img.put_pixel(0, 0, Rgb([255, 0, 0]));     // Red
        img.put_pixel(1, 0, Rgb([0, 255, 0]));     // Green
        img.put_pixel(0, 1, Rgb([0, 0, 255]));     // Blue
        img.put_pixel(1, 1, Rgb([255, 255, 0]));   // Yellow

        // Encode as WebP
        let mut webp_data = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(img)
            .write_to(&mut webp_data, ImageFormat::WebP)
            .expect("Failed to encode WebP");

        // Decompress using our function
        let result = decompress_tile(
            webp_data.get_ref(),
            Compression::Webp,
            2,  // tile_width
            2,  // tile_height
            3,  // bands (RGB)
            1,  // bytes_per_sample
        ).expect("WebP decompression failed");

        // Verify we got 12 bytes (2x2 pixels x 3 channels)
        assert_eq!(result.len(), 12, "Expected 12 bytes for 2x2 RGB image");

        // Verify pixel values (note: lossy compression may alter values slightly,
        // but lossless WebP should preserve exact values)
        // Row 0: [R, G, B, R, G, B] for pixels (0,0) and (1,0)
        // Row 1: [R, G, B, R, G, B] for pixels (0,1) and (1,1)

        // Red pixel (0,0)
        assert!(result[0] > 200, "Red channel of red pixel should be high");
        assert!(result[1] < 50, "Green channel of red pixel should be low");
        assert!(result[2] < 50, "Blue channel of red pixel should be low");

        // Green pixel (1,0)
        assert!(result[3] < 50, "Red channel of green pixel should be low");
        assert!(result[4] > 200, "Green channel of green pixel should be high");
        assert!(result[5] < 50, "Blue channel of green pixel should be low");

        // Blue pixel (0,1)
        assert!(result[6] < 50, "Red channel of blue pixel should be low");
        assert!(result[7] < 50, "Green channel of blue pixel should be low");
        assert!(result[8] > 200, "Blue channel of blue pixel should be high");

        // Yellow pixel (1,1)
        assert!(result[9] > 200, "Red channel of yellow pixel should be high");
        assert!(result[10] > 200, "Green channel of yellow pixel should be high");
        assert!(result[11] < 50, "Blue channel of yellow pixel should be low");
    }
}

// ============================================================
// COMPREHENSIVE INTEGRATION TESTS FOR COG READER
//
// These tests verify correct behavior against real COG files and GDAL output.
// They require test data files in data/grayscale/ to run (skipped if missing).
//
// Key behaviors tested:
// 1. CRS detection from GeoKey tags
// 2. Overview scale calculation (MUST use floor division to match GDAL)
// 3. Coordinate transformation accuracy
// 4. Overview selection algorithm
// 5. Predictor=2 implementation (covered in detail above)
//
// IMPORTANT: These tests catch real-world bugs that unit tests may miss,
// such as the ceiling vs. floor division bug that caused ~6% coordinate errors.
// ============================================================

/// Verifies CRS detection for Web Mercator (EPSG:3857) projection.
///
/// The test file uses EPSG:3857 (Web Mercator), commonly used for web mapping.
/// Correct CRS detection is essential for proper coordinate transformation.
#[test]
fn test_gray_3857_crs_detection() {
    let path = "data/grayscale/gray_3857-cog.tif";
    if !std::path::Path::new(path).exists() {
        println!("Skipping - file not found: {}", path);
        return;
    }

    let reader = CogReader::open(path).expect("Failed to open COG");

    // EPSG:3857 should be detected
    assert_eq!(reader.metadata.crs_code, Some(3857), "CRS should be detected as 3857");
}

/// TEST: Overview geometry and pixel values match GDAL
///
/// An overview's pixel size is the full-resolution pixel size times `full_size / overview_size`
/// per axis (GDAL derives an overview's geotransform as `extent / overview size`); that ratio is
/// generally not an integer (gray_3857-cog.tif: 20966 / 1310 = 16.005, and a /8 level of a
/// 10980 px scene is 1373 px wide, 7.997). Compares sizes, ratios and sampled pixel values of
/// every overview with GDAL.
#[test]
fn test_overview_geometry_matches_gdal() {
    let path = "data/grayscale/gray_3857-cog.tif";
    if !std::path::Path::new(path).exists() {
        println!("Skipping - file not found: {}", path);
        return;
    }

    let reader = CogReader::open(path).expect("Failed to open COG");
    let ds = gdal::Dataset::open(path).expect("GDAL failed to open");
    let band = ds.rasterband(1).expect("Failed to get band 1");
    let (full_w, full_h) = band.size();
    let gdal_gt = ds.geo_transform().expect("geotransform");
    let pixel_scale = reader.metadata.geo_transform.pixel_scale.expect("pixel scale");

    assert!(!reader.overviews.is_empty(), "Should have overviews");
    assert_eq!(band.overview_count().expect("overview count") as usize, reader.overviews.len());

    for (i, ovr) in reader.overviews.iter().enumerate() {
        let gdal_ovr = band.overview(i).expect("GDAL overview");
        let (gw, gh) = gdal_ovr.size();
        assert_eq!((ovr.width, ovr.height), (gw, gh), "overview {i} size");

        // Exact per-axis ratios; the integer `scale` is just their rounding.
        assert!((ovr.scale_x - full_w as f64 / gw as f64).abs() < 1e-12, "overview {i} scale_x");
        assert!((ovr.scale_y - full_h as f64 / gh as f64).abs() < 1e-12, "overview {i} scale_y");
        assert_eq!(ovr.scale, ovr.scale_x.round() as usize, "overview {i} integer scale");

        // Overview pixel size = GDAL's extent / overview size
        let gdal_pixel_x = gdal_gt[1].abs() * full_w as f64 / gw as f64;
        let gdal_pixel_y = gdal_gt[5].abs() * full_h as f64 / gh as f64;
        assert!((pixel_scale[0] * ovr.scale_x - gdal_pixel_x).abs() < 1e-6 * gdal_pixel_x, "overview {i} pixel width");
        assert!((pixel_scale[1] * ovr.scale_y - gdal_pixel_y).abs() < 1e-6 * gdal_pixel_y, "overview {i} pixel height");

        // Sampled pixels decode to GDAL's values
        for fx in [0.03, 0.31, 0.5, 0.77, 0.97] {
            for fy in [0.04, 0.45, 0.66, 0.95] {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let (px, py) = ((fx * gw as f64) as usize, (fy * gh as f64) as usize);
                let want: gdal::raster::Buffer<f32> =
                    gdal_ovr.read_as((px as isize, py as isize), (1, 1), (1, 1), None).expect("GDAL read");
                let tile_index = ovr.tile_index_for_pixel(px, py).expect("pixel in overview");
                let tile = reader.read_overview_tile(i, tile_index).expect("overview tile");
                let got = tile[(py % ovr.tile_height) * ovr.tile_width + px % ovr.tile_width];
                assert!((got - want.data()[0]).abs() < 0.001, "overview {i} pixel ({px},{py}): ours {got}, GDAL {}", want.data()[0]);
            }
        }
    }

    // gray_3857: overview 3 is 1310 px, ratio 16.005 (rounds to 16)
    if let Some(ovr3) = reader.overviews.get(3) {
        assert_eq!(ovr3.scale, 16);
        assert!((ovr3.scale_x - 20966.0 / 1310.0).abs() < 1e-12);
    }
}

/// TEST: Pixel values at known positions match GDAL output
///
/// This test verifies that the tile extraction produces correct pixel values.
/// Values were obtained from GDAL: gdal_translate with -projwin
#[test]
fn test_overview_pixel_values_match_gdal() {
    let path = "data/grayscale/gray_3857-cog.tif";
    if !std::path::Path::new(path).exists() {
        println!("Skipping - file not found: {}", path);
        return;
    }

    let reader = CogReader::open(path).expect("Failed to open COG");

    // Test reading from overview 3 (1310x1310, scale 16)
    if reader.overviews.len() > 3 {
        let ovr_idx = 3;
        let ovr = &reader.overviews[ovr_idx];

        // Verify overview properties
        assert_eq!(ovr.width, 1310, "Overview 3 should be 1310 wide");
        assert_eq!(ovr.height, 1310, "Overview 3 should be 1310 tall");
        assert_eq!(ovr.scale, 16, "Overview 3 scale should be 16");

        // Read tile 0 from overview
        let tile_data = reader.read_overview_tile(ovr_idx, 0).expect("Failed to read overview tile 0");

        // Verify tile data size
        let expected_size = ovr.tile_width * ovr.tile_height;
        assert_eq!(tile_data.len(), expected_size, "Tile data should be {}x{} pixels", ovr.tile_width, ovr.tile_height);

        // Check that we have valid (non-NaN) data
        let valid_count = tile_data.iter().filter(|v| !v.is_nan()).count();
        assert!(valid_count > 0, "Tile should have valid (non-NaN) pixels");

        // Verify pixel values are in expected range for grayscale (0-255)
        let min_val = tile_data.iter().filter(|v| !v.is_nan()).copied().fold(f32::INFINITY, f32::min);
        let max_val = tile_data.iter().filter(|v| !v.is_nan()).copied().fold(f32::NEG_INFINITY, f32::max);

        assert!(min_val >= 0.0, "Min value should be >= 0, got {}", min_val);
        assert!(max_val <= 255.0, "Max value should be <= 255, got {}", max_val);

        // Check specific pixel value that GDAL reports
        // At tile position (0, 0) in overview 3, GDAL shows value ~176
        let corner_value = tile_data[0];
        assert!(
            !corner_value.is_nan() && (100.0..=255.0).contains(&corner_value),
            "Corner value should be valid grayscale, got {}",
            corner_value
        );
    }
}

/// TEST: Scale factor correctly affects coordinate mapping
///
/// This test verifies that the scale factor properly adjusts the pixel_scale
/// when using overviews, which is critical for correct tile generation.
#[test]
fn test_scale_factor_coordinate_mapping() {
    let path = "data/grayscale/gray_3857-cog.tif";
    if !std::path::Path::new(path).exists() {
        println!("Skipping - file not found: {}", path);
        return;
    }

    let reader = CogReader::open(path).expect("Failed to open COG");

    if let (Some(pixel_scale), Some(_tiepoint)) = (
        reader.metadata.geo_transform.pixel_scale,
        reader.metadata.geo_transform.tiepoint,
    ) {
        let base_scale_x = pixel_scale[0];

        for (i, ovr) in reader.overviews.iter().enumerate() {
            // Calculate effective scale for this overview
            // Effective pixel size of this overview: exactly full extent / overview width
            let effective_scale_x = base_scale_x * ovr.scale_x;

            let full_extent_x = base_scale_x * (reader.metadata.width as f64);
            let expected_effective_scale = full_extent_x / (ovr.width as f64);

            let tolerance = expected_effective_scale * 1e-9;
            assert!(
                (effective_scale_x - expected_effective_scale).abs() < tolerance,
                "Overview {} effective scale mismatch: got {}, expected {} (within {})",
                i, effective_scale_x, expected_effective_scale, tolerance
            );
        }
    }
}

/// Validates the overview selection algorithm.
///
/// The best_overview_for_resolution() method should select the smallest overview
/// that can provide sufficient detail for the requested extent. This test
/// verifies that:
/// - Small extents prefer full resolution (or low-index overviews)
/// - Large extents use higher-index overviews for performance
/// - The returned index is always valid
#[test]
fn test_best_overview_selection() {
    let path = "data/grayscale/gray_3857-cog.tif";
    if !std::path::Path::new(path).exists() {
        println!("Skipping - file not found: {}", path);
        return;
    }

    let reader = CogReader::open(path).expect("Failed to open COG");

    // For a small extent (256 pixels worth), should return None (use full res)
    let _full_res = reader.best_overview_for_resolution(256, 256);
    // This might return None or a small overview index depending on the image

    // For a large extent (whole image), should return highest overview
    let large_extent = reader.best_overview_for_resolution(20000, 20000);
    assert!(
        large_extent.is_some() || reader.overviews.is_empty(),
        "Large extent should use an overview"
    );

    // For medium extent, should return appropriate overview
    let medium_extent = reader.best_overview_for_resolution(5000, 5000);
    // Just verify it doesn't panic and returns a valid index
    if let Some(idx) = medium_extent {
        assert!(
            idx < reader.overviews.len(),
            "Overview index {} should be valid",
            idx
        );
    }
}

/// TEST: Horizontal differencing predictor (predictor=2) for multi-byte samples
///
/// This tests the fix for TIFF predictor=2 which requires sample-level accumulation
/// for 16-bit, 32-bit, and 64-bit data types. The bug was performing byte-level
/// accumulation which corrupted multi-byte values.
///
/// Reference: libtiff tif_predict.c casts to uint16_t/uint32_t/uint64_t and adds
/// whole samples, not individual bytes.
#[test]
fn test_predictor2_multibyte_samples() {
    // Test 16-bit horizontal differencing
    // Input: [100, 0, 5, 0, 10, 0] represents [100, 5, 10] as u16 (little-endian)
    // After predictor=2: [100, 105, 115]
    let input_16: Vec<u8> = vec![100, 0, 5, 0, 10, 0]; // 3 u16 samples: 100, 5, 10
    let result_16 = apply_predictor(&input_16, 2, 3, 1, 2).expect("predictor failed");

    // Verify: first sample unchanged, others accumulated
    let s0 = u16::from_le_bytes([result_16[0], result_16[1]]);
    let s1 = u16::from_le_bytes([result_16[2], result_16[3]]);
    let s2 = u16::from_le_bytes([result_16[4], result_16[5]]);

    assert_eq!(s0, 100, "First sample should be unchanged");
    assert_eq!(s1, 105, "Second sample should be 100 + 5 = 105");
    assert_eq!(s2, 115, "Third sample should be 105 + 10 = 115");

    // Test 32-bit horizontal differencing (e.g., Float32 stored as u32)
    // Input: [1000, 50, 100] as u32 differences
    let mut input_32: Vec<u8> = Vec::new();
    input_32.extend_from_slice(&1000u32.to_le_bytes());
    input_32.extend_from_slice(&50u32.to_le_bytes());
    input_32.extend_from_slice(&100u32.to_le_bytes());

    let result_32 = apply_predictor(&input_32, 2, 3, 1, 4).expect("predictor failed");

    let s0_32 = u32::from_le_bytes([result_32[0], result_32[1], result_32[2], result_32[3]]);
    let s1_32 = u32::from_le_bytes([result_32[4], result_32[5], result_32[6], result_32[7]]);
    let s2_32 = u32::from_le_bytes([result_32[8], result_32[9], result_32[10], result_32[11]]);

    assert_eq!(s0_32, 1000, "First u32 sample should be unchanged");
    assert_eq!(s1_32, 1050, "Second u32 sample should be 1000 + 50 = 1050");
    assert_eq!(s2_32, 1150, "Third u32 sample should be 1050 + 100 = 1150");

    // Test 64-bit horizontal differencing (e.g., Float64)
    let mut input_64: Vec<u8> = Vec::new();
    input_64.extend_from_slice(&10000u64.to_le_bytes());
    input_64.extend_from_slice(&500u64.to_le_bytes());
    input_64.extend_from_slice(&1000u64.to_le_bytes());

    let result_64 = apply_predictor(&input_64, 2, 3, 1, 8).expect("predictor failed");

    let s0_64 = u64::from_le_bytes(result_64[0..8].try_into().unwrap());
    let s1_64 = u64::from_le_bytes(result_64[8..16].try_into().unwrap());
    let s2_64 = u64::from_le_bytes(result_64[16..24].try_into().unwrap());

    assert_eq!(s0_64, 10000, "First u64 sample should be unchanged");
    assert_eq!(s1_64, 10500, "Second u64 sample should be 10000 + 500 = 10500");
    assert_eq!(s2_64, 11500, "Third u64 sample should be 10500 + 1000 = 11500");
}

/// TEST: Predictor=2 handles wrapping correctly
///
/// The predictor should use wrapping arithmetic to handle overflow cases
/// that occur in differenced data.
#[test]
fn test_predictor2_wrapping_behavior() {
    // Test u16 wrapping: 65535 + 1 = 0 (wraps)
    let mut input: Vec<u8> = Vec::new();
    input.extend_from_slice(&65535u16.to_le_bytes()); // First sample: max u16
    input.extend_from_slice(&1u16.to_le_bytes());     // Delta: +1 (wraps to 0)

    let result = apply_predictor(&input, 2, 2, 1, 2).expect("predictor failed");

    let s0 = u16::from_le_bytes([result[0], result[1]]);
    let s1 = u16::from_le_bytes([result[2], result[3]]);

    assert_eq!(s0, 65535, "First sample unchanged");
    assert_eq!(s1, 0, "Second sample should wrap: 65535 + 1 = 0");

    // Test u32 wrapping
    let mut input_32: Vec<u8> = Vec::new();
    input_32.extend_from_slice(&0xFFFFFFFFu32.to_le_bytes());
    input_32.extend_from_slice(&2u32.to_le_bytes());

    let result_32 = apply_predictor(&input_32, 2, 2, 1, 4).expect("predictor failed");
    let s1_32 = u32::from_le_bytes([result_32[4], result_32[5], result_32[6], result_32[7]]);
    assert_eq!(s1_32, 1, "u32 should wrap: 0xFFFFFFFF + 2 = 1");
}

/// TEST: Multi-row predictor handling
///
/// Each row should be processed independently - predictor resets at row boundaries.
#[test]
fn test_predictor2_multirow() {
    // 2 rows of 3 u16 samples each
    let mut input: Vec<u8> = Vec::new();
    // Row 1: [100, 10, 20] -> [100, 110, 130]
    input.extend_from_slice(&100u16.to_le_bytes());
    input.extend_from_slice(&10u16.to_le_bytes());
    input.extend_from_slice(&20u16.to_le_bytes());
    // Row 2: [200, 5, 15] -> [200, 205, 220]
    input.extend_from_slice(&200u16.to_le_bytes());
    input.extend_from_slice(&5u16.to_le_bytes());
    input.extend_from_slice(&15u16.to_le_bytes());

    let result = apply_predictor(&input, 2, 3, 1, 2).expect("predictor failed");

    // Row 1 verification
    let r1_s0 = u16::from_le_bytes([result[0], result[1]]);
    let r1_s1 = u16::from_le_bytes([result[2], result[3]]);
    let r1_s2 = u16::from_le_bytes([result[4], result[5]]);

    assert_eq!(r1_s0, 100, "Row 1, sample 0");
    assert_eq!(r1_s1, 110, "Row 1, sample 1: 100 + 10 = 110");
    assert_eq!(r1_s2, 130, "Row 1, sample 2: 110 + 20 = 130");

    // Row 2 verification - should restart from row's first sample
    let r2_s0 = u16::from_le_bytes([result[6], result[7]]);
    let r2_s1 = u16::from_le_bytes([result[8], result[9]]);
    let r2_s2 = u16::from_le_bytes([result[10], result[11]]);

    assert_eq!(r2_s0, 200, "Row 2, sample 0 (fresh start)");
    assert_eq!(r2_s1, 205, "Row 2, sample 1: 200 + 5 = 205");
    assert_eq!(r2_s2, 220, "Row 2, sample 2: 205 + 15 = 220");
}

/// Tests that verify our implementation against GDAL (reference implementation)
/// These tests require GDAL to be installed and the gdal crate as a dev dependency
#[cfg(test)]
mod gdal_verification_tests {
    use super::*;
    use crate::point_query::PointQuery;
    use gdal::Metadata;
    use std::sync::Arc;

    const TEST_COG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/copernicus_dem_san_francisco.tif");

    fn get_test_cog() -> Option<CogReader> {
        if !std::path::Path::new(TEST_COG_PATH).exists() {
            println!("Skipping: test file not found at {}", TEST_COG_PATH);
            return None;
        }
        let reader = crate::LocalRangeReader::new(TEST_COG_PATH).ok()?;
        CogReader::from_reader(Arc::new(reader)).ok()
    }

    /// Verify our geotransform handling matches GDAL exactly
    /// Tests the half-pixel shift for PixelIsPoint datasets (RFC 33)
    #[test]
    fn test_gdal_geotransform_comparison() {
        let Some(cog) = get_test_cog() else { return };

        // Open with GDAL
        let gdal_ds = gdal::Dataset::open(TEST_COG_PATH).expect("GDAL failed to open");
        let gdal_gt = gdal_ds.geo_transform().expect("Failed to get geotransform");

        // Get AREA_OR_POINT metadata
        let area_or_point = gdal_ds.metadata_item("AREA_OR_POINT", "").unwrap_or_default();
        println!("GDAL AREA_OR_POINT: {:?}", area_or_point);
        println!("Our is_point_registered: {}", cog.metadata.geo_transform.is_point_registered);

        // GDAL's geotransform is [origin_x, pixel_width, skew_x, origin_y, skew_y, -pixel_height]
        let gdal_origin_x = gdal_gt[0];
        let gdal_origin_y = gdal_gt[3];
        let gdal_pixel_width = gdal_gt[1];
        let gdal_pixel_height = -gdal_gt[5]; // GDAL uses negative for Y

        println!("GDAL geotransform: {:?}", gdal_gt);
        println!("Our tiepoint: {:?}", cog.metadata.geo_transform.tiepoint);
        println!("Our pixel_scale: {:?}", cog.metadata.geo_transform.pixel_scale);

        // Verify pixel scale matches
        if let Some(scale) = &cog.metadata.geo_transform.pixel_scale {
            assert!((scale[0] - gdal_pixel_width).abs() < 1e-12,
                "Pixel width mismatch: ours={}, gdal={}", scale[0], gdal_pixel_width);
            assert!((scale[1] - gdal_pixel_height).abs() < 1e-12,
                "Pixel height mismatch: ours={}, gdal={}", scale[1], gdal_pixel_height);
        }

        // For PixelIsPoint datasets, GDAL shifts origin by half a pixel
        // Our implementation stores the raw tiepoint and applies the shift in world_to_pixel
        if cog.metadata.geo_transform.is_point_registered {
            println!("\nPixelIsPoint dataset - verifying coordinate transform matches GDAL");

            // Sample several test points and verify pixel coordinates match
            let test_points = [
                (-122.4, 37.78),    // SF Downtown
                (-122.24, 37.88),   // Berkeley Hills
                (-122.5965, 37.9236), // Mt Tam
            ];

            for (lon, lat) in test_points {
                // Our calculation
                let (our_px, our_py) = cog.metadata.geo_transform.world_to_pixel(lon, lat).unwrap();

                // GDAL calculation: px = (x - origin_x) / pixel_width
                let gdal_px = (lon - gdal_origin_x) / gdal_pixel_width;
                let gdal_py = (gdal_origin_y - lat) / gdal_pixel_height;

                println!("Point ({}, {}): ours=({:.6}, {:.6}), gdal=({:.6}, {:.6})",
                    lon, lat, our_px, our_py, gdal_px, gdal_py);

                assert!((our_px - gdal_px).abs() < 0.001,
                    "X pixel mismatch at ({}, {}): ours={}, gdal={}", lon, lat, our_px, gdal_px);
                assert!((our_py - gdal_py).abs() < 0.001,
                    "Y pixel mismatch at ({}, {}): ours={}, gdal={}", lon, lat, our_py, gdal_py);
            }
        }
    }

    /// Verify pixel values match GDAL exactly at specific coordinates
    #[test]
    fn test_gdal_pixel_value_comparison() {
        let Some(cog) = get_test_cog() else { return };

        // Open with GDAL
        let gdal_ds = gdal::Dataset::open(TEST_COG_PATH).expect("GDAL failed to open");
        let band = gdal_ds.rasterband(1).expect("Failed to get band 1");

        // Test coordinates with known elevations
        let test_coords = [
            (-122.4, 37.78, "SF Downtown"),
            (-122.24, 37.88, "Berkeley Hills"),
            (-122.5965, 37.9236, "Mt Tam"),
            (-122.38, 37.79, "SF Bay"),
        ];

        let gdal_gt = gdal_ds.geo_transform().expect("Failed to get geotransform");

        for (lon, lat, name) in test_coords {
            // Calculate pixel coordinates using GDAL's geotransform
            let gdal_px = ((lon - gdal_gt[0]) / gdal_gt[1]) as isize;
            let gdal_py = ((gdal_gt[3] - lat) / (-gdal_gt[5])) as isize;

            // Read GDAL value
            let gdal_buf: gdal::raster::Buffer<f32> = band.read_as((gdal_px, gdal_py), (1, 1), (1, 1), None)
                .expect("GDAL read failed");
            let gdal_value = gdal_buf.data()[0];

            // Read our value via point query
            let our_result = cog.sample_lonlat(lon, lat).expect("Our read failed");
            let our_value = our_result.get(0).unwrap_or(f32::NAN);

            println!("{}: GDAL pixel=({}, {}) value={}, Our pixel={:?} value={}",
                name, gdal_px, gdal_py, gdal_value, our_result.pixel_coords, our_value);

            assert!((our_value - gdal_value).abs() < 0.001,
                "{}: Value mismatch - ours={}, gdal={}", name, our_value, gdal_value);
        }
    }

    /// Verify tile reading matches GDAL at tile boundaries
    #[test]
    fn test_gdal_tile_value_comparison() {
        let Some(cog) = get_test_cog() else { return };

        // Open with GDAL
        let gdal_ds = gdal::Dataset::open(TEST_COG_PATH).expect("GDAL failed to open");
        let band = gdal_ds.rasterband(1).expect("Failed to get band 1");

        // Read specific pixels and compare
        let test_pixels = [
            (0, 0),       // First pixel
            (1023, 0),    // End of first tile row
            (1024, 0),    // Start of second tile column
            (0, 1024),    // Start of second tile row
            (2736, 432),  // Berkeley Hills pixel
        ];

        for (px, py) in test_pixels {
            // Read GDAL value
            let gdal_buf: gdal::raster::Buffer<f32> = band.read_as((px as isize, py as isize), (1, 1), (1, 1), None)
                .expect("GDAL read failed");
            let gdal_value = gdal_buf.data()[0];

            // Read our value
            let our_value = cog.sample(0, px, py).expect("Our read failed").unwrap_or(f32::NAN);

            println!("Pixel ({}, {}): GDAL={}, Ours={}", px, py, gdal_value, our_value);

            assert!((our_value - gdal_value).abs() < 0.001,
                "Pixel ({}, {}): mismatch - ours={}, gdal={}", px, py, our_value, gdal_value);
        }
    }

    // ========== RGB COG Tests ==========

    const RGB_COG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/natural_earth_rgb.tif");

    fn get_rgb_cog() -> Option<CogReader> {
        if !std::path::Path::new(RGB_COG_PATH).exists() {
            println!("Skipping: RGB test file not found at {}", RGB_COG_PATH);
            return None;
        }
        let reader = crate::LocalRangeReader::new(RGB_COG_PATH).ok()?;
        CogReader::from_reader(Arc::new(reader)).ok()
    }

    #[test]
    fn test_rgb_cog_metadata() {
        let Some(cog) = get_rgb_cog() else { return };

        // Verify RGB COG has expected properties
        assert_eq!(cog.metadata.bands, 3, "Should have 3 bands (RGB)");
        assert_eq!(cog.metadata.crs_code, Some(4326));
        assert_eq!(cog.metadata.data_type, crate::CogDataType::UInt8);

        // Should have overviews (on CogReader, not CogMetadata)
        assert!(!cog.overviews.is_empty(), "Should have overviews");
        println!("RGB COG: {}x{}, {} bands, {} overviews",
            cog.metadata.width, cog.metadata.height,
            cog.metadata.bands, cog.overviews.len());
    }

    #[test]
    fn test_rgb_cog_gdal_metadata_comparison() {
        let Some(cog) = get_rgb_cog() else { return };

        let gdal_ds = gdal::Dataset::open(RGB_COG_PATH).expect("GDAL failed to open RGB COG");

        // Compare dimensions
        let (gdal_width, gdal_height) = gdal_ds.raster_size();
        assert_eq!(cog.metadata.width, gdal_width);
        assert_eq!(cog.metadata.height, gdal_height);

        // Compare band count
        assert_eq!(cog.metadata.bands, gdal_ds.raster_count());

        // Compare geotransform
        let gdal_gt = gdal_ds.geo_transform().expect("Failed to get geotransform");
        if let Some(scale) = &cog.metadata.geo_transform.pixel_scale {
            assert!((scale[0] - gdal_gt[1]).abs() < 1e-10,
                "X pixel scale mismatch: ours={}, gdal={}", scale[0], gdal_gt[1]);
        }
    }

    #[test]
    fn test_rgb_cog_multiband_pixel_values() {
        let Some(cog) = get_rgb_cog() else { return };

        let gdal_ds = gdal::Dataset::open(RGB_COG_PATH).expect("GDAL failed to open RGB COG");
        let (width, height) = gdal_ds.raster_size();

        // Test pixels at corners and center
        let test_pixels = [
            (0, 0),                           // Top-left corner
            (width / 2, height / 2),          // Center
            (width - 1, height - 1),          // Bottom-right corner
            (width / 4, height / 4),          // Quarter point
        ];

        for (px, py) in test_pixels {
            // Read all bands from GDAL
            for band_idx in 1..=cog.metadata.bands {
                let band = gdal_ds.rasterband(band_idx).expect("Failed to get band");
                let gdal_buf: gdal::raster::Buffer<u8> = band.read_as((px as isize, py as isize), (1, 1), (1, 1), None)
                    .expect("GDAL read failed");
                let gdal_value = gdal_buf.data()[0] as f32;

                // Read our value (band indices are 0-based)
                let our_value = cog.sample(band_idx - 1, px, py).expect("Our read failed").unwrap_or(f32::NAN);

                assert!((our_value - gdal_value).abs() < 0.01,
                    "Pixel ({}, {}) band {}: mismatch - ours={}, gdal={}",
                    px, py, band_idx, our_value, gdal_value);
            }
        }
    }

    #[test]
    fn test_rgb_cog_overview_dimensions() {
        let Some(cog) = get_rgb_cog() else { return };

        // Verify overviews exist and dimensions decrease
        assert!(!cog.overviews.is_empty(), "Should have overviews");

        let mut prev_width = cog.metadata.width;
        let mut prev_height = cog.metadata.height;

        for (i, overview) in cog.overviews.iter().enumerate() {
            assert!(overview.width < prev_width,
                "Overview {} width should be smaller than previous", i);
            assert!(overview.height < prev_height,
                "Overview {} height should be smaller than previous", i);
            prev_width = overview.width;
            prev_height = overview.height;
        }
    }
}

#[cfg(test)]
mod async_open_tests {
    use super::*;

    // Async open / spawn_blocking (local synthetic file, no network)

    /// Write a small single-band 3857 GeoTIFF and return its temp dir + path.
    fn write_small_tiff() -> (tempfile::TempDir, String) {
        use crate::geotiff_writer::GeoTiffCompression;
        use crate::xyz_tile::{BoundingBox, ReprojectedRaster};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.tif");
        let (w, h) = (64usize, 64usize);
        let raster = ReprojectedRaster {
            pixels: (0..w * h).map(|i| 1.0 + (i % 200) as f32).collect(),
            bands: 1,
            width: w,
            height: h,
            crs: 3857,
            bounds: BoundingBox::new(0.0, 0.0, 6400.0, 6400.0),
            resolution: (100.0, 100.0),
            nodata: None,
        };
        raster.write_geotiff_compressed(&path, GeoTiffCompression::Deflate).unwrap();
        let path = path.to_str().unwrap().to_string();
        (dir, path)
    }

    async fn check_open_async() {
        let (_dir, path) = write_small_tiff();
        let reader = CogReader::open_async(&path).await.unwrap();
        assert_eq!((reader.metadata.width, reader.metadata.height), (64, 64));

        let hinted = CogReader::open_async_with_hint(&path, OverviewQualityHint::NoneUsable)
            .await
            .unwrap();
        assert_eq!(hinted.metadata.width, 64);

        let width = reader.spawn_blocking(|r| Ok(r.metadata.width)).await.unwrap();
        assert_eq!(width, 64);

        let err = CogReader::open_async("/nonexistent/definitely/missing.tif").await;
        assert!(err.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_open_async_multi_thread() {
        check_open_async().await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_open_async_current_thread() {
        check_open_async().await;
    }

    // Remote open over a local HTTP server (no external network).

    use crate::test_support::{build_cog, serve_bytes, CogSpec, Sample};
    use std::time::{Duration, Instant};

    fn http_spec(width: usize, tile: usize, overviews: usize) -> CogSpec {
        CogSpec {
            width,
            height: width,
            tile,
            bands: 1,
            sample: Sample::U8,
            deflate: true,
            predictor: false,
            epsg: 3857,
            origin: (0.0, 1000.0),
            pixel_size: (1.0, 1.0),
            nodata: None,
            overviews,
            sparse: vec![],
            corrupt: vec![],
            pixel: |_, x, y| ((x + y) % 250) as f64 + 1.0,
        }
    }

    fn assert_same_structure(a: &CogReader, b: &CogReader) {
        assert_eq!(a.metadata.tile_offsets, b.metadata.tile_offsets);
        assert_eq!(a.metadata.tile_byte_counts, b.metadata.tile_byte_counts);
        assert_eq!(a.metadata.geo_transform.pixel_scale, b.metadata.geo_transform.pixel_scale);
        assert_eq!(a.metadata.crs_code, b.metadata.crs_code);
        assert_eq!(a.overviews.len(), b.overviews.len());
        for (x, y) in a.overviews.iter().zip(b.overviews.iter()) {
            assert_eq!((x.width, x.scale, &x.tile_offsets), (y.width, y.scale, &y.tile_offsets));
        }
    }

    #[tokio::test]
    async fn open_async_over_http_costs_one_request_and_matches_local_parse() {
        let bytes = build_cog(&http_spec(256, 64, 2));
        let (base, log) = serve_bytes(bytes.clone(), Duration::ZERO);

        let remote = CogReader::open_async_with_hint(&format!("{base}/x.tif"), OverviewQualityHint::NoneUsable)
            .await
            .unwrap();
        assert_eq!(log.lock().len(), 1, "{:?}", log.lock());
        assert!(remote.is_remote());

        let local = CogReader::from_reader_with_hint(
            Arc::new(crate::range_reader::MemoryRangeReader::new(bytes, "mem://open-parity".into())),
            OverviewQualityHint::NoneUsable,
        )
        .unwrap();
        assert_same_structure(&remote, &local);
    }

    #[tokio::test]
    async fn open_async_fetches_large_tag_arrays_concurrently() {
        // 16384 tiles: the offset and byte-count arrays (64 KiB each) lie beyond the prefix.
        let delay = Duration::from_millis(150);
        let bytes = build_cog(&http_spec(2048, 16, 0));
        let (base, log) = serve_bytes(bytes.clone(), delay);

        let start = Instant::now();
        let remote = CogReader::open_async_with_hint(&format!("{base}/big.tif"), OverviewQualityHint::NoneUsable)
            .await
            .unwrap();
        let elapsed = start.elapsed();
        assert!(log.lock().len() >= 3, "arrays were not fetched remotely: {:?}", log.lock());
        // prefix round trip + one round trip for all the arrays together
        assert!(elapsed < delay * 5 / 2, "open took {elapsed:?}; arrays are being fetched serially");

        let local = CogReader::from_reader_with_hint(
            Arc::new(crate::range_reader::MemoryRangeReader::new(bytes, "mem://open-parity-big".into())),
            OverviewQualityHint::NoneUsable,
        )
        .unwrap();
        assert_same_structure(&remote, &local);
    }

    #[tokio::test]
    async fn computed_overview_hint_matches_between_sync_and_async_paths() {
        let bytes = build_cog(&http_spec(512, 64, 3));
        let (base, _log) = serve_bytes(bytes.clone(), Duration::ZERO);
        let remote = CogReader::open_async(&format!("{base}/h.tif")).await.unwrap();
        let local = CogReader::from_reader(Arc::new(crate::range_reader::MemoryRangeReader::new(
            bytes,
            "mem://hint-parity".into(),
        )))
        .unwrap();
        assert_eq!(remote.min_usable_overview, local.min_usable_overview);
        assert_eq!(
            remote.compute_overview_quality_hint_async().await,
            local.compute_overview_quality_hint()
        );
    }

    #[tokio::test]
    async fn read_tile_async_matches_sync_read() {
        let bytes = build_cog(&http_spec(256, 64, 1));
        let (base, _log) = serve_bytes(bytes.clone(), Duration::ZERO);
        let remote = CogReader::open_async_with_hint(&format!("{base}/t.tif"), OverviewQualityHint::NoneUsable)
            .await
            .unwrap();
        let local = CogReader::from_reader_with_hint(
            Arc::new(crate::range_reader::MemoryRangeReader::new(bytes, "mem://tile-parity".into())),
            OverviewQualityHint::NoneUsable,
        )
        .unwrap();
        for idx in [0, 5, 15] {
            assert_eq!(remote.read_tile_async(idx).await.unwrap(), local.read_tile(idx).unwrap());
        }
        assert_eq!(
            remote.read_overview_tile_async(0, 1).await.unwrap(),
            local.read_overview_tile(0, 1).unwrap()
        );
        assert!(remote.read_tile_async(999).await.is_err());
    }

    #[test]
    fn sync_open_of_a_remote_source_works_from_a_plain_thread() {
        let bytes = build_cog(&http_spec(256, 64, 1));
        let (base, _log) = serve_bytes(bytes, Duration::ZERO);
        let reader = CogReader::open_with_hint(&format!("{base}/p.tif"), OverviewQualityHint::NoneUsable).unwrap();
        assert_eq!(reader.metadata.width, 256);
        // The sync read API works on a remotely opened reader too.
        assert_eq!(reader.read_tile(0).unwrap().len(), 64 * 64);
    }

    /// Opening synchronously from inside a runtime blocks that worker but must neither panic
    /// nor deadlock, including on a single-threaded runtime.
    #[tokio::test(flavor = "current_thread")]
    async fn sync_open_of_a_remote_source_inside_a_current_thread_runtime() {
        let bytes = build_cog(&http_spec(256, 64, 1));
        let (base, _log) = serve_bytes(bytes, Duration::from_millis(20));
        let url = format!("{base}/c.tif");
        let reader = CogReader::open_with_hint(&url, OverviewQualityHint::NoneUsable).unwrap();
        assert_eq!(reader.metadata.width, 256);
        let tile = reader.read_tile(1).unwrap();
        assert_eq!(tile.len(), 64 * 64);
        // ... and with the runtime-measured hint (reads sample tiles).
        let computed = CogReader::open(&url).unwrap();
        assert_eq!(computed.metadata.height, 256);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sync_open_of_a_remote_source_inside_a_multi_thread_runtime() {
        let bytes = build_cog(&http_spec(256, 64, 1));
        let (base, _log) = serve_bytes(bytes, Duration::from_millis(20));
        let reader = CogReader::open_with_hint(&format!("{base}/m.tif"), OverviewQualityHint::AllUsable).unwrap();
        assert_eq!(reader.min_usable_overview, Some(0));
    }

    /// A reader opened with `open_async` on one runtime can be used through the sync API from a
    /// `spawn_blocking` thread (the sync read runs on the private I/O runtime).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sync_reads_on_an_async_opened_reader_from_a_blocking_thread() {
        let bytes = build_cog(&http_spec(256, 64, 1));
        let (base, _log) = serve_bytes(bytes, Duration::from_millis(10));
        let reader = CogReader::open_async_with_hint(&format!("{base}/mix.tif"), OverviewQualityHint::AllUsable)
            .await
            .unwrap();
        let via_blocking = reader.spawn_blocking(|r| r.read_tile(3)).await.unwrap();
        let via_async = reader.read_tile_async(3).await.unwrap();
        assert_eq!(via_blocking, via_async);
    }

    #[test]
    fn clones_share_metadata() {
        let bytes = build_cog(&http_spec(128, 64, 1));
        let a = CogReader::from_reader(Arc::new(crate::range_reader::MemoryRangeReader::new(
            bytes,
            "mem://clone-share".into(),
        )))
        .unwrap();
        let b = a.clone();
        assert!(Arc::ptr_eq(&a.metadata, &b.metadata));
        assert!(Arc::ptr_eq(&a.overviews, &b.overviews));
    }

    /// Decoded tiles are keyed by the file's version: a file rewritten in place (same path) is
    /// never served from tiles decoded from its previous contents.
    #[test]
    fn rewritten_local_file_gets_fresh_tiles() {
        use crate::geotiff_writer::GeoTiffCompression;
        use crate::xyz_tile::{BoundingBox, ReprojectedRaster};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rewrite.tif");
        let write = |base: f32| {
            let raster = ReprojectedRaster {
                pixels: (0..64 * 64).map(|i| base + (i % 200) as f32).collect(),
                bands: 1,
                width: 64,
                height: 64,
                crs: 3857,
                bounds: BoundingBox::new(0.0, 0.0, 6400.0, 6400.0),
                resolution: (100.0, 100.0),
                nodata: None,
            };
            raster.write_geotiff_compressed(&path, GeoTiffCompression::Deflate).unwrap();
        };
        let source = path.to_str().unwrap();

        write(1.0);
        let first = CogReader::open_with_hint(source, OverviewQualityHint::NoneUsable).unwrap();
        let before = first.read_tile(0).unwrap();

        write(1001.0);
        // Equal sizes and a coarse file system clock could hide the rewrite: force a new mtime.
        let file = std::fs::File::options().write(true).open(&path).unwrap();
        file.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(5)).unwrap();
        let second = CogReader::open_with_hint(source, OverviewQualityHint::NoneUsable).unwrap();
        let after = second.read_tile(0).unwrap();

        assert_ne!(first.cache_id(), second.cache_id());
        assert!((after[0] - before[0] - 1000.0).abs() < 1e-3, "{} then {}", before[0], after[0]);
    }
}
