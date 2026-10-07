# cogrs

Pure Rust COG (Cloud Optimized `GeoTIFF`) reader library.

## Features

- [Local, HTTP, and S3 sources](#sources)
- [Point queries](#point-queries)
- [XYZ tile extraction](#tile-extraction)
- [Coordinate transforms](#coordinate-transforms)
- [Compression: DEFLATE, LZW, ZSTD, JPEG, WebP](#compression)

## Quick Start

```rust,no_run
use cogrs::{CogReader, PointQuery, TileExtractor};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let reader = CogReader::open_async("path/to/file.tif").await?;

    // Point query (sync I/O; use `reader.spawn_blocking(..)` for remote sources)
    let result = reader.sample_lonlat(-122.4, 37.8)?;

    // XYZ tile extraction (async)
    let tile = TileExtractor::new(&reader)
        .xyz(10, 163, 395)
        .extract()
        .await?;

    Ok(())
}
```

## Sources

From async code (axum/tokio handlers, etc.) always open with `CogReader::open_async`. It runs
the blocking header/IFD reads on tokio's blocking pool and works on both `multi_thread` and
`current_thread` runtimes. S3 and HTTP sources need it: the synchronous `CogReader::open`
blocks the calling thread. Called on an async worker thread it is converted to an error
only on a best-effort basis: with unwinding panics tokio's panic message is still printed
and an error naming `open_async` is returned; with `panic = "abort"` the process aborts;
and in release builds the blocking HTTP reader may silently block the worker thread
instead. Always use `open_async` from async code.

```rust,no_run
use cogrs::CogReader;
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
// Local file
let reader = CogReader::open_async("path/to/file.tif").await?;

// HTTP
let reader = CogReader::open_async("https://example.com/file.tif").await?;

// S3 (uses AWS_* environment variables for credentials)
let reader = CogReader::open_async("s3://bucket/path/to/file.tif").await?;
# Ok(())
# }
```

S3 configuration: `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` for credentials,
`AWS_SKIP_SIGNATURE=true` for anonymous access to public buckets, `AWS_ENDPOINT_URL` /
`AWS_ALLOW_HTTP` for MinIO and other S3-compatible stores. The region is taken from
`AWS_REGION`, then `AWS_DEFAULT_REGION`; if neither is set it is detected from the bucket
(once per bucket per process) unless a custom endpoint is configured.

Blocking APIs (`CogReader::open`, `sample_lonlat`, ...) are fine in plain threads and inside
`spawn_blocking`. To call them from async code on an opened reader, use
`reader.spawn_blocking(|r| r.sample_lonlat(lon, lat)).await`; tile extraction
(`TileExtractor::extract`) is already async.

## Point Queries

```rust,no_run
use cogrs::{CogReader, PointQuery};
# fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
let reader = CogReader::open("elevation.tif")?;

// Sample at lon/lat
let result = reader.sample_lonlat(-122.4, 37.8)?;
for (band, value) in &result.values {
    println!("Band {band}: {value}");
}

// Sample in specific CRS (e.g., UTM zone 10N)
let result = reader.sample_crs(32610, 551000.0, 4185000.0)?;
# Ok(())
# }
```

## Tile Extraction

```rust,no_run
use cogrs::{CogReader, TileExtractor, ResamplingMethod};
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
let reader = CogReader::open_async("imagery.tif").await?;

// Simple extraction (256x256)
let tile = TileExtractor::new(&reader)
    .xyz(10, 163, 395)
    .extract()
    .await?;

// With options
let tile = TileExtractor::new(&reader)
    .xyz(10, 163, 395)
    .output_size(512, 512)
    .resampling(ResamplingMethod::Bilinear)
    .bands(&[0, 1, 2])
    .extract()
    .await?;
# Ok(())
# }
```

## WebP Output

Extracted tiles encode to lossless RGBA8 WebP (pure Rust). Pixels equal to the
COG's nodata value (in all bands) or NaN become transparent; 1-band tiles are
expanded to gray, 3-band to RGB, 4-band keeps its alpha. Values are rounded and
clamped to `0..=255`; use `WebpOptions::rescale` for 16-bit/float data.
Areas of a tile outside the COG extent are filled with the COG's nodata value
(or `NaN` if it declares none), so edge tiles come out transparent there.

```rust,no_run
use cogrs::{CogReader, TileExtractor, WebpOptions};
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
let reader = CogReader::open_async("imagery.tif").await?;
let tile = TileExtractor::new(&reader).xyz(10, 163, 395).extract().await?;

let webp: Vec<u8> = tile.to_webp()?;

// 16-bit source: map 0..3000 linearly onto 0..255, override nodata
let opts = WebpOptions { rescale: Some((0.0, 3000.0)), nodata: Some(0.0) };
let webp = tile.to_webp_with(&opts)?;
# Ok(())
# }
```

## Coordinate Transforms

```rust
use cogrs::{CoordTransformer, project_point};

// One-off transform
let (x, y) = project_point(4326, 3857, -122.4, 37.8)?;

// Reusable transformer
let transformer = CoordTransformer::new(4326, 32610)?;
let (utm_x, utm_y) = transformer.transform(-122.4, 37.8)?;
# Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
```

## Compression

Supported formats are detected automatically:

- DEFLATE
- LZW (8/16-bit, predictors 1-3)
- ZSTD
- JPEG
- WebP

## License

MIT
