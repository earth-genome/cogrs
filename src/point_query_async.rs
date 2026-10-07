//! Asynchronous point queries.
//!
//! The [`PointQuery`](crate::PointQuery) trait is synchronous: reading a tile blocks the calling
//! thread. These `*_async` counterparts on [`CogReader`] fetch the one tile containing the
//! point without holding a thread while waiting (decoding runs on tokio's blocking pool) and
//! share the process-wide tile cache and in-flight de-duplication with tile extraction.
//!
//! All bands of a point come from a single tile fetch.

use std::collections::HashMap;

use futures::{StreamExt, TryStreamExt};

use crate::cog_reader::{CogReader, TileRef};
use crate::geometry::projection::project_point;
use crate::point_query::PointQueryResult;
use crate::tiff_utils::AnyResult;

/// How many points of a batch are looked up concurrently.
const BATCH_CONCURRENCY: usize = 16;

impl CogReader {
    /// Async counterpart of [`CogReader::sample`]: one pixel value, or `None` outside the image.
    ///
    /// # Errors
    /// Returns an error if reading or decompressing the tile containing the pixel fails.
    pub async fn sample_async(&self, band: usize, x: usize, y: usize) -> AnyResult<Option<f32>> {
        let Some(tile_index) = self.metadata.tile_index_for_pixel(x, y) else {
            return Ok(None);
        };
        let (tile, _) = self.read_tile_async_ref(TileRef { overview: None, index: tile_index }).await?;
        Ok(self.value_in_tile(&tile, tile_index, band, x, y))
    }

    /// The sample for `(band, x, y)` within the decoded tile `tile_index`.
    fn value_in_tile(&self, tile: &[f32], tile_index: usize, band: usize, x: usize, y: usize) -> Option<f32> {
        let meta = &self.metadata;
        let tile_col = tile_index % meta.tiles_across;
        let tile_row = tile_index / meta.tiles_across;
        let local_x = x - tile_col * meta.tile_width;
        let local_y = y - tile_row * meta.tile_height;
        tile.get((local_y * meta.tile_width + local_x) * meta.bands + band).copied()
    }

    /// Source pixel `(x, y)` for a coordinate in `crs`, or `None` outside the raster.
    fn locate_pixel(&self, crs: i32, x: f64, y: f64) -> AnyResult<Option<(usize, usize)>> {
        let source_crs = self.metadata.crs_code.unwrap_or(4326);
        let (src_x, src_y) = if crs == source_crs { (x, y) } else { project_point(crs, source_crs, x, y)? };
        let Some((px, py)) = self.metadata.geo_transform.world_to_pixel(src_x, src_y) else {
            return Ok(None);
        };
        // Allow cast precision loss: bounds checking only needs approximate precision
        #[allow(clippy::cast_precision_loss)]
        if px < 0.0 || py < 0.0 || px >= self.metadata.width as f64 || py >= self.metadata.height as f64 {
            return Ok(None);
        }
        // Cast is safe: already bounds-checked against width/height
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Ok(Some((px as usize, py as usize)))
    }

    /// Async counterpart of [`PointQuery::sample_crs`](crate::PointQuery::sample_crs): sample all
    /// bands at coordinates in `crs`.
    ///
    /// # Errors
    /// Returns an error if coordinate projection fails or if reading the tile fails.
    pub async fn sample_crs_async(&self, crs: i32, x: f64, y: f64) -> AnyResult<PointQueryResult> {
        let source_crs = self.metadata.crs_code.unwrap_or(4326);
        let invalid = || PointQueryResult {
            values: HashMap::new(),
            bands: self.metadata.bands,
            is_valid: false,
            pixel_coords: None,
            input_crs: crs,
            raster_crs: source_crs,
        };
        let Some((pixel_x, pixel_y)) = self.locate_pixel(crs, x, y)? else {
            return Ok(invalid());
        };
        let Some(tile_index) = self.metadata.tile_index_for_pixel(pixel_x, pixel_y) else {
            return Ok(invalid());
        };

        let (tile, _) = self.read_tile_async_ref(TileRef { overview: None, index: tile_index }).await?;
        let values = (0..self.metadata.bands)
            .map(|band| (band, self.value_in_tile(&tile, tile_index, band, pixel_x, pixel_y).unwrap_or(f32::NAN)))
            .collect();

        Ok(PointQueryResult {
            values,
            bands: self.metadata.bands,
            is_valid: true,
            pixel_coords: Some((pixel_x, pixel_y)),
            input_crs: crs,
            raster_crs: source_crs,
        })
    }

    /// Async counterpart of [`PointQuery::sample_lonlat`](crate::PointQuery::sample_lonlat).
    ///
    /// # Errors
    /// Returns an error if coordinate projection fails or if reading the tile fails.
    pub async fn sample_lonlat_async(&self, lon: f64, lat: f64) -> AnyResult<PointQueryResult> {
        self.sample_crs_async(4326, lon, lat).await
    }

    /// Async counterpart of [`PointQuery::sample_band_crs`](crate::PointQuery::sample_band_crs).
    ///
    /// # Errors
    /// Returns an error if coordinate projection fails or if reading the tile fails.
    pub async fn sample_band_crs_async(&self, crs: i32, x: f64, y: f64, band: usize) -> AnyResult<Option<f32>> {
        if band >= self.metadata.bands {
            return Ok(None);
        }
        match self.locate_pixel(crs, x, y)? {
            Some((px, py)) => self.sample_async(band, px, py).await,
            None => Ok(None),
        }
    }

    /// Async counterpart of [`PointQuery::sample_points_crs`](crate::PointQuery::sample_points_crs):
    /// points are looked up concurrently (up to 16 at a time), results keep the input order.
    ///
    /// # Errors
    /// Returns the first error from projecting a point or reading a tile.
    pub async fn sample_points_crs_async(&self, crs: i32, points: &[(f64, f64)]) -> AnyResult<Vec<PointQueryResult>> {
        futures::stream::iter(points.iter().map(|&(x, y)| self.sample_crs_async(crs, x, y)))
            .buffered(BATCH_CONCURRENCY)
            .try_collect()
            .await
    }

    /// Async counterpart of [`PointQuery::sample_points_lonlat`](crate::PointQuery::sample_points_lonlat).
    ///
    /// # Errors
    /// Returns the first error from projecting a point or reading a tile.
    pub async fn sample_points_lonlat_async(&self, points: &[(f64, f64)]) -> AnyResult<Vec<PointQueryResult>> {
        self.sample_points_crs_async(4326, points).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use crate::test_support::{build_cog, CogSpec, MockReader, Sample};
    use crate::{CogReader, MemoryRangeReader, OverviewQualityHint, PointQuery};

    fn spec() -> CogSpec {
        CogSpec {
            width: 300,
            height: 200,
            tile: 64,
            bands: 3,
            sample: Sample::U8,
            deflate: true,
            predictor: true,
            epsg: 4326,
            origin: (10.0, 50.0),
            pixel_size: (0.01, 0.01),
            nodata: None,
            overviews: 1,
            sparse: vec![(0, 3)],
            corrupt: vec![],
            pixel: |b, x, y| ((x * 7 + y * 13 + b * 50) % 250) as f64 + 1.0,
        }
    }

    fn sorted(r: &crate::PointQueryResult) -> Vec<(usize, u32)> {
        let mut v: Vec<_> = r.values.iter().map(|(b, v)| (*b, v.to_bits())).collect();
        v.sort_unstable();
        v
    }

    #[tokio::test(start_paused = true)]
    async fn async_samples_match_the_sync_api_and_cost_one_request_per_point() {
        let mock = Arc::new(MockReader::new(build_cog(&spec()), "mock://pq/remote", Duration::from_millis(20)));
        let remote = CogReader::from_async_reader_with_hint(mock.clone(), OverviewQualityHint::NoneUsable).await.unwrap();
        let local = CogReader::from_reader_with_hint(
            Arc::new(MemoryRangeReader::new(build_cog(&spec()), "mem://pq/local".into())),
            OverviewQualityHint::NoneUsable,
        )
        .unwrap();
        mock.reset();

        // Inside tiles (incl. the sparse one), on tile edges, and outside the raster.
        let points = [(10.05, 49.95), (10.3, 49.5), (10.0, 50.0), (11.5, 49.1), (12.0, 49.0), (9.0, 50.0), (10.5, 52.0)];
        for &(lon, lat) in &points {
            let a = remote.sample_lonlat_async(lon, lat).await.unwrap();
            let s = local.sample_lonlat(lon, lat).unwrap();
            assert_eq!((a.is_valid, a.pixel_coords, a.bands), (s.is_valid, s.pixel_coords, s.bands), "{lon},{lat}");
            assert_eq!(sorted(&a), sorted(&s), "{lon},{lat}");
            assert_eq!(
                remote.sample_band_crs_async(4326, lon, lat, 1).await.unwrap().map(f32::to_bits),
                local.sample_band_lonlat(lon, lat, 1).unwrap().map(f32::to_bits)
            );
        }
        // Each distinct tile was fetched once however many bands/points hit it.
        let distinct_tiles: std::collections::HashSet<_> = mock.calls().into_iter().collect();
        assert_eq!(mock.call_count(), distinct_tiles.len(), "{:?}", mock.calls());
    }

    #[tokio::test(start_paused = true)]
    async fn batch_sampling_is_concurrent_and_ordered() {
        let latency = Duration::from_millis(100);
        let mock = Arc::new(MockReader::new(build_cog(&spec()), "mock://pq/batch", latency));
        let remote = CogReader::from_async_reader_with_hint(mock.clone(), OverviewQualityHint::NoneUsable).await.unwrap();
        let local = CogReader::from_reader_with_hint(
            Arc::new(MemoryRangeReader::new(build_cog(&spec()), "mem://pq/batch".into())),
            OverviewQualityHint::NoneUsable,
        )
        .unwrap();
        mock.reset();

        let points: Vec<(f64, f64)> = (0..12).map(|i| (10.0 + f64::from(i) * 0.24, 49.9 - f64::from(i % 3) * 0.6)).collect();
        let t0 = tokio::time::Instant::now();
        let got = remote.sample_points_lonlat_async(&points).await.unwrap();
        assert!(t0.elapsed() < latency * 3, "took {:?}", t0.elapsed());
        let want = local.sample_points_lonlat(&points).unwrap();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert_eq!((g.pixel_coords, sorted(g)), (w.pixel_coords, sorted(w)));
        }
        assert!(mock.max_in_flight() > 1);
    }
}
