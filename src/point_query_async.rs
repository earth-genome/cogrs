//! Asynchronous point queries.
//!
//! The [`PointQuery`](crate::PointQuery) trait is synchronous: reading a tile blocks the calling
//! thread. These `*_async` counterparts on [`CogReader`] fetch tiles containing the
//! points without holding a thread while waiting (decoding runs on tokio's blocking pool) and
//! share the process-wide tile cache and in-flight de-duplication with tile extraction.
//!
//! A batch groups points by full-resolution tile index, de-duplicates, and calls
//! [`crate::tile_fetch::fetch_tiles`] once per chunk of unique tiles.

use std::collections::HashMap;
use std::sync::Arc;

use crate::cog_reader::{CogReader, TileRef};
use crate::point_query::PointQueryResult;
use crate::tiff_utils::AnyResult;

/// Unique tiles fetched together; sequential chunks keep in-flight work bounded.
const TILE_FETCH_CHUNK: usize = 64;

impl CogReader {
    /// Async counterpart of [`CogReader::sample`]: one pixel value, or `None` outside the image.
    ///
    /// # Errors
    /// Returns an error if reading or decompressing the tile containing the pixel fails.
    pub async fn sample_async(&self, band: usize, x: usize, y: usize) -> AnyResult<Option<f32>> {
        self.with_revalidation(|reader| Box::pin(reader.sample_once(band, x, y))).await
    }

    async fn sample_once(&self, band: usize, x: usize, y: usize) -> AnyResult<Option<f32>> {
        let Some(tile_index) = self.metadata.tile_index_for_pixel(x, y) else {
            return Ok(None);
        };
        let (tile, _) = self.read_tile_async_ref(TileRef { overview: None, index: tile_index }).await?;
        Ok(self.value_in_tile(&tile, tile_index, band, x, y))
    }

    // The pixel lookup, per-band extraction and result construction are shared with the
    // synchronous `PointQuery` implementation (`locate_pixel`, `point_result`, ... in
    // `point_query.rs`); only the tile read is asynchronous here.

    /// Async counterpart of [`PointQuery::sample_crs`](crate::PointQuery::sample_crs): sample all
    /// bands at coordinates in `crs`.
    ///
    /// # Errors
    /// Returns an error if coordinate projection fails or if reading the tile fails.
    pub async fn sample_crs_async(&self, crs: i32, x: f64, y: f64) -> AnyResult<PointQueryResult> {
        self.with_revalidation(|reader| Box::pin(reader.sample_crs_once(crs, x, y))).await
    }

    async fn sample_crs_once(&self, crs: i32, x: f64, y: f64) -> AnyResult<PointQueryResult> {
        let Some(pixel) = self.locate_pixel(crs, x, y)? else {
            return Ok(self.empty_point_result(crs));
        };
        let Some(tile_index) = self.metadata.tile_index_for_pixel(pixel.0, pixel.1) else {
            return Ok(self.empty_point_result(crs));
        };

        let (tile, _) = self.read_tile_async_ref(TileRef { overview: None, index: tile_index }).await?;
        Ok(self.point_result(crs, pixel, &tile, tile_index))
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
    /// unique tiles are fetched together (chunks of 64), results keep the input order.
    ///
    /// # Errors
    /// Returns the first error from projecting a point or reading a tile.
    pub async fn sample_points_crs_async(&self, crs: i32, points: &[(f64, f64)]) -> AnyResult<Vec<PointQueryResult>> {
        let points = Arc::<[_]>::from(points);
        self.with_revalidation({
            let points = Arc::clone(&points);
            move |reader| {
                let points = Arc::clone(&points);
                Box::pin(async move { reader.sample_points_crs_once(crs, &points).await })
            }
        })
        .await
    }

    async fn sample_points_crs_once(&self, crs: i32, points: &[(f64, f64)]) -> AnyResult<Vec<PointQueryResult>> {
        let (located, unique) = self.locate_batch(crs, points)?;
        let mut tiles = HashMap::with_capacity(unique.len());
        for chunk in unique.chunks(TILE_FETCH_CHUNK) {
            let fetched = crate::tile_fetch::fetch_tiles(self, None, chunk).await?;
            tiles.extend(fetched.tiles);
        }
        self.resolve_batch(crs, located, &tiles)
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
    async fn batch_sampling_is_deduped_and_ordered() {
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
        let distinct_tiles: std::collections::HashSet<_> = mock.calls().into_iter().collect();
        assert_eq!(mock.call_count(), distinct_tiles.len(), "{:?}", mock.calls());
    }
}
