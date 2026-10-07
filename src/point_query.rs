//! Point query functionality for sampling COG values at geographic coordinates.
//!
//! This module provides the [`PointQuery`] trait for sampling pixel values at geographic
//! coordinates, with support for any coordinate reference system (CRS).
//!
//! # Example
//!
//! ```rust,no_run
//! use cogrs::{CogReader, PointQuery};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//!     let reader = CogReader::open("path/to/cog.tif")?;
//!
//!     // Sample all bands at a lon/lat coordinate
//!     let result = reader.sample_lonlat(-122.4, 37.8)?;
//!     for (band, value) in &result.values {
//!         println!("Band {}: {}", band, value);
//!     }
//!
//!     // Sample at coordinates in a specific CRS (e.g., UTM zone 10N)
//!     let result = reader.sample_crs(32610, 551000.0, 4185000.0)?;
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;

use crate::cog_reader::{CogReader, TileRef};
use crate::geometry::projection::project_point;
use crate::tiff_utils::AnyResult;

/// Result of a point query containing band values as a `HashMap`.
///
/// The `values` field maps band index (0-based) to the sampled value.
/// This makes it easy to access specific bands and iterate over all values.
#[derive(Debug, Clone)]
pub struct PointQueryResult {
    /// Values for each band, keyed by band index (0-based).
    /// Values are `f32::NAN` if the pixel is nodata.
    pub values: HashMap<usize, f32>,

    /// Number of bands in the source raster
    pub bands: usize,

    /// Whether this point was within the raster bounds
    pub is_valid: bool,

    /// The pixel coordinates that were sampled (if valid)
    pub pixel_coords: Option<(usize, usize)>,

    /// The CRS of the input coordinates
    pub input_crs: i32,

    /// The native CRS of the raster
    pub raster_crs: i32,
}

impl PointQueryResult {
    /// Get the value for a specific band, or None if band doesn't exist
    #[inline]
    #[must_use] 
    pub fn get(&self, band: usize) -> Option<f32> {
        self.values.get(&band).copied()
    }

    /// Get all values as a Vec, ordered by band index
    #[must_use] 
    pub fn to_vec(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.bands);
        for i in 0..self.bands {
            result.push(self.values.get(&i).copied().unwrap_or(f32::NAN));
        }
        result
    }

    /// Check if any band has a valid (non-NaN) value
    #[inline]
    #[must_use] 
    pub fn has_valid_data(&self) -> bool {
        self.values.values().any(|v| !v.is_nan())
    }

    /// Get the number of bands with valid (non-NaN) values
    #[must_use] 
    pub fn valid_band_count(&self) -> usize {
        self.values.values().filter(|v| !v.is_nan()).count()
    }

    /// Iterate over (`band_index`, value) pairs
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &f32)> {
        self.values.iter()
    }
}

/// Trait for point queries at geographic coordinates.
///
/// This trait extends [`CogReader`] with methods to sample pixel values
/// at geographic coordinates in any supported CRS.
pub trait PointQuery {
    /// Sample all bands at a lon/lat coordinate (EPSG:4326).
    ///
    /// # Arguments
    /// * `lon` - Longitude in degrees (-180 to 180)
    /// * `lat` - Latitude in degrees (-90 to 90)
    ///
    /// # Returns
    /// A [`PointQueryResult`] containing values for all bands.
    ///
    /// # Errors
    /// Returns an error if coordinate projection fails or if reading from the raster source fails.
    fn sample_lonlat(&self, lon: f64, lat: f64) -> AnyResult<PointQueryResult>;

    /// Sample all bands at coordinates in a specific CRS.
    ///
    /// # Arguments
    /// * `crs` - EPSG code of the input coordinates
    /// * `x` - X coordinate in the specified CRS
    /// * `y` - Y coordinate in the specified CRS
    ///
    /// # Returns
    /// A [`PointQueryResult`] containing values for all bands.
    ///
    /// # Errors
    /// Returns an error if coordinate projection fails or if reading from the raster source fails.
    fn sample_crs(&self, crs: i32, x: f64, y: f64) -> AnyResult<PointQueryResult>;

    /// Sample a single band at lon/lat coordinates (EPSG:4326).
    ///
    /// # Arguments
    /// * `lon` - Longitude in degrees
    /// * `lat` - Latitude in degrees
    /// * `band` - Band index (0-based)
    ///
    /// # Returns
    /// The sampled value, or None if out of bounds or band doesn't exist.
    ///
    /// # Errors
    /// Returns an error if coordinate projection fails or if reading from the raster source fails.
    fn sample_band_lonlat(&self, lon: f64, lat: f64, band: usize) -> AnyResult<Option<f32>>;

    /// Sample a single band at coordinates in a specific CRS.
    ///
    /// # Arguments
    /// * `crs` - EPSG code of the input coordinates
    /// * `x` - X coordinate in the specified CRS
    /// * `y` - Y coordinate in the specified CRS
    /// * `band` - Band index (0-based)
    ///
    /// # Returns
    /// The sampled value, or None if out of bounds or band doesn't exist.
    ///
    /// # Errors
    /// Returns an error if coordinate projection fails or if reading from the raster source fails.
    fn sample_band_crs(&self, crs: i32, x: f64, y: f64, band: usize) -> AnyResult<Option<f32>>;

    /// Sample multiple points at once (batch query) for efficiency.
    ///
    /// # Arguments
    /// * `points` - Slice of (lon, lat) coordinates in EPSG:4326
    ///
    /// # Returns
    /// A Vec of [`PointQueryResult`], one for each input point.
    ///
    /// # Errors
    /// Returns an error if coordinate projection fails or if reading from the raster source fails.
    fn sample_points_lonlat(&self, points: &[(f64, f64)]) -> AnyResult<Vec<PointQueryResult>>;

    /// Sample multiple points in a specific CRS.
    ///
    /// # Arguments
    /// * `crs` - EPSG code of the input coordinates
    /// * `points` - Slice of (x, y) coordinates in the specified CRS
    ///
    /// # Returns
    /// A Vec of [`PointQueryResult`], one for each input point.
    ///
    /// # Errors
    /// Returns an error if coordinate projection fails or if reading from the raster source fails.
    fn sample_points_crs(&self, crs: i32, points: &[(f64, f64)]) -> AnyResult<Vec<PointQueryResult>>;
}

impl PointQuery for CogReader {
    fn sample_lonlat(&self, lon: f64, lat: f64) -> AnyResult<PointQueryResult> {
        self.sample_crs(4326, lon, lat)
    }

    fn sample_crs(&self, crs: i32, x: f64, y: f64) -> AnyResult<PointQueryResult> {
        let Some(pixel) = self.locate_pixel(crs, x, y)? else {
            return Ok(self.empty_point_result(crs));
        };
        let Some(tile_index) = self.metadata.tile_index_for_pixel(pixel.0, pixel.1) else {
            return Ok(self.empty_point_result(crs));
        };

        // One tile read serves every band
        let (tile, _) = self.read_tile_sync(TileRef { overview: None, index: tile_index })?;
        Ok(self.point_result(crs, pixel, &tile, tile_index))
    }

    fn sample_band_lonlat(&self, lon: f64, lat: f64, band: usize) -> AnyResult<Option<f32>> {
        self.sample_band_crs(4326, lon, lat, band)
    }

    fn sample_band_crs(&self, crs: i32, x: f64, y: f64, band: usize) -> AnyResult<Option<f32>> {
        if band >= self.metadata.bands {
            return Ok(None);
        }

        match self.locate_pixel(crs, x, y)? {
            Some((px, py)) => self.sample(band, px, py),
            None => Ok(None),
        }
    }

    fn sample_points_lonlat(&self, points: &[(f64, f64)]) -> AnyResult<Vec<PointQueryResult>> {
        self.sample_points_crs(4326, points)
    }

    fn sample_points_crs(&self, crs: i32, points: &[(f64, f64)]) -> AnyResult<Vec<PointQueryResult>> {
        let mut results = Vec::with_capacity(points.len());

        for &(x, y) in points {
            results.push(self.sample_crs(crs, x, y)?);
        }

        Ok(results)
    }
}

/// Pixel lookup shared by the synchronous [`PointQuery`] implementation and the `*_async`
/// methods (`sample_*_async` in `point_query_async.rs`): only the tile read differs.
impl CogReader {
    /// Source pixel `(x, y)` containing the coordinate `(x, y)` given in `crs`, or `None` if it
    /// lies outside the raster (or the raster has no usable geotransform).
    pub(crate) fn locate_pixel(&self, crs: i32, x: f64, y: f64) -> AnyResult<Option<(usize, usize)>> {
        let source_crs = self.metadata.crs_code.unwrap_or(4326);

        // Transform to source CRS if needed
        let (src_x, src_y) = if crs == source_crs {
            (x, y)
        } else {
            project_point(crs, source_crs, x, y)?
        };

        // Convert world coordinates to pixel coordinates
        let Some((px, py)) = self.metadata.geo_transform.world_to_pixel(src_x, src_y) else {
            return Ok(None);
        };

        // Bounds check
        // Allow cast precision loss: bounds checking only needs approximate precision
        #[allow(clippy::cast_precision_loss)]
        if px < 0.0 || py < 0.0 ||
           px >= self.metadata.width as f64 ||
           py >= self.metadata.height as f64 {
            return Ok(None);
        }

        // Cast is safe: already bounds-checked against width/height
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Ok(Some((px as usize, py as usize)))
    }

    /// Result for a coordinate outside the raster.
    pub(crate) fn empty_point_result(&self, crs: i32) -> PointQueryResult {
        PointQueryResult {
            values: HashMap::new(),
            bands: self.metadata.bands,
            is_valid: false,
            pixel_coords: None,
            input_crs: crs,
            raster_crs: self.metadata.crs_code.unwrap_or(4326),
        }
    }

    /// Result for `pixel`, with all bands read from its decoded tile.
    pub(crate) fn point_result(&self, crs: i32, pixel: (usize, usize), tile: &[f32], tile_index: usize) -> PointQueryResult {
        let values = (0..self.metadata.bands)
            .map(|band| (band, self.value_in_tile(tile, tile_index, band, pixel.0, pixel.1).unwrap_or(f32::NAN)))
            .collect();
        PointQueryResult {
            values,
            bands: self.metadata.bands,
            is_valid: true,
            pixel_coords: Some(pixel),
            input_crs: crs,
            raster_crs: self.metadata.crs_code.unwrap_or(4326),
        }
    }
}

/// Convenience function to sample all bands at a lon/lat coordinate.
///
/// This is a shorthand for `reader.sample_lonlat(lon, lat)?.values`.
///
/// # Example
///
/// ```rust,no_run
/// use cogrs::{CogReader, sample_point};
///
/// fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
///     let reader = CogReader::open("path/to/cog.tif")?;
///     let values = sample_point(&reader, -122.4, 37.8)?;
///     Ok(())
/// }
/// ```
///
/// # Errors
/// Returns an error if coordinate projection fails or if reading from the raster source fails.
pub fn sample_point(reader: &CogReader, lon: f64, lat: f64) -> AnyResult<HashMap<usize, f32>> {
    Ok(reader.sample_lonlat(lon, lat)?.values)
}

/// Convenience function to sample all bands at a coordinate in a specific CRS.
///
/// # Errors
/// Returns an error if coordinate projection fails or if reading from the raster source fails.
pub fn sample_point_crs(
    reader: &CogReader,
    crs: i32,
    x: f64,
    y: f64,
) -> AnyResult<HashMap<usize, f32>> {
    Ok(reader.sample_crs(crs, x, y)?.values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_query_result_to_vec() {
        let mut values = HashMap::new();
        values.insert(0, 1.0);
        values.insert(1, 2.0);
        values.insert(2, 3.0);

        let result = PointQueryResult {
            values,
            bands: 3,
            is_valid: true,
            pixel_coords: Some((10, 20)),
            input_crs: 4326,
            raster_crs: 4326,
        };

        let vec = result.to_vec();
        assert_eq!(vec, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_point_query_result_get() {
        let mut values = HashMap::new();
        values.insert(0, 42.0);
        values.insert(1, 100.0);

        let result = PointQueryResult {
            values,
            bands: 2,
            is_valid: true,
            pixel_coords: Some((5, 5)),
            input_crs: 4326,
            raster_crs: 3857,
        };

        assert_eq!(result.get(0), Some(42.0));
        assert_eq!(result.get(1), Some(100.0));
        assert_eq!(result.get(2), None);
    }

    #[test]
    fn test_point_query_result_has_valid_data() {
        let mut values = HashMap::new();
        values.insert(0, f32::NAN);
        values.insert(1, 5.0);

        let result = PointQueryResult {
            values,
            bands: 2,
            is_valid: true,
            pixel_coords: Some((0, 0)),
            input_crs: 4326,
            raster_crs: 4326,
        };

        assert!(result.has_valid_data());
        assert_eq!(result.valid_band_count(), 1);
    }

    #[test]
    fn test_point_query_result_all_nan() {
        let mut values = HashMap::new();
        values.insert(0, f32::NAN);
        values.insert(1, f32::NAN);

        let result = PointQueryResult {
            values,
            bands: 2,
            is_valid: true,
            pixel_coords: Some((0, 0)),
            input_crs: 4326,
            raster_crs: 4326,
        };

        assert!(!result.has_valid_data());
        assert_eq!(result.valid_band_count(), 0);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::sync::Arc;
    use crate::range_reader::LocalRangeReader;

    /// Test COG file path - Copernicus DEM 30m covering San Francisco Bay Area
    const TEST_COG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/copernicus_dem_san_francisco.tif");

    // Reference values verified with GDAL for precise testing
    mod reference {
        /// SF Downtown (lon=-122.4, lat=37.78)
        pub const SF_DOWNTOWN_LON: f64 = -122.4;
        pub const SF_DOWNTOWN_LAT: f64 = 37.78;
        pub const ELEV_SF_DOWNTOWN: f32 = 5.278_507;

        /// Mt Tamalpais area (lon=-122.5965, lat=37.9236)
        pub const MT_TAM_LON: f64 = -122.5965;
        pub const MT_TAM_LAT: f64 = 37.9236;
        pub const ELEV_MT_TAM: f32 = 782.097;

        /// Berkeley Hills (lon=-122.24, lat=37.88)
        pub const BERKELEY_HILLS_LON: f64 = -122.24;
        pub const BERKELEY_HILLS_LAT: f64 = 37.88;
        pub const ELEV_BERKELEY_HILLS: f32 = 367.597_17;

        /// SF Bay water (lon=-122.38, lat=37.79) - should be 0
        pub const SF_BAY_LON: f64 = -122.38;
        pub const SF_BAY_LAT: f64 = 37.79;
        pub const ELEV_SF_BAY: f32 = 0.0;
    }

    fn get_test_cog() -> Option<CogReader> {
        if !std::path::Path::new(TEST_COG_PATH).exists() {
            println!("Skipping: test file not found at {}", TEST_COG_PATH);
            return None;
        }
        let reader = LocalRangeReader::new(TEST_COG_PATH).ok()?;
        CogReader::from_reader(Arc::new(reader)).ok()
    }

    /// Test PointQuery returns precise values matching GDAL output
    #[test]
    fn test_point_query_precise_values() {
        let Some(cog) = get_test_cog() else { return };

        // SF Downtown - verified with gdallocationinfo
        let result = cog.sample_lonlat(reference::SF_DOWNTOWN_LON, reference::SF_DOWNTOWN_LAT).unwrap();
        assert!(result.is_valid, "SF Downtown should be valid");
        let elev = result.get(0).expect("Should have band 0");
        assert!((elev - reference::ELEV_SF_DOWNTOWN).abs() < 0.01,
            "SF Downtown elevation mismatch: got {}, expected {}", elev, reference::ELEV_SF_DOWNTOWN);

        // Mt Tamalpais - highest peak in SF area
        // Note: Higher tolerance due to steep terrain - adjacent 30m pixels can differ by >10m
        let result = cog.sample_lonlat(reference::MT_TAM_LON, reference::MT_TAM_LAT).unwrap();
        assert!(result.is_valid, "Mt Tam should be valid");
        let elev = result.get(0).expect("Should have band 0");
        assert!((elev - reference::ELEV_MT_TAM).abs() < 2.0,
            "Mt Tam elevation mismatch: got {}, expected {}", elev, reference::ELEV_MT_TAM);

        // Berkeley Hills
        let result = cog.sample_lonlat(reference::BERKELEY_HILLS_LON, reference::BERKELEY_HILLS_LAT).unwrap();
        assert!(result.is_valid, "Berkeley Hills should be valid");
        let elev = result.get(0).expect("Should have band 0");
        assert!((elev - reference::ELEV_BERKELEY_HILLS).abs() < 0.1,
            "Berkeley Hills elevation mismatch: got {}, expected {}", elev, reference::ELEV_BERKELEY_HILLS);

        // SF Bay water - should be 0
        let result = cog.sample_lonlat(reference::SF_BAY_LON, reference::SF_BAY_LAT).unwrap();
        assert!(result.is_valid, "SF Bay should be valid");
        let elev = result.get(0).expect("Should have band 0");
        assert!((elev - reference::ELEV_SF_BAY).abs() < 0.01,
            "SF Bay elevation mismatch: got {}, expected {}", elev, reference::ELEV_SF_BAY);
    }

    /// Test points outside DEM coverage are correctly identified as invalid
    #[test]
    fn test_point_query_outside_coverage() {
        let Some(cog) = get_test_cog() else { return };

        // Points outside the DEM coverage (N37-N38, W123-W122)
        let outside_points = [
            (0.0, 0.0),       // Equator
            (139.7, 35.7),    // Tokyo
            (-122.5, 36.9),   // Just south of coverage
            (-121.9, 37.5),   // Just east of coverage
        ];

        for (lon, lat) in outside_points {
            let result = cog.sample_lonlat(lon, lat).unwrap();
            assert!(!result.is_valid, "Point ({}, {}) should be outside coverage", lon, lat);
        }
    }

    /// Test batch point queries with precise elevation verification
    #[test]
    fn test_point_query_batch() {
        let Some(cog) = get_test_cog() else { return };

        let points = vec![
            (reference::SF_DOWNTOWN_LON, reference::SF_DOWNTOWN_LAT),
            (reference::MT_TAM_LON, reference::MT_TAM_LAT),
            (reference::BERKELEY_HILLS_LON, reference::BERKELEY_HILLS_LAT),
            (0.0, 0.0),       // Equator (outside)
        ];

        let expected = [
            (true, Some(reference::ELEV_SF_DOWNTOWN)),
            (true, Some(reference::ELEV_MT_TAM)),
            (true, Some(reference::ELEV_BERKELEY_HILLS)),
            (false, None),
        ];

        let results = cog.sample_points_lonlat(&points).unwrap();
        assert_eq!(results.len(), 4);

        for (i, (result, (exp_valid, exp_elev))) in results.iter().zip(expected.iter()).enumerate() {
            assert_eq!(result.is_valid, *exp_valid, "Point {} validity mismatch", i);
            if let Some(expected_elev) = exp_elev {
                let elev = result.get(0).expect("Should have band 0");
                assert!((elev - expected_elev).abs() < 0.1,
                    "Point {} elevation mismatch: got {}, expected {}", i, elev, expected_elev);
            }
        }
    }

    /// Test sample_point convenience function with precise values
    #[test]
    fn test_sample_point_convenience() {
        let Some(cog) = get_test_cog() else { return };

        let values = sample_point(&cog, reference::MT_TAM_LON, reference::MT_TAM_LAT).unwrap();

        assert!(values.contains_key(&0), "Should have band 0");
        let elevation = *values.get(&0).unwrap();
        assert!((elevation - reference::ELEV_MT_TAM).abs() < 0.1,
            "Mt Tam elevation mismatch: got {}, expected {}", elevation, reference::ELEV_MT_TAM);
    }
}

/// GDAL verification tests - validate point query results against GDAL
#[cfg(test)]
mod gdal_verification_tests {
    use super::*;
    use std::sync::Arc;
    use crate::range_reader::LocalRangeReader;

    const TEST_COG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/copernicus_dem_san_francisco.tif");

    fn get_test_cog() -> Option<CogReader> {
        if !std::path::Path::new(TEST_COG_PATH).exists() {
            return None;
        }
        let reader = LocalRangeReader::new(TEST_COG_PATH).ok()?;
        CogReader::from_reader(Arc::new(reader)).ok()
    }

    /// Verify point queries match GDAL exactly across a grid of coordinates
    #[test]
    fn test_gdal_point_query_grid() {
        let Some(cog) = get_test_cog() else {
            println!("Skipping: test file not found");
            return;
        };

        let gdal_ds = gdal::Dataset::open(TEST_COG_PATH).expect("GDAL failed to open");
        let band = gdal_ds.rasterband(1).expect("Failed to get band 1");
        let gdal_gt = gdal_ds.geo_transform().expect("Failed to get geotransform");

        // Test a grid of points within the DEM coverage
        let lon_range = [-122.9, -122.7, -122.5, -122.3, -122.1];
        let lat_range = [37.1, 37.3, 37.5, 37.7, 37.9];

        let mut mismatches = 0;
        let mut total = 0;

        for lon in lon_range {
            for lat in lat_range {
                total += 1;

                // Our value
                let our_result = cog.sample_lonlat(lon, lat);
                let our_value = our_result.ok()
                    .filter(|r| r.is_valid)
                    .and_then(|r| r.get(0));

                // GDAL value
                let gdal_px = ((lon - gdal_gt[0]) / gdal_gt[1]) as isize;
                let gdal_py = ((gdal_gt[3] - lat) / (-gdal_gt[5])) as isize;

                // Check if in bounds
                let (width, height) = gdal_ds.raster_size();
                if gdal_px < 0 || gdal_py < 0 || gdal_px >= width as isize || gdal_py >= height as isize {
                    // Both should report invalid/out of bounds
                    assert!(our_value.is_none() || our_value == Some(f32::NAN),
                        "Point ({}, {}) should be out of bounds", lon, lat);
                    continue;
                }

                let gdal_buf: gdal::raster::Buffer<f32> = band.read_as((gdal_px, gdal_py), (1, 1), (1, 1), None)
                    .expect("GDAL read failed");
                let gdal_value = gdal_buf.data()[0];

                let our_val = our_value.unwrap_or(f32::NAN);

                if (our_val - gdal_value).abs() > 0.001 {
                    println!("Mismatch at ({}, {}): ours={}, gdal={}", lon, lat, our_val, gdal_value);
                    mismatches += 1;
                }
            }
        }

        assert_eq!(mismatches, 0, "{}/{} points had mismatches with GDAL", mismatches, total);
        println!("Verified {}/{} points match GDAL exactly", total - mismatches, total);
    }

    /// Verify CRS transformation point queries match GDAL
    #[test]
    fn test_gdal_point_query_with_crs_transform() {
        let Some(cog) = get_test_cog() else {
            println!("Skipping: test file not found");
            return;
        };

        let gdal_ds = gdal::Dataset::open(TEST_COG_PATH).expect("GDAL failed to open");
        let band = gdal_ds.rasterband(1).expect("Failed to get band 1");
        let gdal_gt = gdal_ds.geo_transform().expect("Failed to get geotransform");

        // Test coordinates in UTM Zone 10N (EPSG:32610)
        // These correspond to known locations in the SF area
        let utm_coords = [
            (551000.0, 4180000.0), // Near SF downtown
            (570000.0, 4190000.0), // East Bay
            (540000.0, 4200000.0), // Marin
        ];

        for (utm_x, utm_y) in utm_coords {
            // Our value via CRS transform
            let our_result = cog.sample_crs(32610, utm_x, utm_y);

            // Transform UTM to WGS84 for GDAL comparison
            let transformer = crate::geometry::projection::project_point(32610, 4326, utm_x, utm_y);
            if let Ok((lon, lat)) = transformer {
                let gdal_px = ((lon - gdal_gt[0]) / gdal_gt[1]) as isize;
                let gdal_py = ((gdal_gt[3] - lat) / (-gdal_gt[5])) as isize;

                let (width, height) = gdal_ds.raster_size();
                if gdal_px >= 0 && gdal_py >= 0 && gdal_px < width as isize && gdal_py < height as isize {
                    let gdal_buf: gdal::raster::Buffer<f32> = band.read_as((gdal_px, gdal_py), (1, 1), (1, 1), None)
                        .expect("GDAL read failed");
                    let gdal_value = gdal_buf.data()[0];

                    let our_value = our_result.ok()
                        .filter(|r| r.is_valid)
                        .and_then(|r| r.get(0))
                        .unwrap_or(f32::NAN);

                    assert!((our_value - gdal_value).abs() < 0.01,
                        "UTM ({}, {}) -> lon/lat ({}, {}): ours={}, gdal={}",
                        utm_x, utm_y, lon, lat, our_value, gdal_value);
                }
            }
        }
        println!("CRS transform point queries verified against GDAL");
    }

    // ========== RGB COG Tests ==========

    const RGB_COG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/natural_earth_rgb.tif");

    fn get_rgb_cog() -> Option<CogReader> {
        if !std::path::Path::new(RGB_COG_PATH).exists() {
            return None;
        }
        let reader = LocalRangeReader::new(RGB_COG_PATH).ok()?;
        CogReader::from_reader(Arc::new(reader)).ok()
    }

    #[test]
    fn test_rgb_point_query_multiband() {
        let Some(cog) = get_rgb_cog() else {
            println!("Skipping: RGB test file not found");
            return;
        };

        // Test a point within bounds - use center of image
        let result = cog.sample_lonlat(0.0, 0.0).expect("Center query failed");
        assert!(result.is_valid, "Center point should be valid");
        assert_eq!(result.values.len(), cog.metadata.bands, "Should have all bands");

        // Verify all bands have valid values
        for band in 0..cog.metadata.bands {
            let value = result.get(band);
            assert!(value.is_some(), "Band {} should have a value", band);
            let v = value.unwrap();
            assert!(!v.is_nan(), "Band {} value should not be NaN", band);
        }
    }

    #[test]
    fn test_rgb_point_query_global_coverage() {
        let Some(cog) = get_rgb_cog() else {
            println!("Skipping: RGB test file not found");
            return;
        };

        // Test points at various global locations (assuming global coverage)
        let test_locations = [
            (0.0, 0.0),       // Origin
            (-120.0, 40.0),   // West
            (120.0, 40.0),    // East
            (0.0, 45.0),      // North
            (0.0, -45.0),     // South
        ];

        for (lon, lat) in test_locations {
            let result = cog.sample_lonlat(lon, lat);
            if let Ok(r) = result
                && r.is_valid
            {
                assert_eq!(r.values.len(), cog.metadata.bands,
                    "Point ({}, {}) should have {} bands", lon, lat, cog.metadata.bands);
            }
        }
    }

    #[test]
    fn test_rgb_point_query_gdal_comparison() {
        let Some(cog) = get_rgb_cog() else {
            println!("Skipping: RGB test file not found");
            return;
        };

        let gdal_ds = gdal::Dataset::open(RGB_COG_PATH).expect("GDAL failed to open RGB COG");
        let gdal_gt = gdal_ds.geo_transform().expect("Failed to get geotransform");
        let (width, height) = gdal_ds.raster_size();
        let num_bands = gdal_ds.raster_count();

        // Test a grid of points within bounds
        let test_points = [
            (0.0, 0.0),
            (-90.0, 30.0),
            (90.0, 30.0),
            (-90.0, -30.0),
            (90.0, -30.0),
        ];

        for (lon, lat) in test_points {
            // Our value
            let our_result = cog.sample_lonlat(lon, lat);

            // GDAL pixel coordinates
            let gdal_px = ((lon - gdal_gt[0]) / gdal_gt[1]) as isize;
            let gdal_py = ((gdal_gt[3] - lat) / (-gdal_gt[5])) as isize;

            if gdal_px >= 0 && gdal_py >= 0 && gdal_px < width as isize && gdal_py < height as isize {
                let our_result = our_result.expect("Query should succeed for in-bounds point");

                // Compare all bands
                for band_idx in 1..=num_bands {
                    let band = gdal_ds.rasterband(band_idx).expect("Failed to get band");
                    let gdal_buf: gdal::raster::Buffer<u8> = band.read_as((gdal_px, gdal_py), (1, 1), (1, 1), None)
                        .expect("GDAL read failed");
                    let gdal_value = gdal_buf.data()[0];

                    let our_value = our_result.get(band_idx - 1).unwrap_or(f32::NAN) as u8;

                    assert!((our_value as i16 - gdal_value as i16).abs() <= 1,
                        "({}, {}) band {}: ours={}, gdal={}", lon, lat, band_idx, our_value, gdal_value);
                }
            }
        }
        println!("RGB point queries verified against GDAL");
    }

    #[test]
    fn test_rgb_point_query_result_to_vec() {
        let Some(cog) = get_rgb_cog() else {
            println!("Skipping: RGB test file not found");
            return;
        };

        let result = cog.sample_lonlat(0.0, 0.0).expect("Query failed");
        let values = result.to_vec();

        assert_eq!(values.len(), cog.metadata.bands, "Should have all band values");
        // Verify it returns values in band order
        for (i, value) in values.iter().enumerate() {
            assert_eq!(*value, result.get(i).unwrap());
        }
    }
}
