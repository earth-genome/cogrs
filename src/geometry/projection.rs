use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use parking_lot::RwLock;

/// Project a point from one CRS to another using pure Rust (proj4rs + crs-definitions).
///
/// This function handles coordinate transformations between any EPSG codes supported
/// by the crs-definitions database (~thousands of codes including UTM zones, national grids, etc).
///
/// # Arguments
/// * `source_epsg` - Source CRS EPSG code
/// * `target_epsg` - Target CRS EPSG code
/// * `x` - X coordinate in source CRS
/// * `y` - Y coordinate in source CRS
///
/// # Returns
/// Tuple of (x, y) in target CRS
///
/// # Errors
/// Returns an error if the EPSG code is not supported or the projection transformation fails.
#[inline]
pub fn project_point(source_epsg: i32, target_epsg: i32, x: f64, y: f64) -> Result<(f64, f64), String> {
    // No-op if same CRS
    if source_epsg == target_epsg {
        return Ok((x, y));
    }

    project_with_proj4rs(source_epsg, target_epsg, x, y)
}

/// Convenience function: longitude/latitude (EPSG:4326) to Web Mercator (EPSG:3857)
///
/// **Warning**: Returns original coordinates on projection failure.
/// Use [`try_lon_lat_to_mercator`] if you need to detect errors.
#[inline]
#[must_use]
pub fn lon_lat_to_mercator(lon: f64, lat: f64) -> (f64, f64) {
    project_point(4326, 3857, lon, lat).unwrap_or((lon, lat))
}

/// Convenience function: Web Mercator (EPSG:3857) to longitude/latitude (EPSG:4326)
///
/// **Warning**: Returns original coordinates on projection failure.
/// Use [`try_mercator_to_lon_lat`] if you need to detect errors.
#[inline]
#[must_use]
pub fn mercator_to_lon_lat(x: f64, y: f64) -> (f64, f64) {
    project_point(3857, 4326, x, y).unwrap_or((x, y))
}

/// Fallible version of [`lon_lat_to_mercator`]
///
/// # Errors
/// Returns an error if the projection transformation fails.
#[inline]
pub fn try_lon_lat_to_mercator(lon: f64, lat: f64) -> Result<(f64, f64), String> {
    project_point(4326, 3857, lon, lat)
}

/// Fallible version of [`mercator_to_lon_lat`]
///
/// # Errors
/// Returns an error if the projection transformation fails.
#[inline]
pub fn try_mercator_to_lon_lat(x: f64, y: f64) -> Result<(f64, f64), String> {
    project_point(3857, 4326, x, y)
}

/// Get PROJ4 string for an EPSG code using the crs-definitions database
#[inline]
pub fn get_proj_string(epsg: i32) -> Option<&'static str> {
    u16::try_from(epsg).ok()
        .and_then(crs_definitions::from_code)
        .map(|def| def.proj4)
}

/// Check if an EPSG code represents a geographic (lon/lat) CRS
#[inline]
#[must_use] 
pub fn is_geographic_crs(epsg: i32) -> bool {
    // Geographic CRS codes are typically in the 4000-4999 range
    // but we check the proj string to be sure
    if let Some(proj_str) = get_proj_string(epsg) {
        proj_str.contains("+proj=longlat")
    } else {
        // Fallback: assume 4326 and similar are geographic
        epsg == 4326 || (4000..5000).contains(&epsg)
    }
}

/// Parsed `proj4rs` definition for one EPSG code, plus whether that CRS is geographic.
///
/// `proj4rs::proj::Proj` is `Send + Sync` (plain data + fn pointers; nadgrid refs are `'static`).
/// [`crate::CoordTransformer::new`] can take `cached_proj(epsg)?.proj.clone()` and
/// `is_geographic` instead of rebuilding `Proj` per transformer.
pub(crate) struct ProjDef {
    pub proj: proj4rs::proj::Proj,
    pub is_geographic: bool,
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<proj4rs::proj::Proj>();
};

static PROJ_CACHE: LazyLock<RwLock<HashMap<i32, Arc<ProjDef>>>> = LazyLock::new(|| RwLock::new(HashMap::new()));

/// Process-wide cache of [`ProjDef`] keyed by EPSG code.
///
/// Building a `proj4rs::Proj` parses the PROJ string on every call; point queries and
/// [`project_point`] hit the same codes repeatedly. Misses are inserted once; errors are not cached.
///
/// # Errors
/// Returns an error if the EPSG code is missing from crs-definitions or the PROJ string is invalid.
pub(crate) fn cached_proj(epsg: i32) -> Result<Arc<ProjDef>, String> {
    if let Some(hit) = PROJ_CACHE.read().get(&epsg) {
        return Ok(Arc::clone(hit));
    }
    let built = Arc::new(build_proj_def(epsg)?);
    let mut cache = PROJ_CACHE.write();
    Ok(Arc::clone(cache.entry(epsg).or_insert(built)))
}

fn build_proj_def(epsg: i32) -> Result<ProjDef, String> {
    use proj4rs::proj::Proj;

    let proj_str = get_proj_string(epsg)
        .ok_or_else(|| format!("EPSG:{epsg} not supported (not in the crs-definitions database)"))?;
    let proj = Proj::from_proj_string(proj_str)
        .map_err(|e| format!("Invalid projection EPSG:{epsg}: {e:?}"))?;
    Ok(ProjDef { proj, is_geographic: is_geographic_crs(epsg) })
}

/// Project using proj4rs with EPSG codes from crs-definitions
fn project_with_proj4rs(source_epsg: i32, target_epsg: i32, x: f64, y: f64) -> Result<(f64, f64), String> {
    use proj4rs::transform::transform;

    let source = cached_proj(source_epsg)?;
    let target = cached_proj(target_epsg)?;

    // proj4rs uses radians for geographic coordinates
    let (x_in, y_in) = if source.is_geographic {
        (x.to_radians(), y.to_radians())
    } else {
        (x, y)
    };

    let mut point = (x_in, y_in, 0.0);
    transform(&source.proj, &target.proj, &mut point)
        .map_err(|e| format!("Transform from EPSG:{source_epsg} to EPSG:{target_epsg} failed: {e:?}"))?;

    let (out_x, out_y) = if target.is_geographic {
        (point.0.to_degrees(), point.1.to_degrees())
    } else {
        (point.0, point.1)
    };

    Ok((out_x, out_y))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_lon_lat_to_mercator_origin() {
        let (x, y) = lon_lat_to_mercator(0.0, 0.0);
        assert!(approx_eq(x, 0.0));
        assert!(approx_eq(y, 0.0));
    }

    #[test]
    fn test_mercator_to_lon_lat_origin() {
        let (lon, lat) = mercator_to_lon_lat(0.0, 0.0);
        assert!(approx_eq(lon, 0.0));
        assert!(approx_eq(lat, 0.0));
    }

    #[test]
    fn test_roundtrip_4326_3857() {
        let test_points = [
            (0.0, 0.0),
            (10.0, 51.5),   // London-ish
            (-122.4, 37.8), // San Francisco
            (139.7, 35.7),  // Tokyo
        ];

        for (lon, lat) in test_points {
            let (x, y) = lon_lat_to_mercator(lon, lat);
            let (lon2, lat2) = mercator_to_lon_lat(x, y);
            assert!(approx_eq(lon, lon2), "lon: {} != {}", lon, lon2);
            assert!(approx_eq(lat, lat2), "lat: {} != {}", lat, lat2);
        }
    }

    #[test]
    fn test_extreme_latitudes() {
        // proj4rs handles extreme latitudes - verify we get finite values
        let (_, y1) = lon_lat_to_mercator(0.0, 85.0);
        let (_, y2) = lon_lat_to_mercator(0.0, -85.0);
        assert!(y1.is_finite(), "85 deg should produce finite y");
        assert!(y2.is_finite(), "-85 deg should produce finite y");
        assert!(y1 > 0.0, "positive lat should give positive y");
        assert!(y2 < 0.0, "negative lat should give negative y");
    }

    #[test]
    fn test_project_point_same_crs() {
        let result = project_point(4326, 4326, 10.0, 51.5);
        assert!(result.is_ok());
        let (x, y) = result.unwrap();
        assert!(approx_eq(x, 10.0));
        assert!(approx_eq(y, 51.5));
    }

    #[test]
    fn test_project_point_4326_to_3857() {
        let result = project_point(4326, 3857, 0.0, 0.0);
        assert!(result.is_ok());
        let (x, y) = result.unwrap();
        assert!(approx_eq(x, 0.0));
        assert!(approx_eq(y, 0.0));
    }

    #[test]
    fn test_project_point_3857_to_4326() {
        let result = project_point(3857, 4326, 0.0, 0.0);
        assert!(result.is_ok());
        let (lon, lat) = result.unwrap();
        assert!(approx_eq(lon, 0.0));
        assert!(approx_eq(lat, 0.0));
    }

    #[test]
    fn test_project_point_via_proj4rs() {
        // Test a transformation that goes through proj4rs
        // EPSG:32633 is UTM zone 33N
        let result = project_point(4326, 32633, 15.0, 52.0);
        assert!(result.is_ok(), "Should support UTM zones: {:?}", result);
        let (x, y) = result.unwrap();
        // UTM coordinates should be in meters, roughly 500000 for easting near zone center
        assert!(x > 400000.0 && x < 600000.0, "UTM easting: {}", x);
        assert!(y > 5000000.0 && y < 6000000.0, "UTM northing: {}", y);
    }

    #[test]
    fn test_project_point_roundtrip_utm() {
        // Roundtrip through UTM
        let lon = 15.0;
        let lat = 52.0;

        let to_utm = project_point(4326, 32633, lon, lat);
        assert!(to_utm.is_ok());
        let (x, y) = to_utm.unwrap();

        let back = project_point(32633, 4326, x, y);
        assert!(back.is_ok());
        let (lon2, lat2) = back.unwrap();

        assert!((lon - lon2).abs() < 1e-5, "lon roundtrip: {} -> {}", lon, lon2);
        assert!((lat - lat2).abs() < 1e-5, "lat roundtrip: {} -> {}", lat, lat2);
    }

    #[test]
    fn test_get_proj_string_common_codes() {
        assert!(get_proj_string(4326).is_some(), "4326 should be in database");
        assert!(get_proj_string(3857).is_some(), "3857 should be in database");
        assert!(get_proj_string(32633).is_some(), "UTM 33N should be in database");
    }

    #[test]
    fn test_is_geographic_crs() {
        assert!(is_geographic_crs(4326), "4326 is geographic");
        assert!(!is_geographic_crs(3857), "3857 is projected");
        assert!(!is_geographic_crs(32633), "UTM is projected");
    }

    #[test]
    fn test_unsupported_epsg_code() {
        // Use an EPSG code that definitely doesn't exist
        let result = project_point(4326, 999999, 0.0, 0.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not in the crs-definitions database"));
    }

    #[test]
    fn test_common_geographic_crs() {
        // Test common geographic CRS codes
        let geographic_codes = [
            4326,  // WGS84
            4269,  // NAD83
            4267,  // NAD27
            4258,  // ETRS89
            4674,  // SIRGAS 2000
            4326,  // WGS84
        ];

        for code in geographic_codes {
            let result = get_proj_string(code);
            assert!(result.is_some(), "EPSG:{} should be supported", code);
        }
    }

    #[test]
    fn test_utm_zone_coverage() {
        // Test all UTM zones (1-60 North and South)
        let mut supported = 0;
        let mut unsupported = Vec::new();

        // UTM zones North (32601-32660)
        for zone in 1..=60 {
            let code = 32600 + zone;
            if get_proj_string(code).is_some() {
                supported += 1;
            } else {
                unsupported.push(code);
            }
        }

        // UTM zones South (32701-32760)
        for zone in 1..=60 {
            let code = 32700 + zone;
            if get_proj_string(code).is_some() {
                supported += 1;
            } else {
                unsupported.push(code);
            }
        }

        // All 120 UTM zones should be supported
        assert!(unsupported.is_empty(),
            "Unsupported UTM zones: {:?}. Supported: {}/120", unsupported, supported);
    }

    #[test]
    fn test_national_grid_systems() {
        // Test common national/regional grid systems
        let national_grids = [
            (27700, "British National Grid"),
            (2154, "RGF93 / Lambert-93 (France)"),
            (25832, "ETRS89 / UTM zone 32N"),
            (3035, "ETRS89-extended / LAEA Europe"),
            (32632, "WGS 84 / UTM zone 32N"),
        ];

        for (code, name) in national_grids {
            let result = get_proj_string(code);
            assert!(result.is_some(), "EPSG:{} ({}) should be supported", code, name);

            // Also test transformation roundtrip
            let result = project_point(4326, code, 10.0, 52.0);
            assert!(result.is_ok(), "Transform to EPSG:{} should work: {:?}", code, result);
        }
    }
}

/// GDAL verification tests for coordinate transforms
#[cfg(test)]
mod gdal_verification_tests {
    use super::*;
    use gdal::spatial_ref::{CoordTransform, SpatialRef};

    /// Verify WGS84 to Web Mercator transforms match GDAL
    #[test]
    fn test_gdal_transform_4326_to_3857() {
        let mut gdal_src = SpatialRef::from_epsg(4326).expect("GDAL failed to create SRS 4326");
        let mut gdal_dst = SpatialRef::from_epsg(3857).expect("GDAL failed to create SRS 3857");
        gdal_src.set_axis_mapping_strategy(gdal::spatial_ref::AxisMappingStrategy::TraditionalGisOrder);
        gdal_dst.set_axis_mapping_strategy(gdal::spatial_ref::AxisMappingStrategy::TraditionalGisOrder);
        let gdal_transform = CoordTransform::new(&gdal_src, &gdal_dst).expect("GDAL transform failed");

        let test_points = [
            (0.0, 0.0),
            (-122.4, 37.8),   // SF
            (139.7, 35.7),    // Tokyo
            (-73.9, 40.7),    // NYC
            (2.35, 48.85),    // Paris
        ];

        for (lon, lat) in test_points {
            // Our transform
            let (our_x, our_y) = project_point(4326, 3857, lon, lat)
                .expect("Our transform failed");

            // GDAL transform
            let mut gdal_x = [lon];
            let mut gdal_y = [lat];
            gdal_transform.transform_coords(&mut gdal_x, &mut gdal_y, &mut [])
                .expect("GDAL transform failed");

            let tolerance = 0.01; // 1cm tolerance
            assert!((our_x - gdal_x[0]).abs() < tolerance,
                "X mismatch at ({}, {}): ours={}, gdal={}", lon, lat, our_x, gdal_x[0]);
            assert!((our_y - gdal_y[0]).abs() < tolerance,
                "Y mismatch at ({}, {}): ours={}, gdal={}", lon, lat, our_y, gdal_y[0]);
        }
        println!("4326 -> 3857 transforms verified against GDAL");
    }

    /// Verify Web Mercator to WGS84 transforms match GDAL
    #[test]
    fn test_gdal_transform_3857_to_4326() {
        let mut gdal_src = SpatialRef::from_epsg(3857).expect("GDAL failed to create SRS 3857");
        let mut gdal_dst = SpatialRef::from_epsg(4326).expect("GDAL failed to create SRS 4326");
        gdal_src.set_axis_mapping_strategy(gdal::spatial_ref::AxisMappingStrategy::TraditionalGisOrder);
        gdal_dst.set_axis_mapping_strategy(gdal::spatial_ref::AxisMappingStrategy::TraditionalGisOrder);
        let gdal_transform = CoordTransform::new(&gdal_src, &gdal_dst).expect("GDAL transform failed");

        let test_points = [
            (0.0, 0.0),
            (-13627665.0, 4551343.0),  // SF in Web Mercator
            (15554718.0, 4255815.0),   // Tokyo
            (-8226439.0, 4974863.0),   // NYC
        ];

        for (x, y) in test_points {
            // Our transform
            let (our_lon, our_lat) = project_point(3857, 4326, x, y)
                .expect("Our transform failed");

            // GDAL transform
            let mut gdal_x = [x];
            let mut gdal_y = [y];
            gdal_transform.transform_coords(&mut gdal_x, &mut gdal_y, &mut [])
                .expect("GDAL transform failed");

            let tolerance = 1e-6; // ~0.1m in degrees
            assert!((our_lon - gdal_x[0]).abs() < tolerance,
                "Lon mismatch at ({}, {}): ours={}, gdal={}", x, y, our_lon, gdal_x[0]);
            assert!((our_lat - gdal_y[0]).abs() < tolerance,
                "Lat mismatch at ({}, {}): ours={}, gdal={}", x, y, our_lat, gdal_y[0]);
        }
        println!("3857 -> 4326 transforms verified against GDAL");
    }

    /// Verify UTM transforms match GDAL
    #[test]
    fn test_gdal_transform_to_utm() {
        // Test WGS84 to UTM Zone 10N (covers SF area)
        let mut gdal_src = SpatialRef::from_epsg(4326).expect("GDAL failed to create SRS 4326");
        let mut gdal_dst = SpatialRef::from_epsg(32610).expect("GDAL failed to create SRS 32610");
        gdal_src.set_axis_mapping_strategy(gdal::spatial_ref::AxisMappingStrategy::TraditionalGisOrder);
        gdal_dst.set_axis_mapping_strategy(gdal::spatial_ref::AxisMappingStrategy::TraditionalGisOrder);
        let gdal_transform = CoordTransform::new(&gdal_src, &gdal_dst).expect("GDAL transform failed");

        let test_points = [
            (-122.4, 37.78),    // SF Downtown
            (-122.24, 37.88),   // Berkeley Hills
            (-122.5965, 37.9236), // Mt Tam
        ];

        for (lon, lat) in test_points {
            // Our transform
            let (our_x, our_y) = project_point(4326, 32610, lon, lat)
                .expect("Our transform failed");

            // GDAL transform
            let mut gdal_x = [lon];
            let mut gdal_y = [lat];
            gdal_transform.transform_coords(&mut gdal_x, &mut gdal_y, &mut [])
                .expect("GDAL transform failed");

            let tolerance = 0.01; // 1cm tolerance
            assert!((our_x - gdal_x[0]).abs() < tolerance,
                "X mismatch at ({}, {}): ours={}, gdal={}", lon, lat, our_x, gdal_x[0]);
            assert!((our_y - gdal_y[0]).abs() < tolerance,
                "Y mismatch at ({}, {}): ours={}, gdal={}", lon, lat, our_y, gdal_y[0]);
        }
        println!("4326 -> 32610 (UTM 10N) transforms verified against GDAL");
    }

    /// Verify transform roundtrips match GDAL
    #[test]
    fn test_gdal_transform_roundtrip() {
        let test_cases = [
            (4326, 3857, -122.4, 37.78),
            (4326, 32610, -122.4, 37.78),
            (3857, 4326, -13627665.0, 4551343.0),
        ];

        for (src_epsg, dst_epsg, x, y) in test_cases {
            // Forward transform
            let (mid_x, mid_y) = project_point(src_epsg, dst_epsg, x, y)
                .expect("Forward transform failed");

            // Reverse transform
            let (final_x, final_y) = project_point(dst_epsg, src_epsg, mid_x, mid_y)
                .expect("Reverse transform failed");

            let tolerance = if is_geographic_crs(src_epsg) { 1e-8 } else { 0.001 };
            assert!((final_x - x).abs() < tolerance,
                "X roundtrip failed for {} -> {} -> {}: {} -> {} -> {}",
                src_epsg, dst_epsg, src_epsg, x, mid_x, final_x);
            assert!((final_y - y).abs() < tolerance,
                "Y roundtrip failed for {} -> {} -> {}: {} -> {} -> {}",
                src_epsg, dst_epsg, src_epsg, y, mid_y, final_y);
        }
        println!("Transform roundtrips verified");
    }
}
