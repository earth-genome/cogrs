# Test and bench fixtures

The `*.tif` fixtures are **not committed** (`*.tif` is gitignored). Recreate them
from the crate root with:

```
scripts/fetch-test-data.sh          # skips files that already exist
scripts/fetch-test-data.sh --force  # rebuild everything
```

Tests that cannot find a fixture print "Skipping" and return early (they pass
without checking anything), so run the script before relying on `cargo test`.

## Tools required

`curl`, `unzip`, GDAL command-line tools (`gdal_translate`), libtiff tools
(`tiffset`), and `python3` with `numpy` and the GDAL bindings (`osgeo`). The
script checks for each up front. Network access is needed for the two
downloads. (The `gdal` dev-dependency of the crate also needs libgdal.)

## Fixtures

| File | What it is | Used by |
|---|---|---|
| `tests/data/copernicus_dem_san_francisco.tif` (20 MB) | Real Copernicus GLO-30 tile `N37_00_W123_00`, unmodified: EPSG:4326, 3600x3600, Float32, PixelIsPoint, DEFLATE+predictor 3, 1024 blocks, overviews 1800/900/450. Downloaded from `s3://copernicus-dem-30m/` (anonymous, eu-central-1). | `src/cog_reader.rs` (`test_compare_with_tiff_crate`, `gdal_verification_tests`), `src/point_query.rs` (`integration_tests`, `gdal_verification_tests`), `src/xyz_tile.rs` (`global_cog_tests`) |
| `tests/data/natural_earth_rgb.tif` (1.7 MB) | Natural Earth 1 (50m, `NE1_50M_SR_W`) downsampled to 1440x720, 3-band Byte, EPSG:4326 global COG with overviews. Downloaded from naciscdn.org. | the `test_rgb_*` tests in `cog_reader.rs`, `point_query.rs`, `xyz_tile.rs` |
| `data/grayscale/gray_3857-cog.tif` (2.6 MB) | **Synthetic** (not the author's original). EPSG:3857 world extent, 20966x20966 Byte, 512 tiles, overviews 10483/5241/2620/1310 (floor-halved so overview 3 is exactly 1310 wide, scale 16). Deterministic 16x16 pattern, nearest-resampled; pixel (0,0)=209. | `src/cog_reader.rs`: `test_gray_3857_crs_detection`, `test_overview_scale_uses_floor_division`, `test_overview_pixel_values_match_gdal`, `test_scale_factor_coordinate_mapping`, `test_best_overview_selection` (paths are relative to the crate root) |
| `data/viridis/output_cog.tif` (0.2 MB) | **Synthetic** 1024x512 3-band Byte viridis gradient, EPSG:4326 global COG, 256 tiles, overviews. | `src/cog_reader.rs`: `test_real_cog_file` |

The benches use a synthetic COG generated at startup unless `COGRS_BENCH_COG`
is set; `data/grayscale/gray_3857-cog.tif` fits (world coverage, EPSG:3857).

## Attribution

**Copernicus DEM GLO-30** (`copernicus_dem_san_francisco.tif`). The file is the
unmodified product tile. Per Article 6(a) of the Copernicus WorldDEM-30
licence, when distributing it the notice is:

> © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided
> under COPERNICUS by the European Union and ESA; all rights reserved.

If the data is adapted or modified (Article 6(b)):

> produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus
> Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European
> Union and ESA; all rights reserved.

Article 6(c) additionally requires, in any licence or legal notice covering
distribution to the public: "The organisations in charge of the Copernicus
programme by law or by delegation do not incur any liability for any use of
the Copernicus WorldDEM-30". Do not imply endorsement by the Copernicus
programme (Article 6(d)).

Sources:

- Licence text (COP-DEM-GLO-30-F):
  <https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf>
- AWS Open Data registry entry (licence pointer, bucket, citation):
  <https://registry.opendata.aws/copernicus-dem>
- Licence landing page referenced by the registry:
  <https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM>

The AWS registry asks for the citation: "Copernicus Digital Elevation Model
(DEM) was accessed on `DATE` from https://registry.opendata.aws/copernicus-dem".

**Natural Earth** (`natural_earth_rgb.tif`) is public domain; credit
"Made with Natural Earth" is appreciated but not required
(<https://www.naturalearthdata.com/about/terms-of-use/>, not re-fetched here).
