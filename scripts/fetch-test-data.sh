#!/usr/bin/env bash
# Recreate the test/bench fixtures that are deliberately not committed
# (*.tif is gitignored). Run from the crate root:
#
#   scripts/fetch-test-data.sh [--force]
#
# Fixtures (see tests/data/README.md):
#   tests/data/copernicus_dem_san_francisco.tif   real Copernicus GLO-30 tile (download)
#   tests/data/natural_earth_rgb.tif              Natural Earth 1 50m, downsampled COG (download)
#   data/grayscale/gray_3857-cog.tif              synthetic 20966x20966 EPSG:3857 COG
#   data/viridis/output_cog.tif                   synthetic 1024x512 RGB EPSG:4326 COG
#
# Existing files are skipped unless --force is given.
set -euo pipefail

FORCE=0
case "${1:-}" in
  "") ;;
  --force) FORCE=1 ;;
  *) echo "usage: $0 [--force]" >&2; exit 2 ;;
esac

if [[ ! -f Cargo.toml || ! -d src ]]; then
  echo "error: run this script from the crate root (the directory containing Cargo.toml)" >&2
  exit 1
fi

# ---- tool checks -----------------------------------------------------------
missing=0
need() { # need <command> <hint>
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: '$1' not found - $2" >&2
    missing=1
  fi
}
need curl           "install curl"
need unzip          "install unzip (needed to unpack the Natural Earth download)"
need gdal_translate "install GDAL command-line tools (e.g. gdal-bin)"
need tiffset        "install libtiff tools (e.g. libtiff-tools); used to flag overview IFDs"
need python3        "install Python 3"
if command -v python3 >/dev/null 2>&1; then
  python3 -c 'import numpy' 2>/dev/null \
    || { echo "error: python3 module 'numpy' not found - pip install numpy / apt install python3-numpy" >&2; missing=1; }
  python3 -c 'from osgeo import gdal' 2>/dev/null \
    || { echo "error: python3 module 'osgeo' not found - install GDAL Python bindings (e.g. python3-gdal)" >&2; missing=1; }
fi
(( missing == 0 )) || exit 1

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

want() { # want <path>: true if the file must be (re)built
  if [[ -e "$1" && $FORCE -eq 0 ]]; then
    echo "skip   $1 (exists; use --force to rebuild)"
    return 1
  fi
  echo "build  $1"
  mkdir -p "$(dirname "$1")"
  return 0
}

# ---- 1. Copernicus DEM GLO-30, tile N37 W123 (byte-identical copy) ----------
DEM=tests/data/copernicus_dem_san_francisco.tif
if want "$DEM"; then
  T=Copernicus_DSM_COG_10_N37_00_W123_00_DEM
  curl -fsSL -o "$WORK/dem.tif" \
    "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/$T/$T.tif"
  mv "$WORK/dem.tif" "$DEM"
fi

# ---- 2. Natural Earth 1 (50m, shaded relief + water) -> 1440x720 RGB COG ----
NE=tests/data/natural_earth_rgb.tif
if want "$NE"; then
  curl -fsSL -o "$WORK/ne.zip" https://naciscdn.org/naturalearth/50m/raster/NE1_50M_SR_W.zip
  unzip -q -o "$WORK/ne.zip" -d "$WORK/ne"
  gdal_translate -q -a_srs EPSG:4326 -a_ullr -180 90 180 -90 -outsize 1440 720 -r average \
    -of COG -co COMPRESS=DEFLATE -co PREDICTOR=2 -co LEVEL=9 -co BLOCKSIZE=256 \
    -co RESAMPLING=AVERAGE "$WORK/ne/NE1_50M_SR_W/NE1_50M_SR_W.tif" "$NE"
fi

# ---- 3. Synthetic gray EPSG:3857 COG, 20966x20966, overviews 10483/5241/2620/1310 ----
# The tests need overview 3 to be exactly 1310 wide (scale 16 by floor division).
# GDAL builds overviews with ceil() (1311), so each level is written separately
# into one multi-IFD TIFF, then IFDs 1..4 are flagged as reduced-resolution
# (NewSubfileType=1) with tiffset and the file is rewritten as a COG.
GRAY=data/grayscale/gray_3857-cog.tif
if want "$GRAY"; then
  python3 - "$WORK/gray_256.tif" <<'PY'
import sys
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
n = 256
y, x = np.mgrid[0:n, 0:n] / n
f = 176 + 40 * np.sin(6 * x + 1) * np.cos(5 * y) - (x + y) * 60 + 30 * np.sin(14 * (x - y)) * 0.5
f = np.clip(f, 20, 250)
f[0, 0] = 176
arr = np.rint(f).astype("uint8")
d = gdal.GetDriverByName("GTiff").Create(sys.argv[1], n, n, 1, gdal.GDT_Byte)
d.GetRasterBand(1).WriteArray(arr)
d = None
PY
  gdal_translate -q -outsize 16 16 -r average "$WORK/gray_256.tif" "$WORK/gray_16.tif"

  W=20037508.342789244
  first=1
  for s in 20966 10483 5241 2620 1310; do
    extra=()
    (( first )) || extra=(-co APPEND_SUBDATASET=YES)
    gdal_translate -q -a_srs EPSG:3857 -a_ullr -$W $W $W -$W -outsize "$s" "$s" -r nearest \
      -co TILED=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 \
      -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 \
      ${extra[@]+"${extra[@]}"} "$WORK/gray_16.tif" "$WORK/gray_multi.tif"
    first=0
  done
  for d in 1 2 3 4; do
    # tiffset warns about unknown GeoTIFF tags; those warnings are harmless.
    tiffset -d "$d" -s 254 1 "$WORK/gray_multi.tif" 2>&1 | { grep -v 'Unknown field' || true; }
  done
  gdal_translate -q -of COG -co BLOCKSIZE=512 -co COMPRESS=DEFLATE -co PREDICTOR=2 -co LEVEL=9 \
    -co OVERVIEWS=FORCE_USE_EXISTING "$WORK/gray_multi.tif" "$GRAY"
fi

# ---- 4. Synthetic viridis RGB COG, 1024x512, EPSG:4326 global ---------------
VIR=data/viridis/output_cog.tif
if want "$VIR"; then
  python3 - "$WORK/viridis_src.tif" <<'PY'
import sys
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
w, h = 1024, 512
x = np.linspace(0, 1, w)[None, :] * np.ones((h, 1))
y = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
t = np.clip(0.5 * x + 0.5 * y + 0.08 * np.sin(20 * x) * np.cos(12 * y), 0, 1)
stops = np.array([(0, 68, 1, 84), (0.25, 59, 82, 139), (0.5, 33, 145, 140),
                  (0.75, 94, 201, 98), (1, 253, 231, 37)], float)
rgb = [np.interp(t, stops[:, 0], stops[:, i + 1]).round().astype("uint8") for i in range(3)]
d = gdal.GetDriverByName("GTiff").Create(sys.argv[1], w, h, 3, gdal.GDT_Byte)
for i in range(3):
    d.GetRasterBand(i + 1).WriteArray(rgb[i])
d = None
PY
  gdal_translate -q -a_srs EPSG:4326 -a_ullr -180 90 180 -90 -of COG \
    -co BLOCKSIZE=256 -co COMPRESS=DEFLATE -co PREDICTOR=2 "$WORK/viridis_src.tif" "$VIR"
fi

echo "done"
