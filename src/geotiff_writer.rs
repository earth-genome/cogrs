//! GeoTIFF writer for reprojected rasters
//!
//! This module provides functionality to write [`ReprojectedRaster`] data to GeoTIFF files.
//! It uses pure Rust libraries (no GDAL dependency) and writes proper GeoTIFF metadata
//! including coordinate reference system information.
//!
//! # Example
//!
//! ```rust,no_run
//! use cogrs::{CogReader, Reprojector};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//!     let reader = CogReader::open("input.tif")?;
//!
//!     // Reproject to UTM using Reprojector builder
//!     let raster = Reprojector::new(&reader)
//!         .to_crs(32610)
//!         .resolution(10.0, 10.0)
//!         .extract()
//!         .await?;
//!
//!     // Write to GeoTIFF
//!     raster.write_geotiff("output.tif")?;
//!
//!     Ok(())
//! }
//! ```

use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use tiff::encoder::colortype::{Gray32Float, RGB32Float, RGBA32Float};
use tiff::encoder::{Compression, TiffEncoder};
use tiff::tags::Tag;

use crate::geometry::projection::get_proj_string;
use crate::xyz_tile::ReprojectedRaster;

// GeoTIFF Tag IDs (not in standard tiff crate)
const GEOTIFF_MODELPIXELSCALE: u16 = 33550;
const GEOTIFF_MODELTIEPOINT: u16 = 33922;
const GEOTIFF_GEOKEYDIRECTORY: u16 = 34735;
#[allow(dead_code)]
const GEOTIFF_GEODOUBLEPARAMS: u16 = 34736;
const GEOTIFF_GEOASCIIPARAMS: u16 = 34737;

// GeoKey IDs
const GT_MODEL_TYPE_GEO_KEY: u16 = 1024;
const GT_RASTER_TYPE_GEO_KEY: u16 = 1025;
const GEOGRAPHIC_TYPE_GEO_KEY: u16 = 2048;
const PROJECTED_CS_TYPE_GEO_KEY: u16 = 3072;

// GeoKey values
const MODEL_TYPE_PROJECTED: u16 = 1;
const MODEL_TYPE_GEOGRAPHIC: u16 = 2;
const RASTER_PIXEL_IS_AREA: u16 = 1;

/// Compression method for GeoTIFF output
#[derive(Debug, Clone, Copy, Default)]
pub enum GeoTiffCompression {
    /// No compression - fastest but largest files
    #[default]
    None,
    /// LZW compression - good balance of speed and size
    Lzw,
    /// Deflate (zlib) compression - better compression, slower
    Deflate,
}

/// Error type for GeoTIFF writing operations
#[derive(Debug)]
pub enum GeoTiffWriteError {
    /// I/O error during file operations
    Io(std::io::Error),
    /// TIFF encoding error
    TiffEncode(String),
    /// Invalid raster data
    InvalidData(String),
}

impl std::fmt::Display for GeoTiffWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::TiffEncode(e) => write!(f, "TIFF encoding error: {e}"),
            Self::InvalidData(e) => write!(f, "Invalid data: {e}"),
        }
    }
}

impl std::error::Error for GeoTiffWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GeoTiffWriteError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<tiff::TiffError> for GeoTiffWriteError {
    fn from(e: tiff::TiffError) -> Self {
        Self::TiffEncode(e.to_string())
    }
}

/// Builder for configuring GeoTIFF output
pub struct GeoTiffWriter<'a> {
    raster: &'a ReprojectedRaster,
    compression: GeoTiffCompression,
}

impl<'a> GeoTiffWriter<'a> {
    /// Create a new GeoTIFF writer for a reprojected raster
    #[must_use]
    pub fn new(raster: &'a ReprojectedRaster) -> Self {
        Self {
            raster,
            compression: GeoTiffCompression::default(),
        }
    }

    /// Set the compression method
    #[must_use]
    pub fn compression(mut self, compression: GeoTiffCompression) -> Self {
        self.compression = compression;
        self
    }

    /// Write to a file path
    pub fn write<P: AsRef<Path>>(self, path: P) -> Result<(), GeoTiffWriteError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.write_to(writer)
    }

    /// Write to any writer that implements Write + Seek
    pub fn write_to<W: Write + Seek>(self, writer: W) -> Result<(), GeoTiffWriteError> {
        let raster = self.raster;

        if raster.pixels.is_empty() {
            return Err(GeoTiffWriteError::InvalidData(
                "Raster has no pixel data".to_string(),
            ));
        }

        if raster.width == 0 || raster.height == 0 {
            return Err(GeoTiffWriteError::InvalidData(
                "Raster has zero dimensions".to_string(),
            ));
        }

        let width = raster.width as u32;
        let height = raster.height as u32;

        // Convert our compression enum to tiff crate's enum
        let compression = match self.compression {
            GeoTiffCompression::None => Compression::Uncompressed,
            GeoTiffCompression::Lzw => Compression::Lzw,
            GeoTiffCompression::Deflate => Compression::Deflate(tiff::encoder::DeflateLevel::Fast),
        };

        let encoder = TiffEncoder::new(writer)?.with_compression(compression);
        self.write_image(encoder, width, height)
    }

    fn write_image<W: Write + Seek>(
        &self,
        mut encoder: TiffEncoder<W>,
        width: u32,
        height: u32,
    ) -> Result<(), GeoTiffWriteError> {
        let bands = self.raster.bands;

        // For common band counts (1, 3, 4), use the high-level API with ColorType
        // For arbitrary band counts, use the low-level DirectoryEncoder API
        match bands {
            1 => {
                let mut image = encoder.new_image::<Gray32Float>(width, height)?;
                self.write_geotiff_tags(image.encoder())?;
                image.write_data(&self.raster.pixels)?;
            }
            3 => {
                let mut image = encoder.new_image::<RGB32Float>(width, height)?;
                self.write_geotiff_tags(image.encoder())?;
                image.write_data(&self.raster.pixels)?;
            }
            4 => {
                let mut image = encoder.new_image::<RGBA32Float>(width, height)?;
                self.write_geotiff_tags(image.encoder())?;
                image.write_data(&self.raster.pixels)?;
            }
            _ => {
                // Use low-level API for arbitrary band counts
                self.write_multiband_image(encoder, width, height)?;
            }
        }
        Ok(())
    }

    /// Write a multi-band image using the low-level DirectoryEncoder API.
    /// This supports arbitrary band counts (2, 5, 100+, etc.)
    fn write_multiband_image<W: Write + Seek>(
        &self,
        mut encoder: TiffEncoder<W>,
        width: u32,
        height: u32,
    ) -> Result<(), GeoTiffWriteError> {
        let bands = self.raster.bands;

        // Get directory encoder for low-level control
        let mut dir = encoder.image_directory()?;

        // Required TIFF tags for image
        dir.write_tag(Tag::ImageWidth, width)?;
        dir.write_tag(Tag::ImageLength, height)?;

        // BitsPerSample: 32 bits for each band (f32)
        let bits_per_sample: Vec<u16> = vec![32; bands];
        dir.write_tag(Tag::BitsPerSample, bits_per_sample.as_slice())?;

        // Compression
        let compression_tag: u16 = match self.compression {
            GeoTiffCompression::None => 1,    // No compression
            GeoTiffCompression::Lzw => 5,     // LZW
            GeoTiffCompression::Deflate => 8, // Deflate
        };
        dir.write_tag(Tag::Compression, compression_tag)?;

        // PhotometricInterpretation: 1 = BlackIsZero (grayscale-like for multi-band)
        dir.write_tag(Tag::PhotometricInterpretation, 1u16)?;

        // SamplesPerPixel
        dir.write_tag(Tag::SamplesPerPixel, bands as u16)?;

        // SampleFormat: 3 = IEEE floating point for each band
        let sample_format: Vec<u16> = vec![3; bands];
        dir.write_tag(Tag::SampleFormat, sample_format.as_slice())?;

        // PlanarConfiguration: 1 = Chunky (interleaved RGBRGB...)
        dir.write_tag(Tag::PlanarConfiguration, 1u16)?;

        // RowsPerStrip: write all rows in one strip for simplicity
        dir.write_tag(Tag::RowsPerStrip, height)?;

        // ExtraSamples: mark all bands beyond the first as unspecified (0)
        if bands > 1 {
            let extra_samples: Vec<u16> = vec![0; bands - 1];
            dir.write_tag(Tag::ExtraSamples, extra_samples.as_slice())?;
        }

        // Write GeoTIFF tags
        self.write_geotiff_tags(&mut dir)?;

        // Convert f32 pixels to bytes and write strip data
        let pixel_bytes: Vec<u8> = self
            .raster
            .pixels
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();

        // Write the strip offset and byte count
        let strip_offset = dir.write_data(pixel_bytes.as_slice())?;
        dir.write_tag(Tag::StripOffsets, strip_offset)?;

        let strip_byte_count = pixel_bytes.len() as u32;
        dir.write_tag(Tag::StripByteCounts, strip_byte_count)?;

        // Finish the directory
        dir.finish()?;

        Ok(())
    }

    fn write_geotiff_tags<W: Write + Seek, K: tiff::encoder::TiffKind>(
        &self,
        dir: &mut tiff::encoder::DirectoryEncoder<W, K>,
    ) -> Result<(), GeoTiffWriteError> {
        let raster = self.raster;

        // ModelPixelScale: [ScaleX, ScaleY, ScaleZ]
        let pixel_scale = [raster.resolution.0, raster.resolution.1, 0.0];
        dir.write_tag(Tag::Unknown(GEOTIFF_MODELPIXELSCALE), pixel_scale.as_slice())?;

        // ModelTiepoint: [I, J, K, X, Y, Z]
        // Ties pixel (0, 0) to world coordinate (minx, maxy)
        let tiepoint = [
            0.0,
            0.0,
            0.0,
            raster.bounds.minx,
            raster.bounds.maxy,
            0.0,
        ];
        dir.write_tag(Tag::Unknown(GEOTIFF_MODELTIEPOINT), tiepoint.as_slice())?;

        // Build GeoKeyDirectory
        let geokeys = self.build_geokey_directory(raster);
        dir.write_tag(Tag::Unknown(GEOTIFF_GEOKEYDIRECTORY), geokeys.as_slice())?;

        // Write PROJ string as GeoAsciiParams if we have one
        if let Some(proj_string) = get_proj_string(raster.crs as i32) {
            // GeoAsciiParams needs to be null-terminated with pipe delimiters
            let ascii_params = format!("{proj_string}|");
            dir.write_tag(Tag::Unknown(GEOTIFF_GEOASCIIPARAMS), ascii_params.as_bytes())?;
        }

        Ok(())
    }

    fn build_geokey_directory(&self, raster: &ReprojectedRaster) -> Vec<u16> {
        // GeoKeyDirectory structure:
        // [KeyDirectoryVersion, KeyRevision, MinorRevision, NumberOfKeys,
        //  KeyID1, TIFFTagLocation1, Count1, Value_Offset1, ...]

        let is_geographic = crate::geometry::projection::is_geographic_crs(raster.crs as i32);

        let mut keys = vec![
            1, // KeyDirectoryVersion
            1, // KeyRevision
            0, // MinorRevision
            3, // NumberOfKeys
        ];

        // GTModelTypeGeoKey
        keys.extend_from_slice(&[
            GT_MODEL_TYPE_GEO_KEY,
            0, // TIFFTagLocation = 0 means value is in Value_Offset
            1, // Count
            if is_geographic {
                MODEL_TYPE_GEOGRAPHIC
            } else {
                MODEL_TYPE_PROJECTED
            },
        ]);

        // GTRasterTypeGeoKey
        keys.extend_from_slice(&[GT_RASTER_TYPE_GEO_KEY, 0, 1, RASTER_PIXEL_IS_AREA]);

        // Geographic or Projected CRS type
        if is_geographic {
            keys.extend_from_slice(&[GEOGRAPHIC_TYPE_GEO_KEY, 0, 1, raster.crs as u16]);
        } else {
            keys.extend_from_slice(&[PROJECTED_CS_TYPE_GEO_KEY, 0, 1, raster.crs as u16]);
        }

        keys
    }
}

impl ReprojectedRaster {
    /// Write this raster to a GeoTIFF file.
    ///
    /// This writes a valid GeoTIFF with proper georeferencing metadata.
    /// The output will use the same CRS as the raster's `crs` field.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cogrs::{CogReader, Reprojector};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    ///     let reader = CogReader::open("input.tif")?;
    ///     let raster = Reprojector::new(&reader)
    ///         .to_crs(32610)
    ///         .resolution(10.0, 10.0)
    ///         .extract()
    ///         .await?;
    ///
    ///     raster.write_geotiff("output.tif")?;
    ///     Ok(())
    /// }
    /// ```
    pub fn write_geotiff<P: AsRef<Path>>(&self, path: P) -> Result<(), GeoTiffWriteError> {
        GeoTiffWriter::new(self).write(path)
    }

    /// Write this raster to a GeoTIFF file with compression.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cogrs::{CogReader, Reprojector, geotiff_writer::GeoTiffCompression};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    ///     let reader = CogReader::open("input.tif")?;
    ///     let raster = Reprojector::new(&reader)
    ///         .to_crs(32610)
    ///         .extract()
    ///         .await?;
    ///
    ///     raster.write_geotiff_compressed("output.tif", GeoTiffCompression::Lzw)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn write_geotiff_compressed<P: AsRef<Path>>(
        &self,
        path: P,
        compression: GeoTiffCompression,
    ) -> Result<(), GeoTiffWriteError> {
        GeoTiffWriter::new(self).compression(compression).write(path)
    }

    /// Get a GeoTIFF writer builder for more control over output options.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use cogrs::{CogReader, Reprojector, geotiff_writer::GeoTiffCompression};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    ///     let reader = CogReader::open("input.tif")?;
    ///     let raster = Reprojector::new(&reader)
    ///         .to_crs(32610)
    ///         .extract()
    ///         .await?;
    ///
    ///     raster.geotiff_writer()
    ///         .compression(GeoTiffCompression::Deflate)
    ///         .write("output.tif")?;
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    pub fn geotiff_writer(&self) -> GeoTiffWriter<'_> {
        GeoTiffWriter::new(self)
    }

    /// Write this raster to bytes as a GeoTIFF.
    ///
    /// Useful for serving GeoTIFFs over HTTP or storing in memory.
    pub fn to_geotiff_bytes(&self) -> Result<Vec<u8>, GeoTiffWriteError> {
        let mut buffer = std::io::Cursor::new(Vec::new());
        GeoTiffWriter::new(self).write_to(&mut buffer)?;
        Ok(buffer.into_inner())
    }

    /// Write this raster to bytes as a compressed GeoTIFF.
    pub fn to_geotiff_bytes_compressed(
        &self,
        compression: GeoTiffCompression,
    ) -> Result<Vec<u8>, GeoTiffWriteError> {
        let mut buffer = std::io::Cursor::new(Vec::new());
        GeoTiffWriter::new(self)
            .compression(compression)
            .write_to(&mut buffer)?;
        Ok(buffer.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xyz_tile::BoundingBox;

    fn create_test_raster(bands: usize, width: usize, height: usize) -> ReprojectedRaster {
        let pixels: Vec<f32> = (0..width * height * bands)
            .map(|i| (i % 256) as f32)
            .collect();

        ReprojectedRaster {
            pixels,
            bands,
            width,
            height,
            crs: 32610, // UTM 10N
            bounds: BoundingBox::new(500000.0, 4000000.0, 510000.0, 4010000.0),
            resolution: (10.0, 10.0),
            nodata: None,
        }
    }

    #[test]
    fn test_write_grayscale_geotiff() {
        let raster = create_test_raster(1, 64, 64);
        let bytes = raster.to_geotiff_bytes().unwrap();

        // Check TIFF magic bytes
        assert!(bytes.len() > 8);
        assert!(bytes[0] == b'I' && bytes[1] == b'I' || bytes[0] == b'M' && bytes[1] == b'M');

        println!("Grayscale GeoTIFF: {} bytes", bytes.len());
    }

    #[test]
    fn test_write_rgb_geotiff() {
        let raster = create_test_raster(3, 64, 64);
        let bytes = raster.to_geotiff_bytes().unwrap();

        assert!(bytes.len() > 8);
        println!("RGB GeoTIFF: {} bytes", bytes.len());
    }

    #[test]
    fn test_write_rgba_geotiff() {
        let raster = create_test_raster(4, 64, 64);
        let bytes = raster.to_geotiff_bytes().unwrap();

        assert!(bytes.len() > 8);
        println!("RGBA GeoTIFF: {} bytes", bytes.len());
    }

    #[test]
    fn test_write_compressed_lzw() {
        let raster = create_test_raster(1, 128, 128);
        let uncompressed = raster.to_geotiff_bytes().unwrap();
        let compressed = raster.to_geotiff_bytes_compressed(GeoTiffCompression::Lzw).unwrap();

        println!(
            "Uncompressed: {} bytes, LZW: {} bytes",
            uncompressed.len(),
            compressed.len()
        );
        // Compressed should generally be smaller for this test data
        // (though not always for random data)
    }

    #[test]
    fn test_write_compressed_deflate() {
        let raster = create_test_raster(1, 128, 128);
        let compressed = raster
            .to_geotiff_bytes_compressed(GeoTiffCompression::Deflate)
            .unwrap();

        assert!(compressed.len() > 8);
        println!("Deflate GeoTIFF: {} bytes", compressed.len());
    }

    #[test]
    fn test_geokey_directory_projected() {
        let raster = create_test_raster(1, 10, 10);
        let writer = GeoTiffWriter::new(&raster);
        let geokeys = writer.build_geokey_directory(&raster);

        // Check header
        assert_eq!(geokeys[0], 1); // Version
        assert_eq!(geokeys[1], 1); // Revision
        assert_eq!(geokeys[2], 0); // Minor revision
        assert_eq!(geokeys[3], 3); // Number of keys

        // Check GTModelTypeGeoKey = Projected
        assert_eq!(geokeys[4], GT_MODEL_TYPE_GEO_KEY);
        assert_eq!(geokeys[7], MODEL_TYPE_PROJECTED);

        // Check ProjectedCSTypeGeoKey = 32610
        assert_eq!(geokeys[12], PROJECTED_CS_TYPE_GEO_KEY);
        assert_eq!(geokeys[15], 32610);
    }

    #[test]
    fn test_geokey_directory_geographic() {
        let mut raster = create_test_raster(1, 10, 10);
        raster.crs = 4326;
        raster.bounds = BoundingBox::new(-122.5, 37.0, -122.0, 37.5);
        raster.resolution = (0.05, 0.05);

        let writer = GeoTiffWriter::new(&raster);
        let geokeys = writer.build_geokey_directory(&raster);

        // Check GTModelTypeGeoKey = Geographic
        assert_eq!(geokeys[7], MODEL_TYPE_GEOGRAPHIC);

        // Check GeographicTypeGeoKey = 4326
        assert_eq!(geokeys[12], GEOGRAPHIC_TYPE_GEO_KEY);
        assert_eq!(geokeys[15], 4326);
    }

    #[test]
    fn test_empty_raster_error() {
        let raster = ReprojectedRaster {
            pixels: vec![],
            bands: 1,
            width: 0,
            height: 0,
            crs: 32610,
            bounds: BoundingBox::new(0.0, 0.0, 0.0, 0.0),
            resolution: (1.0, 1.0),
            nodata: None,
        };

        let result = raster.to_geotiff_bytes();
        assert!(result.is_err());
    }

    #[test]
    fn test_multiband_geotiff() {
        // Test writing 5-band raster (arbitrary band count)
        let raster = ReprojectedRaster {
            pixels: vec![1.0; 100 * 5], // 5 bands, 10x10 pixels
            bands: 5,
            width: 10,
            height: 10,
            crs: 32610,
            bounds: BoundingBox::new(0.0, 0.0, 100.0, 100.0),
            resolution: (10.0, 10.0),
            nodata: None,
        };

        let bytes = raster.to_geotiff_bytes().unwrap();
        assert!(!bytes.is_empty());
        println!("5-band GeoTIFF: {} bytes", bytes.len());
    }

    #[test]
    fn test_two_band_geotiff() {
        // Test 2-band raster
        let raster = ReprojectedRaster {
            pixels: vec![1.0; 64 * 2], // 2 bands, 8x8 pixels
            bands: 2,
            width: 8,
            height: 8,
            crs: 32610,
            bounds: BoundingBox::new(0.0, 0.0, 80.0, 80.0),
            resolution: (10.0, 10.0),
            nodata: None,
        };

        let bytes = raster.to_geotiff_bytes().unwrap();
        assert!(!bytes.is_empty());
        println!("2-band GeoTIFF: {} bytes", bytes.len());
    }

    #[test]
    fn test_many_band_geotiff() {
        // Test 16-band raster (hyperspectral-like)
        let bands = 16;
        let raster = ReprojectedRaster {
            pixels: vec![0.5; 100 * bands], // 16 bands, 10x10 pixels
            bands,
            width: 10,
            height: 10,
            crs: 32610,
            bounds: BoundingBox::new(0.0, 0.0, 100.0, 100.0),
            resolution: (10.0, 10.0),
            nodata: None,
        };

        let bytes = raster.to_geotiff_bytes().unwrap();
        assert!(!bytes.is_empty());
        println!("16-band GeoTIFF: {} bytes", bytes.len());

        // Verify it can be read back
        let cursor = std::io::Cursor::new(bytes);
        let mut decoder = tiff::decoder::Decoder::new(cursor).unwrap();
        let (width, height) = decoder.dimensions().unwrap();
        assert_eq!(width, 10);
        assert_eq!(height, 10);
    }

    #[test]
    fn test_write_to_file() {
        let raster = create_test_raster(1, 64, 64);
        let temp_path = "/tmp/test_geotiff_write.tif";

        // Write to file
        raster.write_geotiff(temp_path).unwrap();

        // Check file exists and has content
        let metadata = std::fs::metadata(temp_path).unwrap();
        assert!(metadata.len() > 0);

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_roundtrip_verify_geotiff() {
        use tiff::decoder::Decoder;

        let raster = create_test_raster(1, 32, 32);
        let bytes = raster.to_geotiff_bytes().unwrap();

        // Read back with tiff decoder
        let cursor = std::io::Cursor::new(bytes);
        let mut decoder = Decoder::new(cursor).unwrap();

        // Check dimensions
        let (width, height) = decoder.dimensions().unwrap();
        assert_eq!(width, 32);
        assert_eq!(height, 32);

        println!("Roundtrip GeoTIFF verified: {}x{}", width, height);
    }

    // TODO: Re-enable when CogReader.reproject() method is implemented
    // #[tokio::test]
    // async fn test_reproject_and_write_geotiff() { ... }

    #[test]
    fn test_various_utm_crs() {
        // Test that we correctly generate GeoKeys for various CRS
        let crs_list = [
            (32610, "UTM 10N"),
            (32618, "UTM 18N"),
            (32633, "UTM 33N"),
            (4326, "WGS84"),
            (3857, "Web Mercator"),
        ];

        for (crs, name) in crs_list {
            let mut raster = create_test_raster(1, 10, 10);
            raster.crs = crs;

            let bytes = raster.to_geotiff_bytes().unwrap();
            assert!(!bytes.is_empty(), "Failed to write GeoTIFF for {name} (EPSG:{crs})");
            println!("EPSG:{crs} ({name}): {} bytes", bytes.len());
        }
    }
}
