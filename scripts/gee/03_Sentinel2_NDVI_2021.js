// ============================================================================
// CHATTOGRAM FLOOD SUSCEPTIBILITY RESEARCH
// SEGMENT 03 — SENTINEL-2 NDVI
//
// Exact study area:
// projects/gee-project-505/assets/AOI_Boundary
//
// Master extraction extent:
// projects/gee-project-505/assets/Master_Area_Extraction
//
// Dataset     : Sentinel-2 Level-2A Surface Reflectance Harmonized
// GEE ID      : COPERNICUS/S2_SR_HARMONIZED
// Cloud QA    : GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED
// Period      : 2021-01-01 to 2021-12-31
// Final metric: Annual median NDVI
// Native bands: B4 (Red, 10 m), B8 (NIR, 10 m)
// ============================================================================


// ============================================================================
// 0. PROJECT SETTINGS
// ============================================================================

var AOI_ASSET =
  'projects/gee-project-505/assets/AOI_Boundary';

var MASTER_AREA_ASSET =
  'projects/gee-project-505/assets/Master_Area_Extraction';

var DRIVE_FOLDER =
  'Chattogram_Flood_Research_Data';

var START_DATE = '2021-01-01';
var END_DATE   = '2022-01-01';

// Recommended Cloud Score+ clear-pixel threshold.
var CLEAR_THRESHOLD = 0.60;


// ============================================================================
// 1. LOAD GEOMETRIES
// ============================================================================

var studyArea =
  ee.FeatureCollection(AOI_ASSET);

var masterArea =
  ee.FeatureCollection(MASTER_AREA_ASSET);

var studyGeometry =
  studyArea.geometry();

var exportGeometry =
  masterArea.geometry();

print('================ GEOMETRY QC ================');
print('Exact AOI:', studyArea);
print('Master extraction area:', masterArea);


// ============================================================================
// 2. LOAD SENTINEL-2 SURFACE REFLECTANCE
// ============================================================================

var s2 =
  ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(exportGeometry)
    .filterDate(START_DATE, END_DATE);

print('================ SENTINEL-2 QC ================');
print('Sentinel-2 SR collection:', s2);
print('Number of Sentinel-2 scenes:', s2.size());


// ============================================================================
// 3. LOAD CLOUD SCORE+
// ============================================================================

var cloudScore =
  ee.ImageCollection(
    'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'
  )
    .filterBounds(exportGeometry)
    .filterDate(START_DATE, END_DATE);

print('Cloud Score+ collection:', cloudScore);
print('Cloud Score+ scene count:', cloudScore.size());


// ============================================================================
// 4. LINK CLOUD SCORE+ TO SENTINEL-2
// ============================================================================
//
// cs_cdf ranges approximately from 0 to 1.
// Higher values indicate clearer pixels.
// ============================================================================

var linked =
  s2.linkCollection(
    cloudScore,
    ['cs_cdf']
  );


// ============================================================================
// 5. CLOUD / SHADOW MASK + NDVI FUNCTION
// ============================================================================

function prepareNDVI(image) {

  // ----------------------------------------------------------
  // Cloud Score+ clear-pixel mask
  // ----------------------------------------------------------

  var clearMask =
    image.select('cs_cdf')
      .gte(CLEAR_THRESHOLD);


  // ----------------------------------------------------------
  // Sentinel-2 Scene Classification Layer (SCL) mask
  //
  // Remove:
  // 1  = saturated / defective
  // 3  = cloud shadow
  // 8  = medium-probability cloud
  // 9  = high-probability cloud
  // 10 = cirrus
  // 11 = snow / ice
  //
  // Water (6) is NOT removed here.
  // ----------------------------------------------------------

  var scl =
    image.select('SCL');

  var sclMask =
    scl.neq(1)
      .and(scl.neq(3))
      .and(scl.neq(8))
      .and(scl.neq(9))
      .and(scl.neq(10))
      .and(scl.neq(11));


  // ----------------------------------------------------------
  // Combine quality masks
  // ----------------------------------------------------------

  var qualityMask =
    clearMask.and(sclMask);


  // ----------------------------------------------------------
  // NDVI = (NIR - RED) / (NIR + RED)
  //
  // B8 = NIR
  // B4 = Red
  // ----------------------------------------------------------

  var ndvi =
    image
      .updateMask(qualityMask)
      .normalizedDifference(
        ['B8', 'B4']
      )
      .rename('NDVI');


  return ndvi
    .copyProperties(
      image,
      ['system:time_start']
    );
}


// ============================================================================
// 6. CREATE QUALITY-MASKED NDVI COLLECTION
// ============================================================================

var ndviCollection =
  linked.map(prepareNDVI);

print(
  'Quality-masked NDVI collection:',
  ndviCollection
);


// ============================================================================
// 7. CALCULATE 2021 ANNUAL MEDIAN NDVI
// ============================================================================
//
// Pixel-wise median of all valid cloud-free 2021 observations.
// ============================================================================

var NDVI_2021 =
  ndviCollection
    .median()
    .rename('NDVI_2021_Median');

print('================ NDVI OUTPUT ================');
print('2021 median NDVI:', NDVI_2021);


// ============================================================================
// 8. VALID OBSERVATION COUNT
// ============================================================================
//
// Number of valid cloud-free NDVI observations contributing
// to each pixel.
//
// This is a QC raster, NOT a flood-conditioning factor.
// ============================================================================

var validObservationCount =
  ndviCollection
    .count()
    .rename('Valid_Obs_Count_2021');

print(
  'Valid observation-count raster:',
  validObservationCount
);


// ============================================================================
// 9. NDVI STATISTICS — EXACT AOI ONLY
// ============================================================================

var NDVIstats =
  NDVI_2021.reduceRegion({

    reducer:
      ee.Reducer.minMax()
        .combine({
          reducer2:
            ee.Reducer.mean(),
          sharedInputs:
            true
        })
        .combine({
          reducer2:
            ee.Reducer.stdDev(),
          sharedInputs:
            true
        }),

    geometry:
      studyGeometry,

    // QC statistics at analytical-grid scale.
    scale:
      30,

    maxPixels:
      1e10,

    tileScale:
      4
  });

print(
  'NDVI statistics over exact Chattogram AOI:',
  NDVIstats
);


// ============================================================================
// 10. VALID-OBSERVATION STATISTICS
// ============================================================================

var observationStats =
  validObservationCount.reduceRegion({

    reducer:
      ee.Reducer.minMax()
        .combine({
          reducer2:
            ee.Reducer.mean(),
          sharedInputs:
            true
        }),

    geometry:
      studyGeometry,

    scale:
      30,

    maxPixels:
      1e10,

    tileScale:
      4
  });

print(
  'Valid observation-count statistics:',
  observationStats
);


// ============================================================================
// 11. METADATA / QC TABLE
// ============================================================================

var metadata =
  ee.FeatureCollection([

    ee.Feature(
      null,
      {
        dataset:
          'Sentinel-2 SR Harmonized',

        gee_collection:
          'COPERNICUS/S2_SR_HARMONIZED',

        cloud_dataset:
          'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED',

        study_area:
          'Chattogram, Bangladesh',

        temporal_period:
          '2021-01-01 to 2021-12-31',

        metric:
          'Annual pixel-wise median NDVI',

        red_band:
          'B4',

        nir_band:
          'B8',

        native_band_resolution_m:
          10,

        cloud_score_band:
          'cs_cdf',

        clear_threshold:
          CLEAR_THRESHOLD,

        resampling_for_final_alignment:
          'Bilinear',

        NDVI_min:
          NDVIstats.get(
            'NDVI_2021_Median_min'
          ),

        NDVI_max:
          NDVIstats.get(
            'NDVI_2021_Median_max'
          ),

        NDVI_mean:
          NDVIstats.get(
            'NDVI_2021_Median_mean'
          ),

        NDVI_stdDev:
          NDVIstats.get(
            'NDVI_2021_Median_stdDev'
          ),

        valid_obs_min:
          observationStats.get(
            'Valid_Obs_Count_2021_min'
          ),

        valid_obs_max:
          observationStats.get(
            'Valid_Obs_Count_2021_max'
          ),

        valid_obs_mean:
          observationStats.get(
            'Valid_Obs_Count_2021_mean'
          )
      }
    )

  ]);

print(
  'NDVI metadata/QC:',
  metadata
);


// ============================================================================
// 12. VISUALIZATION
// ============================================================================

Map.centerObject(
  studyArea,
  8
);

Map.addLayer(
  masterArea,
  {color: 'blue'},
  'Master Extraction Area',
  false
);

Map.addLayer(
  studyArea,
  {color: 'red'},
  'Exact Chattogram AOI'
);


var ndviVis = {

  min:
    -0.2,

  max:
    0.8,

  palette: [
    '8c510a',
    'd8b365',
    'f6e8c3',
    'c7eae5',
    '5ab4ac',
    '01665e'
  ]
};


Map.addLayer(
  NDVI_2021.clip(studyGeometry),
  ndviVis,
  'Sentinel-2 Median NDVI 2021'
);


Map.addLayer(
  validObservationCount.clip(studyGeometry),
  {
    min: 0,
    max: 50,
    palette: [
      'red',
      'yellow',
      'green'
    ]
  },
  'Valid Observation Count 2021',
  false
);


// ============================================================================
// 13. EXPORT NDVI OVER MASTER EXTRACTION AREA
// ============================================================================
//
// Export at 10 m in UTM Zone 46N.
//
// Later in QGIS:
// Align to DEM 30 m grid using BILINEAR resampling.
// ============================================================================

Export.image.toDrive({

  image:
    NDVI_2021.toFloat(),

  description:
    '03_NDVI_Sentinel2_Median_2021',

  folder:
    DRIVE_FOLDER,

  fileNamePrefix:
    '03_NDVI_Sentinel2_Median_2021',

  region:
    exportGeometry,

  crs:
    'EPSG:32646',

  scale:
    10,

  maxPixels:
    1e13,

  fileFormat:
    'GeoTIFF',

  formatOptions: {
    cloudOptimized: true
  }

});


// ============================================================================
// 14. EXPORT VALID OBSERVATION-COUNT QC RASTER
// ============================================================================

Export.image.toDrive({

  image:
    validObservationCount.toUint16(),

  description:
    '03_NDVI_Sentinel2_ValidObsCount_2021',

  folder:
    DRIVE_FOLDER,

  fileNamePrefix:
    '03_NDVI_Sentinel2_ValidObsCount_2021',

  region:
    exportGeometry,

  crs:
    'EPSG:32646',

  scale:
    10,

  maxPixels:
    1e13,

  fileFormat:
    'GeoTIFF'
});


// ============================================================================
// 15. EXPORT METADATA / QC CSV
// ============================================================================

Export.table.toDrive({

  collection:
    metadata,

  description:
    '03_NDVI_Sentinel2_2021_Metadata',

  folder:
    DRIVE_FOLDER,

  fileNamePrefix:
    '03_NDVI_Sentinel2_2021_Metadata',

  fileFormat:
    'CSV'
});


// ============================================================================
// END — SEGMENT 03
// ============================================================================