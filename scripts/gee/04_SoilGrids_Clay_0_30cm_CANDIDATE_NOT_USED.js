// ============================================================================
// CHATTOGRAM FLOOD SUSCEPTIBILITY RESEARCH
// SEGMENT 04 — SOILGRIDS TOPSOIL CLAY CONTENT
//
// Exact AOI:
// projects/gee-project-505/assets/AOI_Boundary
//
// Master extraction extent:
// projects/gee-project-505/assets/Master_Area_Extraction
//
// Dataset : SoilGrids 2.0
// Property: Clay content
// Depth   : Thickness-weighted 0–30 cm
// Native information resolution: 250 m
// Output unit: percent (%)
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

print('Exact AOI:', studyArea);
print('Master extraction area:', masterArea);


// ============================================================================
// 2. LOAD SOILGRIDS 2.0 CLAY
// ============================================================================

var clay =
  ee.Image('projects/soilgrids-isric/clay_mean');

print('================ SOILGRIDS QC ================');
print('Clay image:', clay);
print('Available bands:', clay.bandNames());


// ============================================================================
// 3. EXTRACT TOPSOIL DEPTH INTERVALS
// ============================================================================
//
// SoilGrids mapped unit for clay = g/kg.
// Divide by 10 to convert to conventional percentage.
//
// Required depth intervals:
// 0–5 cm
// 5–15 cm
// 15–30 cm
// ============================================================================

var clay_0_5 =
  clay
    .select('clay_0-5cm_mean')
    .divide(10)
    .rename('Clay_0_5cm_pct');

var clay_5_15 =
  clay
    .select('clay_5-15cm_mean')
    .divide(10)
    .rename('Clay_5_15cm_pct');

var clay_15_30 =
  clay
    .select('clay_15-30cm_mean')
    .divide(10)
    .rename('Clay_15_30cm_pct');


// ============================================================================
// 4. THICKNESS-WEIGHTED 0–30 cm CLAY CONTENT
// ============================================================================
//
// Thickness:
// 0–5   = 5 cm
// 5–15  = 10 cm
// 15–30 = 15 cm
//
// Weighted average:
//
// (Clay0-5×5 + Clay5-15×10 + Clay15-30×15) / 30
// ============================================================================

var clay_0_30 =
  clay_0_5.multiply(5)
    .add(
      clay_5_15.multiply(10)
    )
    .add(
      clay_15_30.multiply(15)
    )
    .divide(30)
    .rename('Clay_0_30cm_pct')
    .toFloat();

print(
  'Topsoil clay 0–30 cm:',
  clay_0_30
);


// ============================================================================
// 5. AOI STATISTICS
// ============================================================================

var clayStats =
  clay_0_30.reduceRegion({

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

    scale:
      250,

    maxPixels:
      1e9

  });

print(
  'Clay 0–30 cm statistics over Chattogram:',
  clayStats
);


// ============================================================================
// 6. METADATA
// ============================================================================

var metadata =
  ee.FeatureCollection([

    ee.Feature(
      null,
      {
        dataset:
          'SoilGrids 2.0',

        gee_asset:
          'projects/soilgrids-isric/clay_mean',

        study_area:
          'Chattogram, Bangladesh',

        property:
          'Clay content',

        depth:
          '0-30 cm thickness-weighted mean',

        source_resolution_m:
          250,

        raw_unit:
          'g/kg',

        output_unit:
          'percent',

        conversion:
          'raw value divided by 10',

        weighting:
          '0-5cm x5; 5-15cm x10; 15-30cm x15; divided by 30',

        recommended_final_resampling:
          'Bilinear',

        clay_min_pct:
          clayStats.get(
            'Clay_0_30cm_pct_min'
          ),

        clay_max_pct:
          clayStats.get(
            'Clay_0_30cm_pct_max'
          ),

        clay_mean_pct:
          clayStats.get(
            'Clay_0_30cm_pct_mean'
          ),

        clay_stdDev_pct:
          clayStats.get(
            'Clay_0_30cm_pct_stdDev'
          )
      }
    )

  ]);

print(
  'Soil metadata:',
  metadata
);


// ============================================================================
// 7. VISUALIZATION
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

Map.addLayer(
  clay_0_30.clip(studyGeometry),
  {
    min: 10,
    max: 50,
    palette: [
      'ffffcc',
      'ffeda0',
      'feb24c',
      'f03b20',
      'bd0026'
    ]
  },
  'SoilGrids Clay 0-30 cm (%)'
);


// ============================================================================
// 8. EXPORT OVER MASTER EXTRACTION AREA
// ============================================================================
//
// Export at source-information scale (250 m) in UTM 46N.
// Later align to the 30 m DEM reference grid using BILINEAR.
// ============================================================================

Export.image.toDrive({

  image:
    clay_0_30,

  description:
    '04_Soil_SoilGrids_Clay_0_30cm',

  folder:
    DRIVE_FOLDER,

  fileNamePrefix:
    '04_Soil_SoilGrids_Clay_0_30cm',

  region:
    exportGeometry,

  crs:
    'EPSG:32646',

  scale:
    250,

  maxPixels:
    1e13,

  fileFormat:
    'GeoTIFF',

  formatOptions: {
    cloudOptimized: true
  }

});


// ============================================================================
// 9. EXPORT METADATA CSV
// ============================================================================

Export.table.toDrive({

  collection:
    metadata,

  description:
    '04_Soil_SoilGrids_Clay_0_30cm_Metadata',

  folder:
    DRIVE_FOLDER,

  fileNamePrefix:
    '04_Soil_SoilGrids_Clay_0_30cm_Metadata',

  fileFormat:
    'CSV'

});


// ============================================================================
// END — SEGMENT 04
// ============================================================================