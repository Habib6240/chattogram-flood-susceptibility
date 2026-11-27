// ============================================================================
// CHATTOGRAM FLOOD SUSCEPTIBILITY RESEARCH
// SEGMENT 02 — ESA WORLDCOVER 2021 v200
//
// Exact AOI:
// projects/gee-project-505/assets/AOI_Boundary
//
// Common master extraction extent:
// projects/gee-project-505/assets/Master_Area_Extraction
//
// Dataset : ESA WorldCover 2021 v200
// GEE ID  : ESA/WorldCover/v200
// Variable: Land Use / Land Cover
// Native resolution: 10 m
// Data type: Categorical
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

print('Exact Chattogram AOI:', studyArea);
print('Master extraction area:', masterArea);


// ============================================================================
// 2. LOAD ESA WORLDCOVER 2021 v200
// ============================================================================

var worldCover =
  ee.ImageCollection('ESA/WorldCover/v200')
    .first()
    .select('Map');

print('ESA WorldCover image:', worldCover);


// ============================================================================
// 3. SOURCE PROJECTION / RESOLUTION QC
// ============================================================================

var sourceProjection =
  worldCover.projection();

print(
  'WorldCover native projection:',
  sourceProjection
);

print(
  'WorldCover nominal pixel size (m):',
  sourceProjection.nominalScale()
);

var sourceProjectionInfo =
  sourceProjection.getInfo();


// ============================================================================
// 4. CLASS HISTOGRAM OVER EXACT CHATTOGRAM AOI
// ============================================================================

var classHistogram =
  worldCover.reduceRegion({

    reducer:
      ee.Reducer.frequencyHistogram(),

    geometry:
      studyGeometry,

    scale:
      10,

    maxPixels:
      1e10

  });

print(
  'LULC class pixel-frequency histogram over AOI:',
  classHistogram
);


// ============================================================================
// 5. CLASS DEFINITIONS
// ============================================================================
//
// 10  Tree cover
// 20  Shrubland
// 30  Grassland
// 40  Cropland
// 50  Built-up
// 60  Bare / sparse vegetation
// 70  Snow and ice
// 80  Permanent water bodies
// 90  Herbaceous wetland
// 95  Mangroves
// 100 Moss and lichen
// ============================================================================

var classCodes =
  ee.List([
    10, 20, 30, 40, 50,
    60, 70, 80, 90, 95, 100
  ]);

var classNames =
  ee.List([
    'Tree cover',
    'Shrubland',
    'Grassland',
    'Cropland',
    'Built-up',
    'Bare or sparse vegetation',
    'Snow and ice',
    'Permanent water bodies',
    'Herbaceous wetland',
    'Mangroves',
    'Moss and lichen'
  ]);


// ============================================================================
// 6. CALCULATE AREA OF EACH CLASS WITHIN EXACT AOI
// ============================================================================

var classAreaTable =
  ee.FeatureCollection(

    ee.List.sequence(
      0,
      classCodes.length().subtract(1)
    ).map(function(i) {

      i = ee.Number(i);

      var classCode =
        ee.Number(classCodes.get(i));

      var className =
        ee.String(classNames.get(i));

      var areaImage =
        ee.Image.pixelArea()
          .updateMask(
            worldCover.eq(classCode)
          );

      var areaResult =
        areaImage.reduceRegion({

          reducer:
            ee.Reducer.sum(),

          geometry:
            studyGeometry,

          scale:
            10,

          maxPixels:
            1e10

        });

      var areaM2 =
        ee.Number(
          ee.Algorithms.If(
            areaResult.get('area'),
            areaResult.get('area'),
            0
          )
        );

      return ee.Feature(
        null,
        {
          class_code:
            classCode,

          class_name:
            className,

          area_m2:
            areaM2,

          area_km2:
            areaM2.divide(1e6)
        }
      );

    })

  );

print(
  'WorldCover class-area table:',
  classAreaTable
);


// ============================================================================
// 7. METADATA TABLE
// ============================================================================

var metadata =
  ee.FeatureCollection([

    ee.Feature(
      null,
      {
        dataset:
          'ESA WorldCover 2021 v200',

        gee_collection:
          'ESA/WorldCover/v200',

        reference_year:
          2021,

        native_resolution_m:
          10,

        number_of_classes:
          11,

        variable:
          'Land Use / Land Cover',

        data_type:
          'Categorical',

        recommended_resampling:
          'Nearest Neighbour',

        study_area:
          'Chattogram, Bangladesh'
      }
    )

  ]);

print(
  'LULC metadata:',
  metadata
);


// ============================================================================
// 8. VISUALIZATION
// ============================================================================

var palette = [
  '006400',  // 10 Tree
  'ffbb22',  // 20 Shrub
  'ffff4c',  // 30 Grass
  'f096ff',  // 40 Cropland
  'fa0000',  // 50 Built-up
  'b4b4b4',  // 60 Bare
  'f0f0f0',  // 70 Snow
  '0064c8',  // 80 Water
  '0096a0',  // 90 Wetland
  '00cf75',  // 95 Mangrove
  'fae6a0'   // 100 Moss/lichen
];

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
  worldCover.clip(studyGeometry),
  {
    min: 10,
    max: 100,
    palette: palette
  },
  'ESA WorldCover 2021 v200'
);


// ============================================================================
// 9. EXPORT NATIVE WORLDCOVER OVER MASTER AREA
// ============================================================================
//
// IMPORTANT:
// LULC is categorical.
//
// Native 10 m grid is retained.
// Later QGIS alignment to the 30 m master grid MUST use
// Nearest Neighbour, NOT Bilinear.
// ============================================================================

Export.image.toDrive({

  image:
    worldCover.toUint8(),

  description:
    '02_LULC_ESA_WorldCover_2021_v200',

  folder:
    DRIVE_FOLDER,

  fileNamePrefix:
    '02_LULC_ESA_WorldCover_2021_v200',

  region:
    exportGeometry,

  crs:
    sourceProjectionInfo.crs,

  crsTransform:
    sourceProjectionInfo.transform,

  maxPixels:
    1e13,

  fileFormat:
    'GeoTIFF',

  formatOptions: {
    cloudOptimized: true
  }

});


// ============================================================================
// 10. EXPORT CLASS-AREA QC TABLE
// ============================================================================

Export.table.toDrive({

  collection:
    classAreaTable,

  description:
    '02_LULC_ESA_WorldCover_2021_ClassArea_QC',

  folder:
    DRIVE_FOLDER,

  fileNamePrefix:
    '02_LULC_ESA_WorldCover_2021_ClassArea_QC',

  fileFormat:
    'CSV'
});


// ============================================================================
// 11. EXPORT METADATA
// ============================================================================

Export.table.toDrive({

  collection:
    metadata,

  description:
    '02_LULC_ESA_WorldCover_2021_Metadata',

  folder:
    DRIVE_FOLDER,

  fileNamePrefix:
    '02_LULC_ESA_WorldCover_2021_Metadata',

  fileFormat:
    'CSV'
});


// ============================================================================
// END — SEGMENT 02
// ============================================================================