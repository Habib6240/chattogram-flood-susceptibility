// ============================================================================
// CHATTogram FLOOD SUSCEPTIBILITY RESEARCH
// SEGMENT 01 — CHIRPS v3 CLIMATOLOGICAL RAINFALL
//
// Exact study area:
// projects/gee-project-505/assets/AOI_Boundary
//
// Master extraction area:
// projects/gee-project-505/assets/Master_Area_Extraction
//
// Dataset : CHIRPS v3 Pentad
// Period  : 1991–2020
// Output  : Mean Annual Precipitation (MAP), mm/year
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

var START_YEAR = 1991;
var END_YEAR   = 2020;
var N_YEARS    = END_YEAR - START_YEAR + 1;


// ============================================================================
// 1. LOAD BOTH GEOMETRIES AUTOMATICALLY
// ============================================================================

var studyArea =
  ee.FeatureCollection(AOI_ASSET);

var masterArea =
  ee.FeatureCollection(MASTER_AREA_ASSET);

var studyGeometry =
  studyArea.geometry();

var exportGeometry =
  masterArea.geometry();

print('================ STUDY AREA ================');
print('Exact AOI:', studyArea);
print('Master extraction area:', masterArea);


// ============================================================================
// 2. LOAD CHIRPS v3 PENTAD
// ============================================================================

var chirps =
  ee.ImageCollection('UCSB-CHC/CHIRPS/V3/PENTAD')
    .filterDate(
      START_YEAR + '-01-01',
      (END_YEAR + 1) + '-01-01'
    )
    .select('precipitation');

print('================ CHIRPS QC ================');

print(
  'Actual pentad count:',
  chirps.size()
);

print(
  'Expected pentad count:',
  N_YEARS * 72
);


// ============================================================================
// 3. SOURCE GRID / PROJECTION QC
// ============================================================================

var firstImage =
  ee.Image(chirps.first());

var sourceProjection =
  firstImage.projection();

print(
  'CHIRPS native projection:',
  sourceProjection
);

print(
  'CHIRPS nominal pixel size (m):',
  sourceProjection.nominalScale()
);

// Used later to preserve the native CHIRPS grid.
var sourceProjectionInfo =
  sourceProjection.getInfo();


// ============================================================================
// 4. TEMPORAL COMPLETENESS QC
// ============================================================================

var pentadCountByYear =
  chirps.aggregate_histogram('year');

print(
  'Pentad count by year:',
  pentadCountByYear
);


// ============================================================================
// 5. YEAR-BY-YEAR QC TABLE
// ============================================================================

var years =
  ee.List.sequence(
    START_YEAR,
    END_YEAR
  );

var annualQC =
  ee.FeatureCollection(

    years.map(function(year) {

      year = ee.Number(year);

      var start =
        ee.Date.fromYMD(
          year,
          1,
          1
        );

      var end =
        start.advance(
          1,
          'year'
        );

      var yearlyCollection =
        chirps.filterDate(
          start,
          end
        );

      // CHIRPS precipitation = mm/pentad
      // Sum = annual precipitation in mm/year.

      var annualRainfall =
        yearlyCollection
          .sum()
          .rename(
            'Annual_Precip_mm'
          );

      // Statistics are calculated ONLY over the exact Chattogram AOI.

      var aoiMean =
        annualRainfall
          .reduceRegion({
            reducer:
              ee.Reducer.mean(),

            geometry:
              studyGeometry,

            scale:
              5566,

            maxPixels:
              1e9
          })
          .get(
            'Annual_Precip_mm'
          );

      return ee.Feature(
        null,
        {
          year:
            year,

          pentad_count:
            yearlyCollection.size(),

          AOI_mean_annual_rainfall_mm:
            aoiMean
        }
      );

    })

  );

print(
  'Annual rainfall QC table:',
  annualQC
);


// ============================================================================
// 6. MEAN ANNUAL PRECIPITATION — 1991–2020
// ============================================================================
//
// Sum all pentads over 30 years and divide by 30.
//
// Result = climatological Mean Annual Precipitation
// Unit   = mm/year.
//
// IMPORTANT:
// Do NOT clip MAP to AOI here.
// We need valid surrounding CHIRPS cells for later reprojection/resampling.
// ============================================================================

var MAP =
  chirps
    .sum()
    .divide(N_YEARS)
    .rename(
      'MAP_1991_2020_mm_year'
    );

print('================ MAP ================');
print('MAP raster:', MAP);


// ============================================================================
// 7. MAP STATISTICS — EXACT AOI ONLY
// ============================================================================

var MAPstats =
  MAP.reduceRegion({

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
      5566,

    maxPixels:
      1e9

  });

print(
  'MAP statistics over exact Chattogram AOI:',
  MAPstats
);


// ============================================================================
// 8. METADATA / QC TABLE
// ============================================================================

var metadata =
  ee.FeatureCollection([

    ee.Feature(
      null,
      {
        dataset:
          'CHIRPS v3 Pentad',

        gee_collection:
          'UCSB-CHC/CHIRPS/V3/PENTAD',

        study_area:
          'Chattogram, Bangladesh',

        temporal_period:
          '1991-2020',

        number_of_years:
          N_YEARS,

        expected_pentads:
          N_YEARS * 72,

        actual_pentads:
          chirps.size(),

        native_resolution_degree:
          0.05,

        nominal_resolution_m:
          sourceProjection.nominalScale(),

        output_variable:
          'Mean Annual Precipitation',

        output_unit:
          'mm/year',

        MAP_min_mm_year:
          MAPstats.get(
            'MAP_1991_2020_mm_year_min'
          ),

        MAP_max_mm_year:
          MAPstats.get(
            'MAP_1991_2020_mm_year_max'
          ),

        MAP_mean_mm_year:
          MAPstats.get(
            'MAP_1991_2020_mm_year_mean'
          ),

        MAP_stdDev_mm_year:
          MAPstats.get(
            'MAP_1991_2020_mm_year_stdDev'
          )
      }
    )

  ]);

print(
  'Rainfall metadata/QC:',
  metadata
);


// ============================================================================
// 9. MAP VISUALIZATION
// ============================================================================

// Exact AOI
Map.centerObject(
  studyArea,
  8
);

Map.addLayer(
  masterArea,
  {
    color: 'blue'
  },
  'Master Extraction Area',
  false
);

Map.addLayer(
  studyArea,
  {
    color: 'red'
  },
  'Exact Chattogram AOI'
);


// Display only the AOI portion for visual inspection.
// This DOES NOT alter the exported raster.

var MAP_AOI_preview =
  MAP.clip(
    studyGeometry
  );

var rainfallVis = {
  min: 2700,
  max: 3750,

  palette: [
    'ffffcc',
    'c2e699',
    '78c679',
    '31a354',
    '006837'
  ]
};

Map.addLayer(
  MAP_AOI_preview,
  rainfallVis,
  'CHIRPS v3 MAP 1991-2020'
);


// ============================================================================
// 10. EXPORT MAP OVER MASTER EXTRACTION AREA
// ============================================================================
//
// Critical design:
//
// Statistics → exact AOI
// Export     → larger Master_Area_Extraction
//
// This provides surrounding rainfall cells for later bilinear reprojection
// and prevents edge gaps along the exact Chattogram boundary.
//
// Native CHIRPS CRS/grid is retained.
// ============================================================================

Export.image.toDrive({

  image:
    MAP.toFloat(),

  description:
    '01_Rainfall_CHIRPSv3_MAP_1991_2020',

  folder:
    DRIVE_FOLDER,

  fileNamePrefix:
    '01_Rainfall_CHIRPSv3_MAP_1991_2020',

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
// 11. EXPORT ANNUAL QC CSV
// ============================================================================

Export.table.toDrive({

  collection:
    annualQC,

  description:
    '01_Rainfall_CHIRPSv3_Annual_QC_1991_2020',

  folder:
    DRIVE_FOLDER,

  fileNamePrefix:
    '01_Rainfall_CHIRPSv3_Annual_QC_1991_2020',

  fileFormat:
    'CSV'
});


// ============================================================================
// 12. EXPORT METADATA CSV
// ============================================================================

Export.table.toDrive({

  collection:
    metadata,

  description:
    '01_Rainfall_CHIRPSv3_Metadata_1991_2020',

  folder:
    DRIVE_FOLDER,

  fileNamePrefix:
    '01_Rainfall_CHIRPSv3_Metadata_1991_2020',

  fileFormat:
    'CSV'
});


// ============================================================================
// END — SEGMENT 01
// ============================================================================