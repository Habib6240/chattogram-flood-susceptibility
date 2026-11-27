// ============================================================================
// BUET ML — B01 SENTINEL-1 MULTI-EVENT AVAILABILITY AUDIT
// Study Area: Chattogram, Bangladesh
//
// PURPOSE
// -------
// 1. Check Sentinel-1 availability around candidate flood events.
// 2. Identify PRE / DURING / POST-event scenes.
// 3. Record orbit direction, relative orbit, platform, polarization,
//    acquisition date/time and AOI coverage.
// 4. Export a detailed scene table and an event-level summary.
//
// IMPORTANT
// ---------
// - This script DOES NOT generate flood masks.
// - No flood thresholding is performed here.
// - ee.ImageCollection.filterDate(start, end) uses an EXCLUSIVE end date.
// - Zero-scene events are handled safely.
// ============================================================================



// ============================================================================
// 01. AOI
// ============================================================================

var aoiFC = ee.FeatureCollection(
  'projects/gee-project-505/assets/AOI_Boundary'
);

var aoi = aoiFC.geometry();

Map.centerObject(aoi, 9);

Map.addLayer(
  aoiFC.style({
    color: 'red',
    fillColor: '00000000',
    width: 2
  }),
  {},
  'Chattogram AOI',
  false
);



// ============================================================================
// 02. CANDIDATE FLOOD EVENTS
//
// reported_start / reported_end
//     = flood period being investigated
//
// audit_start / audit_end
//     = wider Sentinel-1 discovery window
//
// NOTE:
// audit_end is EXCLUSIVE.
// ============================================================================

var events = [

  // --------------------------------------------------------------------------
  // E01 — July 2015
  // --------------------------------------------------------------------------
  {
    id: 'E01_2015_Jul',

    reported_start: '2015-07-23',
    reported_end:   '2015-07-31',

    audit_start:    '2015-07-15',
    audit_end:      '2015-08-08'
  },


  // --------------------------------------------------------------------------
  // E02 — June 2017
  // --------------------------------------------------------------------------
  {
    id: 'E02_2017_Jun',

    reported_start: '2017-06-11',
    reported_end:   '2017-06-14',

    audit_start:    '2017-06-03',
    audit_end:      '2017-06-22'
  },


  // --------------------------------------------------------------------------
  // E03 — July 2019
  // --------------------------------------------------------------------------
  {
    id: 'E03_2019_Jul',

    reported_start: '2019-07-07',
    reported_end:   '2019-07-15',

    audit_start:    '2019-06-29',
    audit_end:      '2019-07-23'
  },


  // --------------------------------------------------------------------------
  // E04 — July 2021
  // --------------------------------------------------------------------------
  {
    id: 'E04_2021_Jul',

    reported_start: '2021-07-27',
    reported_end:   '2021-07-30',

    audit_start:    '2021-07-19',
    audit_end:      '2021-08-08'
  },


  // --------------------------------------------------------------------------
  // E05 — August 2023
  // --------------------------------------------------------------------------
  {
    id: 'E05_2023_Aug',

    reported_start: '2023-08-05',
    reported_end:   '2023-08-09',

    audit_start:    '2023-07-28',
    audit_end:      '2023-08-18'
  },


  // --------------------------------------------------------------------------
  // E06 — August 2024
  // --------------------------------------------------------------------------
  {
    id: 'E06_2024_Aug',

    reported_start: '2024-08-21',
    reported_end:   '2024-08-24',

    audit_start:    '2024-08-13',
    audit_end:      '2024-09-02'
  },


  // --------------------------------------------------------------------------
  // E07 — July 2026
  //
  // Reported flood period provided for the current research workflow:
  // 09–16 July 2026
  // --------------------------------------------------------------------------
  {
    id: 'E07_2026_Jul',

    reported_start: '2026-07-09',
    reported_end:   '2026-07-16',

    // Wider window so that suitable PRE and POST scenes can also be found.
    audit_start:    '2026-07-01',

    // Exclusive end date
    audit_end:      '2026-07-24'
  }

];



// ============================================================================
// 03. BASE SENTINEL-1 COLLECTION
//
// Homogeneous acquisition filtering:
//
// Dataset       : COPERNICUS/S1_GRD
// Instrument    : IW
// Polarization  : VV + VH
//
// We DO NOT filter orbit direction yet.
// Both ASCENDING and DESCENDING are retained for availability auditing.
// ============================================================================

var s1 = ee.ImageCollection(
  'COPERNICUS/S1_GRD'
)

.filterBounds(aoi)

.filter(
  ee.Filter.eq(
    'instrumentMode',
    'IW'
  )
)

.filter(
  ee.Filter.listContains(
    'transmitterReceiverPolarisation',
    'VV'
  )
)

.filter(
  ee.Filter.listContains(
    'transmitterReceiverPolarisation',
    'VH'
  )
);


print(
  'BASE SENTINEL-1 VV+VH IW COLLECTION',
  s1
);



// ============================================================================
// 04. AOI AREA
// ============================================================================

var aoiAreaM2 = aoi.area(10);



// ============================================================================
// 05. FUNCTION:
// CONVERT ONE SENTINEL-1 IMAGE TO ONE METADATA FEATURE
// ============================================================================

function imageToFeature(image, evt) {

  image = ee.Image(image);


  // --------------------------------------------------------------------------
  // Acquisition time
  // --------------------------------------------------------------------------

  var acquisitionDate = ee.Date(
    image.get('system:time_start')
  );


  // --------------------------------------------------------------------------
  // Reported event dates
  //
  // reported_end should behave as an inclusive date.
  // Therefore advance one day and compare against an exclusive endpoint.
  // --------------------------------------------------------------------------

  var reportedStart = ee.Date(
    evt.reported_start
  );

  var reportedEndExclusive = ee.Date(
    evt.reported_end
  ).advance(
    1,
    'day'
  );


  // --------------------------------------------------------------------------
  // PRE / DURING / POST classification
  // --------------------------------------------------------------------------

  var timingClass = ee.String(

    ee.Algorithms.If(

      acquisitionDate.millis()
        .lt(
          reportedStart.millis()
        ),

      'PRE_EVENT',

      ee.Algorithms.If(

        acquisitionDate.millis()
          .lt(
            reportedEndExclusive.millis()
          ),

        'DURING_EVENT',

        'POST_EVENT'

      )
    )
  );


  // --------------------------------------------------------------------------
  // AOI coverage
  //
  // Percentage of study-area geometry intersected by the Sentinel-1 scene.
  // --------------------------------------------------------------------------

  var intersection = image
    .geometry()
    .intersection(
      aoi,
      10
    );


  var coveragePct = intersection
    .area(10)
    .divide(
      aoiAreaM2
    )
    .multiply(
      100
    );


  // --------------------------------------------------------------------------
  // High-coverage flag
  //
  // 1 = at least 90% of AOI covered
  // 0 = less than 90%
  // --------------------------------------------------------------------------

  var coverage90 = ee.Number(

    ee.Algorithms.If(

      coveragePct.gte(90),

      1,

      0

    )
  );


  // --------------------------------------------------------------------------
  // Polarization metadata
  // --------------------------------------------------------------------------

  var polList = ee.List(
    image.get(
      'transmitterReceiverPolarisation'
    )
  );


  // --------------------------------------------------------------------------
  // Output metadata feature
  // --------------------------------------------------------------------------

  return ee.Feature(
    null,
    {

      // EVENT INFORMATION
      event_id:
        evt.id,

      reported_start:
        evt.reported_start,

      reported_end:
        evt.reported_end,

      audit_start:
        evt.audit_start,

      audit_end_exclusive:
        evt.audit_end,


      // IMAGE IDENTIFICATION
      image_id:
        image.get(
          'system:index'
        ),


      // ACQUISITION TIME
      acquisition_datetime_UTC:
        acquisitionDate.format(
          'YYYY-MM-dd HH:mm:ss'
        ),

      acquisition_date:
        acquisitionDate.format(
          'YYYY-MM-dd'
        ),

      acquisition_time_ms:
        image.get(
          'system:time_start'
        ),


      // EVENT TIMING
      timing_class:
        timingClass,


      // PLATFORM / ORBIT
      platform:
        image.get(
          'platform_number'
        ),

      orbit_pass:
        image.get(
          'orbitProperties_pass'
        ),

      relative_orbit:
        image.get(
          'relativeOrbitNumber_start'
        ),

      absolute_orbit:
        image.get(
          'orbitNumber_start'
        ),


      // SENSOR
      instrument_mode:
        image.get(
          'instrumentMode'
        ),

      polarizations:
        polList.join('+'),

      resolution_m:
        image.get(
          'resolution_meters'
        ),


      // AOI COVERAGE
      coverage_pct:
        coveragePct,

      coverage_ge_90pct:
        coverage90

    }
  );

}



// ============================================================================
// 06. PROCESS EACH EVENT
//
// IMPORTANT FIX:
// collection.toList(0) is illegal in Earth Engine.
//
// Therefore:
//
// safeCount = max(sceneCount, 1)
//
// If sceneCount = 0:
//     collection.toList(1)
// still returns an empty list,
// but DOES NOT throw a count=0 error.
// ============================================================================


var allSceneTables = [];

var allSummaryFeatures = [];


events.forEach(
  function(evt) {


    // ========================================================================
    // 06A. EVENT-SPECIFIC COLLECTION
    // ========================================================================

    var collection = s1.filterDate(
      evt.audit_start,
      evt.audit_end
    );


    var sceneCount = collection.size();


    // ========================================================================
    // 06B. SAFE LIST CONVERSION
    // ========================================================================

    var safeCount = ee.Number(
      sceneCount
    ).max(
      1
    );


    var imageList = collection.toList(
      safeCount
    );


    // ========================================================================
    // 06C. CONVERT IMAGES TO FEATURES
    // ========================================================================

    var featureList = imageList.map(
      function(item) {

        return imageToFeature(
          ee.Image(item),
          evt
        );

      }
    );


    var table = ee.FeatureCollection(
      featureList
    );


    // ========================================================================
    // 06D. EVENT SUMMARY COUNTS
    // ========================================================================

    var nTotal = table.size();


    var nPre = table
      .filter(
        ee.Filter.eq(
          'timing_class',
          'PRE_EVENT'
        )
      )
      .size();


    var nDuring = table
      .filter(
        ee.Filter.eq(
          'timing_class',
          'DURING_EVENT'
        )
      )
      .size();


    var nPost = table
      .filter(
        ee.Filter.eq(
          'timing_class',
          'POST_EVENT'
        )
      )
      .size();


    var nCoverage90 = table
      .filter(
        ee.Filter.eq(
          'coverage_ge_90pct',
          1
        )
      )
      .size();


    // ========================================================================
    // 06E. ORBIT-DIRECTION SUMMARY
    // ========================================================================

    var nAscending = table
      .filter(
        ee.Filter.eq(
          'orbit_pass',
          'ASCENDING'
        )
      )
      .size();


    var nDescending = table
      .filter(
        ee.Filter.eq(
          'orbit_pass',
          'DESCENDING'
        )
      )
      .size();


    // ========================================================================
    // 06F. EVENT SUMMARY FEATURE
    // ========================================================================

    var summaryFeature = ee.Feature(
      null,
      {

        event_id:
          evt.id,

        reported_start:
          evt.reported_start,

        reported_end:
          evt.reported_end,

        audit_start:
          evt.audit_start,

        audit_end_exclusive:
          evt.audit_end,

        total_scenes:
          nTotal,

        pre_event_scenes:
          nPre,

        during_event_scenes:
          nDuring,

        post_event_scenes:
          nPost,

        ascending_scenes:
          nAscending,

        descending_scenes:
          nDescending,

        scenes_coverage_ge_90pct:
          nCoverage90

      }
    );


    // ========================================================================
    // 06G. STORE RESULTS
    // ========================================================================

    allSceneTables.push(
      table
    );


    allSummaryFeatures.push(
      summaryFeature
    );


    // ========================================================================
    // 06H. CONSOLE OUTPUT
    // ========================================================================

    print(
      '============================================================'
    );


    print(
      'EVENT:',
      evt.id
    );


    print(
      'Reported flood period:',
      evt.reported_start,
      'to',
      evt.reported_end
    );


    print(
      'Sentinel-1 audit window:',
      evt.audit_start,
      'to',
      evt.audit_end,
      '(end exclusive)'
    );


    print(
      'Total matching VV+VH IW scenes:',
      sceneCount
    );


    print(
      'Scene metadata:',
      table.sort(
        'acquisition_time_ms'
      )
    );

  }
);



// ============================================================================
// 07. MERGE ALL EVENT-SPECIFIC SCENE TABLES
// ============================================================================

var mergedScenes = ee.FeatureCollection([]);


allSceneTables.forEach(
  function(fc) {

    mergedScenes = mergedScenes.merge(
      fc
    );

  }
);



// ============================================================================
// 08. CREATE EVENT-LEVEL SUMMARY TABLE
// ============================================================================

var eventSummary = ee.FeatureCollection(
  allSummaryFeatures
);



// ============================================================================
// 09. FINAL CONSOLE OUTPUT
// ============================================================================

print(
  '============================================================'
);


print(
  'ALL SENTINEL-1 SCENES',
  mergedScenes.sort(
    'acquisition_time_ms'
  )
);


print(
  '============================================================'
);


print(
  'EVENT-LEVEL SUMMARY',
  eventSummary
);


print(
  '============================================================'
);


print(
  'TOTAL SCENES ACROSS ALL EVENTS:',
  mergedScenes.size()
);



// ============================================================================
// 10. EXPORT — DETAILED SENTINEL-1 SCENE AUDIT
// ============================================================================

Export.table.toDrive({

  collection:
    mergedScenes,

  description:
    'BUET_B01_S1_Scene_Availability_Audit',

  folder:
    'Chattogram_Flood_Research_Data',

  fileNamePrefix:
    'BUET_B01_S1_Scene_Availability_Audit',

  fileFormat:
    'CSV',

  selectors: [

    'event_id',

    'reported_start',

    'reported_end',

    'audit_start',

    'audit_end_exclusive',

    'image_id',

    'acquisition_datetime_UTC',

    'acquisition_date',

    'timing_class',

    'platform',

    'orbit_pass',

    'relative_orbit',

    'absolute_orbit',

    'instrument_mode',

    'polarizations',

    'resolution_m',

    'coverage_pct',

    'coverage_ge_90pct'

  ]

});



// ============================================================================
// 11. EXPORT — EVENT-LEVEL SUMMARY
// ============================================================================

Export.table.toDrive({

  collection:
    eventSummary,

  description:
    'BUET_B01_S1_Event_Summary',

  folder:
    'Chattogram_Flood_Research_Data',

  fileNamePrefix:
    'BUET_B01_S1_Event_Summary',

  fileFormat:
    'CSV',

  selectors: [

    'event_id',

    'reported_start',

    'reported_end',

    'audit_start',

    'audit_end_exclusive',

    'total_scenes',

    'pre_event_scenes',

    'during_event_scenes',

    'post_event_scenes',

    'ascending_scenes',

    'descending_scenes',

    'scenes_coverage_ge_90pct'

  ]

});



// ============================================================================
// 12. END
// ============================================================================

print(
  'B01 SCRIPT SETUP COMPLETE.'
);

print(
  'Review event counts before launching exports.'
);