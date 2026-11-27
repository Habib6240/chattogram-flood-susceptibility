// ============================================================================
// BUET ML — B01C SENTINEL-1 WIDE-WINDOW RAW SCENE AUDIT
// Study area: Chattogram, Bangladesh
//
// PURPOSE
// -------
// 1. Search 60 days before each flood event.
// 2. Include reported flood period.
// 3. Include 14 days after flood event.
// 4. Export RAW scene metadata only.
// 5. Track/date/orbit matching will be done offline from CSV.
//
// MEMORY-SAFE:
// - No mosaics
// - No geometry unions
// - No server-side grouping
// - No repeated aggregate/filter loops
// ============================================================================


// ============================================================================
// 01. AOI
// ============================================================================

var aoi = ee.FeatureCollection(
  'projects/gee-project-505/assets/AOI_Boundary'
).geometry();

Map.centerObject(aoi, 9);


// ============================================================================
// 02. EVENTS
// ============================================================================

var events = [

  {
    id: 'E02_2017_Jun',
    reported_start: '2017-06-11',
    reported_end:   '2017-06-14'
  },

  {
    id: 'E03_2019_Jul',
    reported_start: '2019-07-07',
    reported_end:   '2019-07-15'
  },

  {
    id: 'E04_2021_Jul',
    reported_start: '2021-07-27',
    reported_end:   '2021-07-30'
  },

  {
    id: 'E05_2023_Aug',
    reported_start: '2023-08-05',
    reported_end:   '2023-08-09'
  },

  {
    id: 'E06_2024_Aug',
    reported_start: '2024-08-21',
    reported_end:   '2024-08-24'
  },

  {
    id: 'E07_2026_Jul',
    reported_start: '2026-07-09',
    reported_end:   '2026-07-16'
  }

];


// ============================================================================
// 03. BASE SENTINEL-1 COLLECTION
// ============================================================================

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')

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


// ============================================================================
// 04. IMAGE -> METADATA FEATURE
// ============================================================================

function imageToFeature(image, evt) {

  image = ee.Image(image);

  var acquisition = ee.Date(
    image.get('system:time_start')
  );

  var eventStart = ee.Date(
    evt.reported_start
  );

  var eventEndExclusive = ee.Date(
    evt.reported_end
  ).advance(1, 'day');


  var timingClass = ee.String(

    ee.Algorithms.If(

      acquisition.millis()
        .lt(eventStart.millis()),

      'PRE_EVENT',

      ee.Algorithms.If(

        acquisition.millis()
          .lt(eventEndExclusive.millis()),

        'DURING_EVENT',

        'POST_EVENT'
      )
    )
  );


  var daysFromStart = acquisition.difference(
    eventStart,
    'day'
  );


  return ee.Feature(
    null,
    {

      event_id:
        evt.id,

      reported_start:
        evt.reported_start,

      reported_end:
        evt.reported_end,

      acquisition_date:
        acquisition.format(
          'YYYY-MM-dd'
        ),

      acquisition_datetime_UTC:
        acquisition.format(
          'YYYY-MM-dd HH:mm:ss'
        ),

      timing_class:
        timingClass,

      days_from_reported_start:
        daysFromStart,

      image_id:
        image.get(
          'system:index'
        ),

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

      instrument_mode:
        image.get(
          'instrumentMode'
        ),

      resolution_m:
        image.get(
          'resolution_meters'
        )

    }
  );
}


// ============================================================================
// 05. BUILD RAW TABLE
// ============================================================================

var allTables = [];


events.forEach(function(evt) {

  var eventStart = ee.Date(
    evt.reported_start
  );

  var eventEndExclusive = ee.Date(
    evt.reported_end
  ).advance(1, 'day');


  // 60 days BEFORE flood
  var auditStart = eventStart.advance(
    -60,
    'day'
  );

  // 14 days AFTER reported flood end
  var auditEnd = eventEndExclusive.advance(
    14,
    'day'
  );


  var collection = s1.filterDate(
    auditStart,
    auditEnd
  );


  var sceneCount = collection.size();


  // Safe list conversion
  var safeCount = ee.Number(
    sceneCount
  ).max(1);


  var imageList = collection.toList(
    safeCount
  );


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


  allTables.push(
    table
  );


  // Lightweight console only
  print(
    evt.id + ' total raw scenes:',
    sceneCount
  );

});


// ============================================================================
// 06. MERGE RAW TABLES
// ============================================================================

var merged = ee.FeatureCollection([]);


allTables.forEach(
  function(fc) {

    merged = merged.merge(
      fc
    );

  }
);


// Do NOT print the full collection.
// Do NOT call complex grouping operations.

print(
  'B01C RAW METADATA TABLE CREATED.'
);


// ============================================================================
// 07. EXPORT
// ============================================================================

Export.table.toDrive({

  collection:
    merged,

  description:
    'BUET_B01C_S1_Wide_Window_Raw_Audit',

  folder:
    'Chattogram_Flood_Research_Data',

  fileNamePrefix:
    'BUET_B01C_S1_Wide_Window_Raw_Audit',

  fileFormat:
    'CSV',

  selectors: [

    'event_id',

    'reported_start',

    'reported_end',

    'acquisition_date',

    'acquisition_datetime_UTC',

    'timing_class',

    'days_from_reported_start',

    'image_id',

    'platform',

    'orbit_pass',

    'relative_orbit',

    'absolute_orbit',

    'instrument_mode',

    'resolution_m'

  ]

});


print(
  'EXPORT TASK READY.'
);