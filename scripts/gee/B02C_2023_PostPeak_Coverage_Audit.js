// ============================================================================
// BUET ML — B02C 2023 POST-PEAK SENTINEL-1 COVERAGE AUDIT
// Purpose:
// Test 2023 matched PRE -> post-peak acquisitions as possible
// persistent-inundation observations.
// ============================================================================


// ============================================================================
// 01. AOI
// ============================================================================

var aoi = ee.FeatureCollection(
  'projects/gee-project-505/assets/AOI_Boundary'
).geometry();

Map.centerObject(aoi, 9);

var aoiArea = aoi.area(100);


// ============================================================================
// 02. BASE S1 COLLECTION
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
// 03. 2023 CANDIDATE PAIRS
// ============================================================================

var pairs = [

  {
    pair_id: 'E05_2023_R41',
    event_id: 'E05_2023_Aug',
    pre_date: '2023-07-29',
    observation_date: '2023-08-10',
    platform: 'A',
    pass: 'ASCENDING',
    orbit: 41
  },

  {
    pair_id: 'E05_2023_R77',
    event_id: 'E05_2023_Aug',
    pre_date: '2023-07-31',
    observation_date: '2023-08-12',
    platform: 'A',
    pass: 'DESCENDING',
    orbit: 77
  },

  {
    pair_id: 'E05_2023_R114',
    event_id: 'E05_2023_Aug',
    pre_date: '2023-08-03',
    observation_date: '2023-08-15',
    platform: 'A',
    pass: 'ASCENDING',
    orbit: 114
  }

];


// ============================================================================
// 04. GET ACQUISITION GROUP
// ============================================================================

function getGroup(dateString, platform, pass, orbit) {

  var start = ee.Date(dateString);

  var end = start.advance(
    1,
    'day'
  );

  return s1

    .filterDate(
      start,
      end
    )

    .filter(
      ee.Filter.eq(
        'platform_number',
        platform
      )
    )

    .filter(
      ee.Filter.eq(
        'orbitProperties_pass',
        pass
      )
    )

    .filter(
      ee.Filter.eq(
        'relativeOrbitNumber_start',
        orbit
      )
    );
}


// ============================================================================
// 05. PROCESS PAIRS
// ============================================================================

var output = [];


pairs.forEach(function(p) {

  var pre = getGroup(
    p.pre_date,
    p.platform,
    p.pass,
    p.orbit
  );

  var obs = getGroup(
    p.observation_date,
    p.platform,
    p.pass,
    p.orbit
  );


  var preCount = pre.size();

  var obsCount = obs.size();


  var preGeom = pre.geometry();

  var obsGeom = obs.geometry();


  var preCoverage = preGeom
    .intersection(
      aoi,
      100
    )
    .area(100)
    .divide(aoiArea)
    .multiply(100);


  var obsCoverage = obsGeom
    .intersection(
      aoi,
      100
    )
    .area(100)
    .divide(aoiArea)
    .multiply(100);


  var commonGeom = preGeom
    .intersection(
      obsGeom,
      100
    )
    .intersection(
      aoi,
      100
    );


  var commonCoverage = commonGeom
    .area(100)
    .divide(aoiArea)
    .multiply(100);


  var gapDays = ee.Date(
    p.observation_date
  ).difference(
    ee.Date(p.pre_date),
    'day'
  );


  var coverage95 = ee.Number(
    ee.Algorithms.If(
      commonCoverage.gte(95),
      1,
      0
    )
  );


  var feature = ee.Feature(
    null,
    {

      pair_id:
        p.pair_id,

      event_id:
        p.event_id,

      pre_date:
        p.pre_date,

      observation_date:
        p.observation_date,

      observation_type:
        'POST_PEAK_PERSISTENT_INUNDATION_CANDIDATE',

      gap_days:
        gapDays,

      platform:
        p.platform,

      orbit_pass:
        p.pass,

      relative_orbit:
        p.orbit,

      pre_scene_count:
        preCount,

      observation_scene_count:
        obsCount,

      pre_coverage_pct:
        preCoverage,

      observation_coverage_pct:
        obsCoverage,

      common_coverage_pct:
        commonCoverage,

      common_coverage_ge_95:
        coverage95

    }
  );


  output.push(
    feature
  );


  print(
    p.pair_id,
    'PRE scenes:',
    preCount,
    'OBS scenes:',
    obsCount
  );

});


var result = ee.FeatureCollection(
  output
);


print(
  '2023 POST-PEAK COVERAGE AUDIT:',
  result
);


// ============================================================================
// 06. EXPORT
// ============================================================================

Export.table.toDrive({

  collection:
    result,

  description:
    'BUET_B02C_2023_PostPeak_Coverage_Audit',

  folder:
    'Chattogram_Flood_Research_Data',

  fileNamePrefix:
    'BUET_B02C_2023_PostPeak_Coverage_Audit',

  fileFormat:
    'CSV',

  selectors: [

    'pair_id',

    'event_id',

    'pre_date',

    'observation_date',

    'observation_type',

    'gap_days',

    'platform',

    'orbit_pass',

    'relative_orbit',

    'pre_scene_count',

    'observation_scene_count',

    'pre_coverage_pct',

    'observation_coverage_pct',

    'common_coverage_pct',

    'common_coverage_ge_95'

  ]

});


print(
  'B02C 2023 COVERAGE AUDIT READY.'
);