// ============================================================================
// BUET ML — B02A SELECTED PRE/FLOOD PAIR COVERAGE AUDIT
// Chattogram, Bangladesh
//
// PURPOSE:
// Check ONLY selected same-track PRE ↔ FLOOD pairs.
// No flood mapping / thresholding yet.
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
// 02. BASE SENTINEL-1
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
// 03. SELECTED CANDIDATE PAIRS
// ============================================================================

var pairs = [

  {
    pair_id: 'E02_2017_R114',
    event_id: 'E02_2017_Jun',
    pre_date: '2017-05-30',
    flood_date: '2017-06-11',
    platform: 'A',
    pass: 'ASCENDING',
    orbit: 114
  },

  {
    pair_id: 'E03_2019_R41',
    event_id: 'E03_2019_Jul',
    pre_date: '2019-07-02',
    flood_date: '2019-07-14',
    platform: 'A',
    pass: 'ASCENDING',
    orbit: 41
  },

  {
    pair_id: 'E03_2019_R114_ALT',
    event_id: 'E03_2019_Jul',
    pre_date: '2019-06-25',
    flood_date: '2019-07-07',
    platform: 'A',
    pass: 'ASCENDING',
    orbit: 114
  },

  {
    pair_id: 'E04_2021_R41',
    event_id: 'E04_2021_Jul',
    pre_date: '2021-07-15',
    flood_date: '2021-07-27',
    platform: 'A',
    pass: 'ASCENDING',
    orbit: 41
  },

  {
    pair_id: 'E04_2021_R77_ALT',
    event_id: 'E04_2021_Jul',
    pre_date: '2021-07-17',
    flood_date: '2021-07-29',
    platform: 'A',
    pass: 'DESCENDING',
    orbit: 77
  },

  {
    pair_id: 'E06_2024_R114',
    event_id: 'E06_2024_Aug',
    pre_date: '2024-08-09',
    flood_date: '2024-08-21',
    platform: 'A',
    pass: 'ASCENDING',
    orbit: 114
  },

  {
    pair_id: 'E07_2026_R77',
    event_id: 'E07_2026_Jul',
    pre_date: '2026-06-28',
    flood_date: '2026-07-10',
    platform: 'D',
    pass: 'DESCENDING',
    orbit: 77
  }

];


// ============================================================================
// 04. GET ONE ACQUISITION GROUP
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
// 05. PROCESS EACH PAIR
// ============================================================================

var outputFeatures = [];


pairs.forEach(function(p) {

  var pre = getGroup(
    p.pre_date,
    p.platform,
    p.pass,
    p.orbit
  );

  var flood = getGroup(
    p.flood_date,
    p.platform,
    p.pass,
    p.orbit
  );


  // --------------------------------------------------------------------------
  // Scene counts
  // --------------------------------------------------------------------------

  var preCount = pre.size();

  var floodCount = flood.size();


  // --------------------------------------------------------------------------
  // Footprint geometries
  // Only 1–2 images per acquisition -> lightweight
  // --------------------------------------------------------------------------

  var preGeom = pre.geometry();

  var floodGeom = flood.geometry();


  // --------------------------------------------------------------------------
  // AOI coverage
  // --------------------------------------------------------------------------

  var preCoverage = preGeom
    .intersection(
      aoi,
      100
    )
    .area(100)
    .divide(aoiArea)
    .multiply(100);


  var floodCoverage = floodGeom
    .intersection(
      aoi,
      100
    )
    .area(100)
    .divide(aoiArea)
    .multiply(100);


  // --------------------------------------------------------------------------
  // Common PRE/FLOOD footprint
  // This is the most important value for change detection.
  // --------------------------------------------------------------------------

  var commonGeom = preGeom
    .intersection(
      floodGeom,
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


  // --------------------------------------------------------------------------
  // PRE ↔ FLOOD interval
  // --------------------------------------------------------------------------

  var gapDays = ee.Date(
    p.flood_date
  ).difference(
    ee.Date(p.pre_date),
    'day'
  );


  // --------------------------------------------------------------------------
  // Quality flag
  // --------------------------------------------------------------------------

  var coveragePass95 = ee.Number(
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

      flood_date:
        p.flood_date,

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

      flood_scene_count:
        floodCount,

      pre_coverage_pct:
        preCoverage,

      flood_coverage_pct:
        floodCoverage,

      common_coverage_pct:
        commonCoverage,

      common_coverage_ge_95:
        coveragePass95

    }
  );


  outputFeatures.push(
    feature
  );


  // Lightweight console output
  print(
    p.pair_id,
    'PRE scenes:',
    preCount,
    'FLOOD scenes:',
    floodCount
  );

});


var result = ee.FeatureCollection(
  outputFeatures
);


print(
  'SELECTED PAIR COVERAGE AUDIT:',
  result
);


// ============================================================================
// 06. EXPORT
// ============================================================================

Export.table.toDrive({

  collection:
    result,

  description:
    'BUET_B02A_Selected_Pair_Coverage_Audit',

  folder:
    'Chattogram_Flood_Research_Data',

  fileNamePrefix:
    'BUET_B02A_Selected_Pair_Coverage_Audit',

  fileFormat:
    'CSV',

  selectors: [

    'pair_id',

    'event_id',

    'pre_date',

    'flood_date',

    'gap_days',

    'platform',

    'orbit_pass',

    'relative_orbit',

    'pre_scene_count',

    'flood_scene_count',

    'pre_coverage_pct',

    'flood_coverage_pct',

    'common_coverage_pct',

    'common_coverage_ge_95'

  ]

});


print(
  'B02A COVERAGE AUDIT READY.'
);