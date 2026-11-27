// ============================================================================
// BUET ML — B03U URBAN-AWARE MULTI-EVENT SENTINEL-1 FLOOD MASK MASTER
// Study Area: Chattogram, Bangladesh
//
// TWO FLOOD-DETECTION PATHWAYS:
//
// A. OPEN-WATER / NON-URBAN PATH
//    Flood-induced SAR darkening:
//    PRE - OBS > event-specific threshold
//    + absolute dark-water gate
//
// B. URBAN DOUBLE-BOUNCE PATH
//    Flood-induced VV brightening:
//    OBS_VV - PRE_VV > event-specific threshold
//    + GHSL independent built-up support
//    + VV increase >= VH increase
//
// FINAL FLOOD = OPEN-WATER FLOOD OR URBAN FLOOD
//
// IMPORTANT:
// GHSL is used ONLY as label-generation support.
// It is NOT an ML predictor.
//
// OUTPUT CODING:
// 1   = high-confidence flood
// 0   = observed non-flood
// 255 = invalid / excluded / NoData
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
// 02. EXACT TARGET GRID
// ============================================================================

var TARGET_CRS = 'EPSG:32646';

var XMIN = 335302.9010404373;
var YMIN = 2417725.3250392843;
var XMAX = 420052.9010404373;
var YMAX = 2544715.3250392843;

var TARGET_TRANSFORM = [
  30,
  0,
  XMIN,
  0,
  -30,
  YMAX
];

var EXPORT_REGION = ee.Geometry.Rectangle(
  [
    XMIN,
    YMIN,
    XMAX,
    YMAX
  ],
  TARGET_CRS,
  false
);


// ============================================================================
// 03. PARAMETERS
// ============================================================================

// ---------------------------------------------------------------------------
// Common SAR processing
// ---------------------------------------------------------------------------

var SPECKLE_RADIUS_M = 30;

var MAX_SLOPE_DEG = 10;

var MIN_INCIDENCE_ANGLE = 30;
var MAX_INCIDENCE_ANGLE = 45;

var PERMANENT_WATER_OCCURRENCE = 90;


// ---------------------------------------------------------------------------
// Original open-water flood pathway
// ---------------------------------------------------------------------------

var MIN_OPEN_CHANGE_DB = 1.5;

var FLOOD_VV_MAX_DB = -12;
var FLOOD_VH_MAX_DB = -18;

var MIN_OPEN_CONNECTED_PIXELS = 8;


// ---------------------------------------------------------------------------
// NEW urban double-bounce pathway
// ---------------------------------------------------------------------------

// Conservative minimum increase in VV.
// Event Otsu is used, but cannot fall below this.
var MIN_URBAN_VV_INCREASE_DB = 2.0;

// GHSL 10 m cell area = 100 m2.
// >=20 m2 means at least ~20% built surface support.
var GHSL_BUILT_MIN_M2 = 20;

// Urban flood often forms fragmented patterns.
// Less restrictive than open-water cluster threshold.
var MIN_URBAN_CONNECTED_PIXELS = 4;

// At 30 m, retain an urban signal if at least ~25%
// of contributing source support is positive.
var URBAN_30M_FRACTION_THRESHOLD = 0.25;


// ============================================================================
// 04. FINAL CORE EVENTS
// ============================================================================

var events = [

  {
    event_id: 'E03_2019_Jul',
    label_type: 'DURING_EVENT',
    pre_date: '2019-07-02',
    obs_date: '2019-07-14',
    platform: 'A',
    pass: 'ASCENDING',
    orbit: 41
  },

  {
    event_id: 'E04_2021_Jul',
    label_type: 'DURING_EVENT',
    pre_date: '2021-07-17',
    obs_date: '2021-07-29',
    platform: 'A',
    pass: 'DESCENDING',
    orbit: 77
  },

  {
    event_id: 'E05_2023_Aug',
    label_type: 'POST_PEAK_PERSISTENT_INUNDATION',
    pre_date: '2023-07-31',
    obs_date: '2023-08-12',
    platform: 'A',
    pass: 'DESCENDING',
    orbit: 77
  },

  {
    event_id: 'E07_2026_Jul',
    label_type: 'DURING_EVENT',
    pre_date: '2026-06-28',
    obs_date: '2026-07-10',
    platform: 'D',
    pass: 'DESCENDING',
    orbit: 77
  }

];


// ============================================================================
// 05. SENTINEL-1 COLLECTION
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
  'Base Sentinel-1 collection ready.'
);


// ============================================================================
// 06. STATIC QC / ANCILLARY MASKS
// ============================================================================


// ---------------------------------------------------------------------------
// 06A. Permanent water
// ---------------------------------------------------------------------------

var jrc = ee.Image(
  'JRC/GSW1_4/GlobalSurfaceWater'
);

var permanentWater = jrc
  .select('occurrence')
  .unmask(0)
  .gte(
    PERMANENT_WATER_OCCURRENCE
  );


// ---------------------------------------------------------------------------
// 06B. Terrain slope
// ---------------------------------------------------------------------------

var srtm = ee.Image(
  'USGS/SRTMGL1_003'
);

var terrainSlopeDeg = ee.Terrain
  .slope(srtm);

var acceptableSlope = terrainSlopeDeg
  .lte(
    MAX_SLOPE_DEG
  );


// ---------------------------------------------------------------------------
// 06C. Independent GHSL built-up support
//
// GHSL 2018, 10 m.
// NOT the ESA WorldCover predictor.
// ---------------------------------------------------------------------------

var ghslBuilt2018 = ee.Image(

  ee.ImageCollection(
    'JRC/GHSL/P2023A/GHS_BUILT_S_10m'
  )

  .filterDate(
    '2018-01-01',
    '2019-01-01'
  )

  .first()

)

.select(
  'built_surface'
)

.clip(aoi);


var urbanSupport = ghslBuilt2018
  .gte(
    GHSL_BUILT_MIN_M2
  )
  .rename(
    'urban_support'
  );


Map.addLayer(
  urbanSupport.selfMask(),
  {
    palette: ['FF00FF']
  },
  'GHSL urban support',
  false
);


// ============================================================================
// 07. GET MATCHED SENTINEL-1 ACQUISITION
// ============================================================================

function getAcquisition(
  evt,
  dateString
) {

  var start = ee.Date(
    dateString
  );

  var end = start.advance(
    1,
    'day'
  );


  var collection = s1

    .filterDate(
      start,
      end
    )

    .filter(
      ee.Filter.eq(
        'platform_number',
        evt.platform
      )
    )

    .filter(
      ee.Filter.eq(
        'orbitProperties_pass',
        evt.pass
      )
    )

    .filter(
      ee.Filter.eq(
        'relativeOrbitNumber_start',
        evt.orbit
      )
    );


  var mosaic = collection
    .mosaic()
    .clip(aoi);


  var angle = mosaic.select(
    'angle'
  );


  var angleMask = angle
    .gt(
      MIN_INCIDENCE_ANGLE
    )
    .and(
      angle.lt(
        MAX_INCIDENCE_ANGLE
      )
    );


  var vv = mosaic
    .select('VV')
    .updateMask(
      angleMask
    )
    .focalMedian(
      SPECKLE_RADIUS_M,
      'circle',
      'meters'
    )
    .rename(
      'VV'
    );


  var vh = mosaic
    .select('VH')
    .updateMask(
      angleMask
    )
    .focalMedian(
      SPECKLE_RADIUS_M,
      'circle',
      'meters'
    )
    .rename(
      'VH'
    );


  return vv
    .addBands(vh)
    .set(
      'scene_count',
      collection.size()
    );

}


// ============================================================================
// 08. ROBUST OTSU
// ============================================================================

function otsu(histogram) {

  histogram = ee.Dictionary(
    histogram
  );


  var counts = ee.Array(
    histogram.get(
      'histogram'
    )
  );


  var means = ee.Array(
    histogram.get(
      'bucketMeans'
    )
  );


  var size = ee.Number(
    means
      .length()
      .get([0])
  );


  var total = ee.Number(
    counts
      .reduce(
        ee.Reducer.sum(),
        [0]
      )
      .get([0])
  );


  var weightedSum = ee.Number(
    means
      .multiply(
        counts
      )
      .reduce(
        ee.Reducer.sum(),
        [0]
      )
      .get([0])
  );


  var globalMean = weightedSum
    .divide(
      total
    );


  var splitIndices = ee.List.sequence(
    1,
    size.subtract(1)
  );


  var bssList = splitIndices.map(
    function(i) {

      i = ee.Number(i);


      var aCounts = counts.slice(
        0,
        0,
        i
      );


      var aCount = ee.Number(
        aCounts
          .reduce(
            ee.Reducer.sum(),
            [0]
          )
          .get([0])
      );


      var aMeans = means.slice(
        0,
        0,
        i
      );


      var aWeightedSum = ee.Number(
        aMeans
          .multiply(
            aCounts
          )
          .reduce(
            ee.Reducer.sum(),
            [0]
          )
          .get([0])
      );


      var aMean = aWeightedSum
        .divide(
          aCount
        );


      var bCount = total
        .subtract(
          aCount
        );


      var bWeightedSum = weightedSum
        .subtract(
          aWeightedSum
        );


      var bMean = bWeightedSum
        .divide(
          bCount
        );


      var classA = aCount
        .multiply(
          aMean
            .subtract(
              globalMean
            )
            .pow(2)
        );


      var classB = bCount
        .multiply(
          bMean
            .subtract(
              globalMean
            )
            .pow(2)
        );


      return classA.add(
        classB
      );

    }
  );


  var bssArray = ee.Array(
    bssList
  );


  var candidateThresholds = means.slice(
    0,
    0,
    size.subtract(1)
  );


  var threshold = candidateThresholds
    .sort(
      bssArray
    )
    .get([-1]);


  return ee.Number(
    threshold
  );

}


// ============================================================================
// 09. BINARY -> EXACT 30 m GRID
// ============================================================================


// ---------------------------------------------------------------------------
// Open-water majority conversion
// ---------------------------------------------------------------------------

function binaryTo30mMajority(
  binaryImage
) {

  return binaryImage

    .unmask(0)

    .toFloat()

    .reduceResolution({

      reducer:
        ee.Reducer.mean(),

      maxPixels:
        1024

    })

    .reproject({

      crs:
        TARGET_CRS,

      crsTransform:
        TARGET_TRANSFORM

    })

    .gte(
      0.5
    );

}


// ---------------------------------------------------------------------------
// Urban fractional conversion
// ---------------------------------------------------------------------------

function binaryTo30mUrban(
  binaryImage
) {

  return binaryImage

    .unmask(0)

    .toFloat()

    .reduceResolution({

      reducer:
        ee.Reducer.mean(),

      maxPixels:
        1024

    })

    .reproject({

      crs:
        TARGET_CRS,

      crsTransform:
        TARGET_TRANSFORM

    })

    .gte(
      URBAN_30M_FRACTION_THRESHOLD
    );

}


// ============================================================================
// 10. AREA HELPER
// ============================================================================

function areaKm2(
  binaryImage,
  bandName
) {

  var result = binaryImage

    .eq(1)

    .selfMask()

    .multiply(
      ee.Image.pixelArea()
    )

    .rename(
      bandName
    )

    .reduceRegion({

      reducer:
        ee.Reducer.sum(),

      geometry:
        aoi,

      crs:
        TARGET_CRS,

      scale:
        30,

      maxPixels:
        1e9,

      tileScale:
        4

    });


  var areaM2 = ee.Number(

    ee.Algorithms.If(

      result.contains(
        bandName
      ),

      result.get(
        bandName
      ),

      0

    )

  );


  return areaM2.divide(
    1e6
  );

}


// ============================================================================
// 11. PROCESS FOUR EVENTS
// ============================================================================

var summaryFeatures = [];


events.forEach(
  function(evt) {


    // ========================================================================
    // 11A. LOAD ACQUISITIONS
    // ========================================================================

    var pre = getAcquisition(
      evt,
      evt.pre_date
    );


    var obs = getAcquisition(
      evt,
      evt.obs_date
    );


    var preSceneCount = ee.Number(
      pre.get(
        'scene_count'
      )
    );


    var obsSceneCount = ee.Number(
      obs.get(
        'scene_count'
      )
    );


    print(
      evt.event_id +
      ' PRE scenes:',
      preSceneCount
    );


    print(
      evt.event_id +
      ' OBS scenes:',
      obsSceneCount
    );


    // ========================================================================
    // 11B. COMMON OBSERVATION DOMAIN
    // ========================================================================

    var commonObserved = pre
      .select('VV')
      .mask()

      .and(
        obs
          .select('VV')
          .mask()
      )

      .and(
        pre
          .select('VH')
          .mask()
      )

      .and(
        obs
          .select('VH')
          .mask()
      );


    // ========================================================================
    // 11C. LABEL DOMAIN
    // ========================================================================

    var labelDomain = commonObserved

      .and(
        permanentWater.not()
      )

      .and(
        acceptableSlope
      )

      .clip(aoi);


    // ========================================================================
    // 11D. BACKSCATTER VALUES
    // ========================================================================

    var preVV = pre.select(
      'VV'
    );


    var preVH = pre.select(
      'VH'
    );


    var obsVV = obs.select(
      'VV'
    );


    var obsVH = obs.select(
      'VH'
    );


    // ========================================================================
    // 11E. ORIGINAL OPEN-WATER CHANGE
    //
    // PRE - OBS
    //
    // Positive = darkening after flood
    // ========================================================================

    var deltaVVDecrease = preVV
      .subtract(
        obsVV
      )
      .rename(
        'deltaVV_decrease'
      );


    var deltaVHDecrease = preVH
      .subtract(
        obsVH
      )
      .rename(
        'deltaVH_decrease'
      );


    var openChangeScore = deltaVVDecrease
      .add(
        deltaVHDecrease
      )
      .divide(
        2
      )
      .rename(
        'open_change_score'
      );


    // ========================================================================
    // 11F. OPEN-WATER OTSU
    // ========================================================================

    var openHistogramResult = openChangeScore

      .updateMask(
        labelDomain
      )

      .reduceRegion({

        reducer:
          ee.Reducer.histogram({
            maxBuckets: 128,
            minBucketWidth: 0.05
          }),

        geometry:
          aoi,

        scale:
          60,

        bestEffort:
          true,

        maxPixels:
          1e9,

        tileScale:
          4

      });


    var openHistogramDictionary = ee.Dictionary(
      openHistogramResult.get(
        'open_change_score'
      )
    );


    var openOtsuThreshold = otsu(
      openHistogramDictionary
    );


    var openAppliedThreshold = openOtsuThreshold
      .max(
        MIN_OPEN_CHANGE_DB
      );


    // ========================================================================
    // 11G. ORIGINAL DARK-WATER GATE
    // ========================================================================

    var darkWaterGate = obsVV
      .lt(
        FLOOD_VV_MAX_DB
      )

      .and(
        obsVH.lt(
          FLOOD_VH_MAX_DB
        )
      );


    // ========================================================================
    // 11H. OPEN-WATER FLOOD CANDIDATE
    // ========================================================================

    var rawOpenFlood = openChangeScore

      .gt(
        openAppliedThreshold
      )

      .and(
        darkWaterGate
      )

      .updateMask(
        labelDomain
      );


    var openConnected = rawOpenFlood
      .selfMask()
      .connectedPixelCount(
        100,
        true
      );


    var cleanOpenFlood = rawOpenFlood

      .updateMask(
        openConnected.gte(
          MIN_OPEN_CONNECTED_PIXELS
        )
      )

      .unmask(0)

      .and(
        labelDomain
      );


    // ========================================================================
    // 11I. NEW URBAN VV INCREASE
    //
    // OBS - PRE
    //
    // Positive = flood-associated brightening / double-bounce candidate
    // ========================================================================

    var vvIncrease = obsVV
      .subtract(
        preVV
      )
      .rename(
        'urban_vv_increase'
      );


    var vhIncrease = obsVH
      .subtract(
        preVH
      )
      .rename(
        'urban_vh_increase'
      );


    // Independent urban label-support domain
    var urbanLabelDomain = labelDomain
      .and(
        urbanSupport
      );


    // ========================================================================
    // 11J. URBAN VV-INCREASE OTSU
    // ========================================================================

    var urbanHistogramResult = vvIncrease

      .updateMask(
        urbanLabelDomain
      )

      .reduceRegion({

        reducer:
          ee.Reducer.histogram({
            maxBuckets: 128,
            minBucketWidth: 0.05
          }),

        geometry:
          aoi,

        scale:
          30,

        bestEffort:
          true,

        maxPixels:
          1e9,

        tileScale:
          4

      });


    var urbanHistogramDictionary = ee.Dictionary(
      urbanHistogramResult.get(
        'urban_vv_increase'
      )
    );


    var urbanOtsuThreshold = otsu(
      urbanHistogramDictionary
    );


    var urbanAppliedThreshold = urbanOtsuThreshold
      .max(
        MIN_URBAN_VV_INCREASE_DB
      );


    // ========================================================================
    // 11K. URBAN POLARISATION-CONTRAST SUPPORT
    //
    // Urban double-bounce response is expected to be at least as strong
    // in co-polarized VV as in cross-polarized VH.
    // ========================================================================

    var urbanPolarisationSupport = vvIncrease
      .gte(
        vhIncrease
      );


    // ========================================================================
    // 11L. RAW URBAN FLOOD
    // ========================================================================

    var rawUrbanFlood = vvIncrease

      .gt(
        urbanAppliedThreshold
      )

      .and(
        urbanPolarisationSupport
      )

      .and(
        urbanSupport
      )

      .updateMask(
        labelDomain
      );


    // ========================================================================
    // 11M. URBAN CONNECTED-PIXEL CLEANING
    // ========================================================================

    var urbanConnected = rawUrbanFlood
      .selfMask()
      .connectedPixelCount(
        100,
        true
      );


    var cleanUrbanFlood = rawUrbanFlood

      .updateMask(
        urbanConnected.gte(
          MIN_URBAN_CONNECTED_PIXELS
        )
      )

      .unmask(0)

      .and(
        labelDomain
      );


    // ========================================================================
    // 11N. 30 m CONVERSION
    // ========================================================================

    var openFlood30 = binaryTo30mMajority(
      cleanOpenFlood
    );


    var urbanFlood30 = binaryTo30mUrban(
      cleanUrbanFlood
    );


    var valid30 = binaryTo30mMajority(
      labelDomain
    );


    // ========================================================================
    // 11O. COMBINED FLOOD
    // ========================================================================

    var combinedFlood30 = openFlood30
      .or(
        urbanFlood30
      );


    // Urban pixels newly added relative to original B03 pathway
    var urbanAdded30 = urbanFlood30
      .and(
        openFlood30.not()
      );


    // ========================================================================
    // 11P. FINAL LABEL
    // ========================================================================

    var finalMask = combinedFlood30

      .updateMask(
        valid30
      )

      .rename(
        'flood'
      );


    var exportMask = finalMask

      .unmask(
        255,
        false
      )

      .toByte();


    // ========================================================================
    // 11Q. AREAS
    // ========================================================================

    var openFloodKm2 = areaKm2(
      openFlood30.updateMask(
        valid30
      ),
      'open_flood_area'
    );


    var urbanFloodKm2 = areaKm2(
      urbanFlood30.updateMask(
        valid30
      ),
      'urban_flood_area'
    );


    var urbanAddedKm2 = areaKm2(
      urbanAdded30.updateMask(
        valid30
      ),
      'urban_added_area'
    );


    var combinedFloodKm2 = areaKm2(
      combinedFlood30.updateMask(
        valid30
      ),
      'combined_flood_area'
    );


    var validAreaKm2 = areaKm2(
      valid30,
      'valid_area'
    );


    var floodFractionPct = ee.Number(

      ee.Algorithms.If(

        validAreaKm2.gt(0),

        combinedFloodKm2
          .divide(
            validAreaKm2
          )
          .multiply(
            100
          ),

        0

      )

    );


    // ========================================================================
    // 11R. SUMMARY
    // ========================================================================

    var summary = ee.Feature(
      null,
      {

        event_id:
          evt.event_id,

        label_type:
          evt.label_type,

        pre_date:
          evt.pre_date,

        observation_date:
          evt.obs_date,

        platform:
          evt.platform,

        orbit_pass:
          evt.pass,

        relative_orbit:
          evt.orbit,

        gap_days:
          ee.Date(
            evt.obs_date
          )
          .difference(
            ee.Date(
              evt.pre_date
            ),
            'day'
          ),

        pre_scene_count:
          preSceneCount,

        observation_scene_count:
          obsSceneCount,

        open_otsu_threshold_db:
          openOtsuThreshold,

        open_applied_threshold_db:
          openAppliedThreshold,

        urban_otsu_vv_increase_db:
          urbanOtsuThreshold,

        urban_applied_vv_increase_db:
          urbanAppliedThreshold,

        minimum_open_change_db:
          MIN_OPEN_CHANGE_DB,

        minimum_urban_vv_increase_db:
          MIN_URBAN_VV_INCREASE_DB,

        ghsl_built_min_m2:
          GHSL_BUILT_MIN_M2,

        min_open_connected_pixels:
          MIN_OPEN_CONNECTED_PIXELS,

        min_urban_connected_pixels:
          MIN_URBAN_CONNECTED_PIXELS,

        urban_30m_fraction_threshold:
          URBAN_30M_FRACTION_THRESHOLD,

        vv_dark_gate_db:
          FLOOD_VV_MAX_DB,

        vh_dark_gate_db:
          FLOOD_VH_MAX_DB,

        minimum_incidence_angle_deg:
          MIN_INCIDENCE_ANGLE,

        maximum_incidence_angle_deg:
          MAX_INCIDENCE_ANGLE,

        maximum_slope_deg:
          MAX_SLOPE_DEG,

        permanent_water_occurrence_pct:
          PERMANENT_WATER_OCCURRENCE,

        valid_label_area_km2:
          validAreaKm2,

        original_open_flood_area_km2:
          openFloodKm2,

        urban_candidate_area_km2:
          urbanFloodKm2,

        urban_newly_added_area_km2:
          urbanAddedKm2,

        combined_flood_area_km2:
          combinedFloodKm2,

        combined_flood_fraction_pct:
          floodFractionPct

      }
    );


    summaryFeatures.push(
      summary
    );


    // ========================================================================
    // 11S. MAP DISPLAY
    // ========================================================================

    Map.addLayer(
      openFlood30
        .eq(1)
        .selfMask(),
      {
        palette: ['00FFFF']
      },
      evt.event_id +
      '_OpenFlood',
      false
    );


    Map.addLayer(
      urbanAdded30
        .eq(1)
        .selfMask(),
      {
        palette: ['FF00FF']
      },
      evt.event_id +
      '_UrbanAdded',
      false
    );


    Map.addLayer(
      finalMask
        .eq(1)
        .selfMask(),
      {
        palette: ['0000FF']
      },
      evt.event_id +
      '_CombinedFlood',
      false
    );


    // ========================================================================
    // 11T. EXPORT COMBINED EVENT MASK
    // ========================================================================

    Export.image.toDrive({

      image:
        exportMask,

      description:
        'BUET_B03U_' +
        evt.event_id +
        '_FloodMask30m',

      folder:
        'Chattogram_Flood_Research_Data',

      fileNamePrefix:
        'BUET_B03U_' +
        evt.event_id +
        '_FloodMask30m',

      region:
        EXPORT_REGION,

      crs:
        TARGET_CRS,

      crsTransform:
        TARGET_TRANSFORM,

      maxPixels:
        1e9,

      fileFormat:
        'GeoTIFF',

      formatOptions: {

        cloudOptimized:
          true,

        noData:
          255

      }

    });


    print(
      evt.event_id +
      ' B03U processing graph created.'
    );

  }
);


// ============================================================================
// 12. SUMMARY EXPORT
// ============================================================================

var summaryFC = ee.FeatureCollection(
  summaryFeatures
);


print(
  '============================================================'
);

print(
  'B03U URBAN-AWARE EVENT SUMMARY',
  summaryFC
);

print(
  '============================================================'
);


Export.table.toDrive({

  collection:
    summaryFC,

  description:
    'BUET_B03U_Flood_Mask_Summary',

  folder:
    'Chattogram_Flood_Research_Data',

  fileNamePrefix:
    'BUET_B03U_Flood_Mask_Summary',

  fileFormat:
    'CSV',

  selectors: [

    'event_id',

    'label_type',

    'pre_date',

    'observation_date',

    'platform',

    'orbit_pass',

    'relative_orbit',

    'gap_days',

    'pre_scene_count',

    'observation_scene_count',

    'open_otsu_threshold_db',

    'open_applied_threshold_db',

    'urban_otsu_vv_increase_db',

    'urban_applied_vv_increase_db',

    'minimum_open_change_db',

    'minimum_urban_vv_increase_db',

    'ghsl_built_min_m2',

    'min_open_connected_pixels',

    'min_urban_connected_pixels',

    'urban_30m_fraction_threshold',

    'vv_dark_gate_db',

    'vh_dark_gate_db',

    'minimum_incidence_angle_deg',

    'maximum_incidence_angle_deg',

    'maximum_slope_deg',

    'permanent_water_occurrence_pct',

    'valid_label_area_km2',

    'original_open_flood_area_km2',

    'urban_candidate_area_km2',

    'urban_newly_added_area_km2',

    'combined_flood_area_km2',

    'combined_flood_fraction_pct'

  ]

});


print(
  'B03U MASTER SCRIPT READY.'
);

print(
  'Expected Tasks: 4 B03U GeoTIFFs + 1 B03U summary CSV.'
);