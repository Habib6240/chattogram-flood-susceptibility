// ============================================================================
// BUET ML — B03 MULTI-EVENT SENTINEL-1 FLOOD MASK MASTER
// Study Area: Chattogram, Bangladesh
//
// CORE EVENTS:
// E03 = 2019
// E04 = 2021
// E05 = 2023 (post-peak persistent inundation)
// E07 = 2026
//
// PURPOSE
// -------
// 1. Load matched PRE and flood/observation Sentinel-1 acquisitions.
// 2. Use same platform + pass + relative orbit for each event.
// 3. Apply focal-median speckle reduction.
// 4. Calculate VV + VH backscatter decrease.
// 5. Determine event-specific Otsu threshold.
// 6. Apply conservative dark-water gate.
// 7. Remove permanent water and steep terrain.
// 8. Remove isolated flood pixels.
// 9. Convert to exact 30 m analytical grid.
// 10. Export four flood-label rasters + one summary CSV.
//
// OUTPUT CODING
// -------------
// 1   = high-confidence event flood
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
//
// Reference:
// DEM_Clipped_UTM46N
// EPSG:32646
// 2825 x 4233
// 30 m
// ============================================================================

var TARGET_CRS = 'EPSG:32646';

var XMIN = 335302.9010404373;
var YMIN = 2417725.3250392843;
var XMAX = 420052.9010404373;
var YMAX = 2544715.3250392843;


// GDAL-style affine transform:
// [xScale, xShear, xOrigin,
//  yShear, yScale, yOrigin]

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

// Speckle reduction
var SPECKLE_RADIUS_M = 30;


// Remove steep terrain from high-confidence SAR flood labels
var MAX_SLOPE_DEG = 10;


// Otsu threshold cannot fall below this backscatter decrease
var MIN_CHANGE_DB = 1.5;


// Flood observation should also be sufficiently dark
var FLOOD_VV_MAX_DB = -12;
var FLOOD_VH_MAX_DB = -18;


// Minimum connected flood cluster
var MIN_CONNECTED_PIXELS = 8;


// JRC occurrence >= 90% treated as permanent water
var PERMANENT_WATER_OCCURRENCE = 90;


// Sentinel-1 incidence-angle QC
var MIN_INCIDENCE_ANGLE = 30;
var MAX_INCIDENCE_ANGLE = 45;



// ============================================================================
// 04. FINAL CORE EVENTS
// ============================================================================

var events = [

  // --------------------------------------------------------------------------
  // 2019
  // --------------------------------------------------------------------------

  {
    event_id:
      'E03_2019_Jul',

    label_type:
      'DURING_EVENT',

    pre_date:
      '2019-07-02',

    obs_date:
      '2019-07-14',

    platform:
      'A',

    pass:
      'ASCENDING',

    orbit:
      41
  },


  // --------------------------------------------------------------------------
  // 2021
  // --------------------------------------------------------------------------

  {
    event_id:
      'E04_2021_Jul',

    label_type:
      'DURING_EVENT',

    pre_date:
      '2021-07-17',

    obs_date:
      '2021-07-29',

    platform:
      'A',

    pass:
      'DESCENDING',

    orbit:
      77
  },


  // --------------------------------------------------------------------------
  // 2023
  //
  // Observation occurs after the initially reported peak period.
  // Therefore it is explicitly labelled as persistent post-peak inundation.
  // --------------------------------------------------------------------------

  {
    event_id:
      'E05_2023_Aug',

    label_type:
      'POST_PEAK_PERSISTENT_INUNDATION',

    pre_date:
      '2023-07-31',

    obs_date:
      '2023-08-12',

    platform:
      'A',

    pass:
      'DESCENDING',

    orbit:
      77
  },


  // --------------------------------------------------------------------------
  // 2026
  // --------------------------------------------------------------------------

  {
    event_id:
      'E07_2026_Jul',

    label_type:
      'DURING_EVENT',

    pre_date:
      '2026-06-28',

    obs_date:
      '2026-07-10',

    platform:
      'D',

    pass:
      'DESCENDING',

    orbit:
      77
  }

];



// ============================================================================
// 05. BASE SENTINEL-1 COLLECTION
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
// 06. STATIC QC MASKS
// ============================================================================


// ----------------------------------------------------------------------------
// 06A. Permanent water
// ----------------------------------------------------------------------------

var jrc = ee.Image(
  'JRC/GSW1_4/GlobalSurfaceWater'
);


var permanentWater = jrc
  .select('occurrence')
  .unmask(0)
  .gte(
    PERMANENT_WATER_OCCURRENCE
  );


// ----------------------------------------------------------------------------
// 06B. Terrain slope
// ----------------------------------------------------------------------------

var srtm = ee.Image(
  'USGS/SRTMGL1_003'
);


var terrainSlopeDeg = ee.Terrain
  .slope(srtm);


var acceptableSlope = terrainSlopeDeg
  .lte(
    MAX_SLOPE_DEG
  );



// ============================================================================
// 07. GET ONE MATCHED SENTINEL-1 ACQUISITION
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


  // Adjacent Sentinel-1 tiles from the same acquisition
  // are mosaicked together.
  var mosaic = collection
    .mosaic()
    .clip(aoi);


  // --------------------------------------------------------------------------
  // Incidence-angle QC
  // --------------------------------------------------------------------------

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


  // --------------------------------------------------------------------------
  // VV
  // --------------------------------------------------------------------------

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


  // --------------------------------------------------------------------------
  // VH
  // --------------------------------------------------------------------------

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
// 08. ROBUST OTSU FUNCTION
//
// FIX:
// Previous implementation had N histogram means but N-1 BSS values,
// producing:
//
// Array.sort:
// Array arguments must have same length along one axis.
//
// Here both candidateThresholds and BSS have EXACTLY N-1 elements.
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


  // --------------------------------------------------------------------------
  // Total observations
  // --------------------------------------------------------------------------

  var total = ee.Number(
    counts
      .reduce(
        ee.Reducer.sum(),
        [0]
      )
      .get([0])
  );


  // --------------------------------------------------------------------------
  // Total weighted sum
  // --------------------------------------------------------------------------

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


  // --------------------------------------------------------------------------
  // N bins produce N-1 meaningful split positions
  // --------------------------------------------------------------------------

  var splitIndices = ee.List.sequence(
    1,
    size.subtract(1)
  );


  // --------------------------------------------------------------------------
  // Between-class sum of squares
  // --------------------------------------------------------------------------

  var bssList = splitIndices.map(
    function(i) {

      i = ee.Number(i);


      // ----------------------------------------------------------------------
      // Class A
      // ----------------------------------------------------------------------

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


      // ----------------------------------------------------------------------
      // Class B
      // ----------------------------------------------------------------------

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


      // ----------------------------------------------------------------------
      // Between-class variance
      // ----------------------------------------------------------------------

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


  // --------------------------------------------------------------------------
  // Exactly N-1 BSS scores
  // --------------------------------------------------------------------------

  var bssArray = ee.Array(
    bssList
  );


  // --------------------------------------------------------------------------
  // Exactly N-1 candidate thresholds
  //
  // Using first N-1 histogram bucket means.
  // --------------------------------------------------------------------------

  var candidateThresholds = means.slice(
    0,
    0,
    size.subtract(1)
  );


  // --------------------------------------------------------------------------
  // Select threshold associated with maximum BSS
  // --------------------------------------------------------------------------

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
// 09. BINARY IMAGE -> EXACT 30 m ANALYTICAL GRID
// ============================================================================

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



// ============================================================================
// 10. PROCESS ALL FOUR EVENTS
// ============================================================================

var summaryFeatures = [];



events.forEach(
  function(evt) {


    // ========================================================================
    // 10A. LOAD PRE AND OBSERVATION ACQUISITIONS
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
    // 10B. COMMON OBSERVATION DOMAIN
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
    // 10C. HIGH-CONFIDENCE SAR LABEL DOMAIN
    // ========================================================================

    var labelDomain = commonObserved

      // Stable permanent water removed
      .and(
        permanentWater.not()
      )

      // Steep terrain removed
      .and(
        acceptableSlope
      )

      .clip(
        aoi
      );



    // ========================================================================
    // 10D. BACKSCATTER CHANGE
    //
    // Change = PRE - OBS
    //
    // Flood-induced darkening:
    // OBS becomes more negative -> positive change.
    // ========================================================================

    var deltaVV = pre
      .select('VV')
      .subtract(
        obs.select('VV')
      )
      .rename(
        'deltaVV'
      );


    var deltaVH = pre
      .select('VH')
      .subtract(
        obs.select('VH')
      )
      .rename(
        'deltaVH'
      );


    var changeScore = deltaVV
      .add(
        deltaVH
      )
      .divide(
        2
      )
      .rename(
        'change_score'
      );



    // ========================================================================
    // 10E. HISTOGRAM FOR OTSU
    // ========================================================================

    var histogramResult = changeScore

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



    var histogramDictionary = ee.Dictionary(
      histogramResult.get(
        'change_score'
      )
    );



    // ========================================================================
    // 10F. EVENT-SPECIFIC OTSU THRESHOLD
    // ========================================================================

    var otsuThreshold = otsu(
      histogramDictionary
    );


    // Prevent extremely weak threshold
    var appliedThreshold = otsuThreshold
      .max(
        MIN_CHANGE_DB
      );



    // ========================================================================
    // 10G. ABSOLUTE DARK-WATER GATE
    // ========================================================================

    var obsVV = obs.select(
      'VV'
    );


    var obsVH = obs.select(
      'VH'
    );


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
    // 10H. RAW FLOOD CANDIDATE
    // ========================================================================

    var rawFlood = changeScore
      .gt(
        appliedThreshold
      )
      .and(
        darkWaterGate
      )
      .updateMask(
        labelDomain
      );



    // ========================================================================
    // 10I. CONNECTED-PIXEL CLEANING
    // ========================================================================

    var connectedPixels = rawFlood
      .selfMask()
      .connectedPixelCount(
        100,
        true
      );


    var cleanFlood10m = rawFlood

      .updateMask(
        connectedPixels.gte(
          MIN_CONNECTED_PIXELS
        )
      )

      .unmask(0)

      .and(
        labelDomain
      );



    // ========================================================================
    // 10J. CONVERT FLOOD AND VALID DOMAIN TO 30 m
    // ========================================================================

    var flood30 = binaryTo30mMajority(
      cleanFlood10m
    );


    var valid30 = binaryTo30mMajority(
      labelDomain
    );



    // ========================================================================
    // 10K. FINAL FLOOD LABEL
    //
    // value 1 = flood
    // value 0 = non-flood
    // masked  = invalid SAR label domain
    // ========================================================================

    var finalMask = flood30
      .updateMask(
        valid30
      )
      .rename(
        'flood'
      );



    // ========================================================================
    // 10L. EXPORT CODING
    //
    // 1   flood
    // 0   observed non-flood
    // 255 NoData
    // ========================================================================

    var exportMask = finalMask
      .unmask(
        255,
        false
      )
      .toByte();



    // ========================================================================
    // 10M. FLOOD AREA
    // ========================================================================

    var floodAreaResult = finalMask
      .eq(1)
      .selfMask()
      .multiply(
        ee.Image.pixelArea()
      )
      .rename(
        'flood_area'
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



    var floodAreaM2 = ee.Number(
      ee.Algorithms.If(
        floodAreaResult.contains(
          'flood_area'
        ),
        floodAreaResult.get(
          'flood_area'
        ),
        0
      )
    );


    var floodKm2 = floodAreaM2
      .divide(
        1e6
      );



    // ========================================================================
    // 10N. VALID LABEL AREA
    // ========================================================================

    var validAreaResult = valid30
      .selfMask()
      .multiply(
        ee.Image.pixelArea()
      )
      .rename(
        'valid_area'
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



    var validAreaM2 = ee.Number(
      ee.Algorithms.If(
        validAreaResult.contains(
          'valid_area'
        ),
        validAreaResult.get(
          'valid_area'
        ),
        0
      )
    );


    var validAreaKm2 = validAreaM2
      .divide(
        1e6
      );



    // ========================================================================
    // 10O. FLOODED FRACTION OF VALID LABEL DOMAIN
    // ========================================================================

    var floodFractionPct = ee.Number(
      ee.Algorithms.If(
        validAreaKm2.gt(0),
        floodKm2
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
    // 10P. SUMMARY FEATURE
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

        otsu_threshold_db:
          otsuThreshold,

        applied_threshold_db:
          appliedThreshold,

        minimum_change_db:
          MIN_CHANGE_DB,

        vv_dark_gate_db:
          FLOOD_VV_MAX_DB,

        vh_dark_gate_db:
          FLOOD_VH_MAX_DB,

        min_incidence_angle_deg:
          MIN_INCIDENCE_ANGLE,

        max_incidence_angle_deg:
          MAX_INCIDENCE_ANGLE,

        maximum_slope_deg:
          MAX_SLOPE_DEG,

        minimum_connected_pixels:
          MIN_CONNECTED_PIXELS,

        permanent_water_occurrence_pct:
          PERMANENT_WATER_OCCURRENCE,

        valid_label_area_km2:
          validAreaKm2,

        flood_area_km2:
          floodKm2,

        flood_fraction_of_valid_domain_pct:
          floodFractionPct

      }
    );


    summaryFeatures.push(
      summary
    );



    // ========================================================================
    // 10Q. MAP DISPLAY
    // ========================================================================

    Map.addLayer(
      changeScore
        .updateMask(
          labelDomain
        ),
      {
        min: -3,
        max: 6
      },
      evt.event_id +
      '_ChangeScore',
      false
    );


    Map.addLayer(
      finalMask
        .eq(1)
        .selfMask(),
      {
        min: 1,
        max: 1,
        palette: ['00FFFF']
      },
      evt.event_id +
      '_FloodMask',
      false
    );



    // ========================================================================
    // 10R. EXPORT EVENT FLOOD MASK
    // ========================================================================

    Export.image.toDrive({

      image:
        exportMask,

      description:
        'BUET_B03_' +
        evt.event_id +
        '_FloodMask30m',

      folder:
        'Chattogram_Flood_Research_Data',

      fileNamePrefix:
        'BUET_B03_' +
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
      ' processing graph created.'
    );

  }
);



// ============================================================================
// 11. EVENT SUMMARY
// ============================================================================

var summaryFC = ee.FeatureCollection(
  summaryFeatures
);


print(
  '============================================================'
);


print(
  'B03 EVENT SUMMARY',
  summaryFC
);


print(
  '============================================================'
);



// ============================================================================
// 12. EXPORT EVENT SUMMARY CSV
// ============================================================================

Export.table.toDrive({

  collection:
    summaryFC,

  description:
    'BUET_B03_Flood_Mask_Summary',

  folder:
    'Chattogram_Flood_Research_Data',

  fileNamePrefix:
    'BUET_B03_Flood_Mask_Summary',

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

    'otsu_threshold_db',

    'applied_threshold_db',

    'minimum_change_db',

    'vv_dark_gate_db',

    'vh_dark_gate_db',

    'min_incidence_angle_deg',

    'max_incidence_angle_deg',

    'maximum_slope_deg',

    'minimum_connected_pixels',

    'permanent_water_occurrence_pct',

    'valid_label_area_km2',

    'flood_area_km2',

    'flood_fraction_of_valid_domain_pct'

  ]

});



// ============================================================================
// 13. END
// ============================================================================

print(
  'B03 MASTER SCRIPT READY.'
);

print(
  'Expected Tasks: 4 flood-mask GeoTIFFs + 1 summary CSV.'
);