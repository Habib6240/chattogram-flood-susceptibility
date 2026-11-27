from qgis.core import QgsProject, QgsRasterLayer

GROUP_NAME = "07_FINAL_FACTORS_PreScreening"
REFERENCE_NAME = "DEM_Clipped_UTM46N"

EXT_TOL = 0.01      # metres
PIXEL_TOL = 1e-8    # metres

root = QgsProject.instance().layerTreeRoot()
group = root.findGroup(GROUP_NAME)

if group is None:
    print("ERROR: Group not found:", GROUP_NAME)

else:
    refs = QgsProject.instance().mapLayersByName(REFERENCE_NAME)

    if not refs:
        print("ERROR: Reference raster not found:", REFERENCE_NAME)

    else:
        ref = refs[0]
        re = ref.extent()

        print("\n================ REFERENCE GRID ================")
        print("Layer      :", ref.name())
        print("CRS        :", ref.crs().authid())
        print("Width      :", ref.width())
        print("Height     :", ref.height())
        print("Pixel X    :", ref.rasterUnitsPerPixelX())
        print("Pixel Y    :", ref.rasterUnitsPerPixelY())
        print("Extent     :", re.xMinimum(), re.yMinimum(),
              re.xMaximum(), re.yMaximum())

        print("\n================ FINAL FACTOR AUDIT ================")

        count = 0
        ok_count = 0

        for node in group.findLayers():

            lyr = node.layer()

            if not isinstance(lyr, QgsRasterLayer):
                continue

            count += 1
            e = lyr.extent()

            same_crs = lyr.crs().authid() == ref.crs().authid()
            same_width = lyr.width() == ref.width()
            same_height = lyr.height() == ref.height()

            same_px = (
                abs(lyr.rasterUnitsPerPixelX() -
                    ref.rasterUnitsPerPixelX()) < PIXEL_TOL
                and
                abs(lyr.rasterUnitsPerPixelY() -
                    ref.rasterUnitsPerPixelY()) < PIXEL_TOL
            )

            dxmin = abs(e.xMinimum() - re.xMinimum())
            dymin = abs(e.yMinimum() - re.yMinimum())
            dxmax = abs(e.xMaximum() - re.xMaximum())
            dymax = abs(e.yMaximum() - re.yMaximum())

            same_extent = (
                dxmin < EXT_TOL
                and dymin < EXT_TOL
                and dxmax < EXT_TOL
                and dymax < EXT_TOL
            )

            grid_ok = (
                same_crs and
                same_width and
                same_height and
                same_px and
                same_extent
            )

            if grid_ok:
                ok_count += 1

            provider = lyr.dataProvider()

            if provider.sourceHasNoDataValue(1):
                nodata = provider.sourceNoDataValue(1)
            else:
                nodata = "None"

            print("\n--------------------------------------------")
            print("Layer      :", lyr.name())
            print("CRS        :", lyr.crs().authid())
            print("Size       :", lyr.width(), "x", lyr.height())
            print("Pixel      :", lyr.rasterUnitsPerPixelX(),
                  "x", lyr.rasterUnitsPerPixelY())
            print("NoData     :", nodata)

            print("Extent Δm  :",
                  round(dxmin, 8),
                  round(dymin, 8),
                  round(dxmax, 8),
                  round(dymax, 8))

            print("GRID CHECK :", "OK" if grid_ok else "MISMATCH")

        print("\n==============================================")
        print("Raster count :", count)
        print("Grid OK      :", ok_count)
        print("Expected     : 14")
        print("FINAL STATUS :",
              "PASS" if count == 14 and ok_count == 14 else "CHECK REQUIRED")
        print("==============================================")