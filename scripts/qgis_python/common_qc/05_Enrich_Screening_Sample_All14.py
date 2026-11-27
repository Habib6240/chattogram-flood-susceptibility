from qgis.core import QgsProject
from osgeo import gdal
import csv

INPUT_CSV = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
    r"\Factor_Screening_Results\00_Samples"
    r"\03_Factor_Screening_Sample_1km.csv"
)

OUTPUT_CSV = (
    r"D:\Research\Conference Paper\Fully New Work"
    r"\QGIS & Here Generated File"
    r"\Factor_Screening_Results\00_Samples"
    r"\05_Factor_Screening_Sample_All14.csv"
)

extra_layers = {
    "Aspect": "Aspect_deg_30",
    "LULC": "02_LULC_ESA_WorldCover_2021_Chattogram_30m_STACK",
    "HSG": "04_Soil_HYSOGs250m_Chattogram_30m_STACK"
}

arrays = {}

for short_name, layer_name in extra_layers.items():

    layers = QgsProject.instance().mapLayersByName(layer_name)

    if not layers:
        raise Exception("Layer not found: " + layer_name)

    ds = gdal.Open(layers[0].source())
    arrays[short_name] = ds.GetRasterBand(1).ReadAsArray()


with open(INPUT_CSV, "r", newline="", encoding="utf-8") as fin, \
     open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)

    fieldnames = reader.fieldnames + [
        "Aspect",
        "LULC",
        "HSG"
    ]

    writer = csv.DictWriter(
        fout,
        fieldnames=fieldnames
    )

    writer.writeheader()

    count = 0

    for rec in reader:

        r = int(rec["row"])
        c = int(rec["col"])

        rec["Aspect"] = float(arrays["Aspect"][r, c])
        rec["LULC"] = int(arrays["LULC"][r, c])
        rec["HSG"] = int(arrays["HSG"][r, c])

        writer.writerow(rec)
        count += 1


print("\n=========== ENRICHED SCREENING DATA ===========")
print("Rows exported :", count)
print("Output        :", OUTPUT_CSV)
print("================================================")