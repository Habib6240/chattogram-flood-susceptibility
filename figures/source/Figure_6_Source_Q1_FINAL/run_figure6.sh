#!/bin/sh
# INPUTS = folder holding the six input files listed in README.txt (not redistributed). Python 3 with numpy, pandas, opencv-python (reads LZW/Deflate GeoTIFF), tifffile, matplotlib, Pillow; font Liberation Sans.
IN=${INPUTS:-./inputs}
python3 -I make_figure6_q1.py --inputs $IN --out output --dpi 600          # add --classes supplied to draw panel (c) from the supplied common-score class rasters instead (not the locked system)
python3 -I qc_figure6.py $IN output
python3 -I cvd_check_fig6.py output/Figure_6_CrossFamily_Spatial_Comparison_Q1_FINAL.png output/qa_renders
