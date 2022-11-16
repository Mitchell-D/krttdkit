"""
Helper script for generating truecolor RGBs using already-generated
ABIManager pkls
"""
import datetime as dt
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path

from ABIManager import ABIManager
from geo_plot import geo_rgb_plot, generate_raw_image

plot_spec = {
    "title":"Truecolor RGB",
    "title_size":8,
    "gridline_color":"gray",
    #"fig_size":(16,9),
    #"dpi":1200,
    "borders":True,
    "border_width":0.5,
    "border_color":"black",
    "cb_orient":"vertical",
    "cb_label":"",
    "cb_tick_count":15,
    "cb_levels":80,
    "cb_size":.6,
    "cb_pad":.05,
    "cb_label_format":"{x:.1f}",
    #"cb_cmap":"CMRmap",
    "cb_cmap":"jet",
    "xtick_count":12,
    "xtick_size":8,
    "ytick_count":12,
    "ytick_size":8,
    }


# If single_frame is True, selects the DataArray closest to the target
# time and generates a single image based on it.
single_frame=False
target_time = dt.datetime(year=2018, month=2, day=14, hour=17, minute=2)

#fig_path_gif = Path("/home/krttd/uah/22.f/aes572/hw3/data/figures/laura-conus_airmass.gif")
fig_path_gif = Path("/home/krttd/uah/22.f/aes572/hw3/data/figures/laura-conus_airmass-2km-fullres.gif")

# Get ABIManagers from each band's pickled values
pkl_dir = Path("/home/krttd/uah/22.f/aes572/hw3/data/pkls/laura_conus")
am08 = ABIManager().load_pkl(pkl_dir.joinpath(
    Path("laura-conus_tb_b08_2km.pkl"))) # Band 8
am10 = ABIManager().load_pkl(pkl_dir.joinpath(
    Path("laura-conus_tb_b10_2km.pkl"))) # Band 10
am12 = ABIManager().load_pkl(pkl_dir.joinpath(
    Path("laura-conus_tb_b12_2km.pkl"))) # Band 12
am13 = ABIManager().load_pkl(pkl_dir.joinpath(
    Path("laura-conus_tb_b13_2km.pkl"))) # Band 13

# Temperatures are in differential degrees celsius
R = am08.data["b8ref"].data-am10.data["b10ref"].data
G = am12.data["b12ref"].data-am13.data["b13ref"].data
B = am08.data["b8ref"].data-273.15

# Normalize to recipe value ranges.
R = np.clip((R--26.2)/(0.6--26.2), 0, 1)
G = np.clip((G--42.2)/(6.7--42.2), 0, 1)
B = np.clip((B--64.65)/(-29.25--64.65), 0, 1)

#print(R,G,B)
print(np.amin(R), np.amax(R))
print(np.amin(G), np.amax(G))
print(np.amin(B), np.amax(B))

# Invert Blue channel
B = 1-B

generate_raw_image(
        RGB=np.stack((R,G,B), axis=3),
        image_path=fig_path_gif,
        gif=True,
        fps=20
        )

"""
geo_rgb_plot(
        R=R, G=G, B=B,
        lat=am08.data["lat"], lon=am08.data["lon"],
        fig_path=fig_path_gif,
        #plot_spec=plot_spec,
        animate=True
        )
"""
