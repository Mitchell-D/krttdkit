import krttdkit as kk
from pathlib import Path
from datetime import datetime as dt
from krttdkit import RecipeBook as RB
import numpy as np

pkldir = Path("/home/krttd/uah/22.f/aes572/hw2take2/data/pkls")
figdir = Path("/home/krttd/uah/22.f/aes572/hw2take2/data/figures/ian")
buffdir = Path("/home/krttd/uah/22.f/aes572/hw2take2/data/buffer")

gm = kk.GridManager(buffer_dir=buffdir.as_posix())
red_key = "ian-look3_ref_b02_p5km-goes16"
inf_key = "ian-look3_ref_b05_1km-goes16"
out_key = "ian-look3-t1_ref_b5-b2diff_1km-goes16"

label = "ref"
gm.load_pkl(pkldir.joinpath(red_key+".pkl"))
gm.stride(label, 2)
red = gm.subgrids[label]["am"]

# Load the band 5 infrared pkl and take the channel difference
gm.load_pkl(pkldir.joinpath(inf_key+".pkl"))
diff = red.data[label].isel(y=slice(0,182)) - \
        gm.subgrids[label]["am"].data[label].isel(y=slice(0,182))

# If a use_target_time is True, a static image will be plotted
# using the data closest to the provided target time
use_target_time = True
target_time = dt(year=2022, month=9, day=27, hour=22) # t1
#target_time = dt(year=2022, month=9, day=27, hour=20) # t2

# Define a subgrid of the full pkl to use
get_subgrid = False
lat_range = (24, 26)
lon_range = (-84,-81)

gif_fps = 10

plot_spec = {
    #"title":"ABI Band 2 Reflectance (0.5km)",
    "title":"ABI Band 2-5 Difference Reflectance (1km)",
    "title_size":8,
    "gridline_color":"gray",
    "fig_size":None,
    "borders":True,
    "border_width":0.5,
    "border_color":"black",
    "cb_orient":"vertical",
    "cb_label":"Reflectance",
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

am = gm.subgrids[label]["am"]

# Select a single time
if use_target_time:
    diff = diff.isel(time=am.index_at_time(target_time))

# Select a geographic subgrid of the data
if get_subgrid:
    lat_ind_range, lon_ind_range = am.get_subgrid_indeces(
            lat_range=lat_range, lon_range=lon_range)
    diff = diff.isel(y=slice(*lat_ind_range),
                                 x=slice(*lon_ind_range))

# Animate if array is 3d
ext = (".gif", ".png")[use_target_time]

#"""
print("generating matplotlib image")
# Plot a scalar projection with matplotlib
kk.geo_plot.geo_scalar_plot(
    data=diff,
    #data=gm.subgrids[label]["am"].data[label].data,
    lat=am.data.coords["lat"].data[:,:182],
    lon=am.data.coords["lon"].data[:,:182],
    fig_path=figdir.joinpath(Path(out_key+ext)),
    #fig_path=figdir.joinpath(Path(sat_key+".png")),
    plot_spec=plot_spec,
    animate=not use_target_time,
    )
#"""


#"""
print("generating raw image")
# Plot a "raw" unprojected gif at full resolution

diff.data = kk.GridManager.norm_to_uint8(
        diff.data, resolution=256)
kk.geo_plot.generate_raw_image(
        #RGB=gm.subgrids[label]["am"].data[label].data,
        #RGB=target_arr.data,
        RGB=diff,
        image_path=figdir.joinpath(Path(out_key+"-raw"+ext)),
        #image_path=figdir.joinpath(Path(sat_key+"-raw.png")),
        gif=not use_target_time,
        fps=gif_fps,
        )
#"""

gm.clear()
