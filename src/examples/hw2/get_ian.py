import krttdkit as kk
from pathlib import Path
from datetime import datetime as dt
import numpy as np

""" Script for generating imagery in bands 2, 5, and 13 """

pkldir = Path("/home/krttd/uah/22.f/aes572/hw2take2/data/pkls")
#figdir = Path("/home/krttd/uah/22.f/aes572/hw2take2/data/figures/ian")
#figdir = Path("/home/krttd/uah/22.f/aes572/hw2take2/data/figures/ian/look5")
figdir = Path("/home/krttd/uah/22.f/aes572/hw2take2/data/figures/ian/restricted")
buffdir = Path("/home/krttd/uah/22.f/aes572/hw2take2/data/buffer")

gm = kk.GridManager(buffer_dir=buffdir.as_posix())

use_target_time = False
target_time = dt(year=2022, month=9, day=27, hour=22) # t1
gif_fps = 10 # Only applies if use_target_time is False
time_str = f" ({target_time})" if use_target_time else ""

# Define a subgrid of the full pkl to use
get_subgrid = True
grid_center = (26.5, -81) # look 3/4
#grid_center = (23.9, -83.6) # look 5
grid_aspect = (4,4)
dpi=100
upper_bound = 230

#'''
# band 13
sat_key = "ian_tb_b13_2km-goes16"
#out_key = "ian_tb_b13_2km-goes16"
out_key = "ian-look4_tb_b13_2km-goes16"
plot_spec = {
    #"title": "GOES 16 Band 13 Brightness Temp"+time_str,
    "title": f"GOES 16 Band 13 Tb (Lower bound {upper_bound})"+time_str,
    "cb_label": "Tb (K)",
    }
#'''
'''
# band 5
sat_key = "ian_ref_b05_1km-goes16"
#out_key = "ian_ref_b05_1km-goes16"
out_key = "ian-look5_ref_b05_1km-goes16"
plot_spec = {
    "title":"GOES 16 Band 5 Reflectance"+time_str,
    "cb_label":"Reflectance",
    }
'''
'''
# band 2
sat_key = "ian_ref_b02_p5km-goes16"
#out_key = "ian_ref_b02_p5km-goes16"
out_key = "ian-look5_ref_b02_p5km-goes16"
plot_spec = {
    "title":"GOES 16 Band 2 Reflectance"+time_str,
    "cb_label":"Reflectance" }
'''

""" Get subgrids or select a frame with target_time """

gm.load_pkl(pkldir.joinpath(sat_key+".pkl"))
plot_spec.update({"dpi":dpi})
label = list(gm.subgrids.keys())[0]
am = gm.subgrids[label]["am"]
#matplotlib_path = figdir.joinpath(Path(out_key)),
raw_path = figdir.joinpath(Path(out_key+"-raw")),
if not upper_bound is None:
    am.restrict_data(bounds=(None, upper_bound),
                     replace_val=(None, upper_bound))

kk.geo_plot.get_scalar_graphics( am=am,
        matplotlib_path=figdir.joinpath(Path(out_key + \
                ("-restricted" if not upper_bound is None else ""))),
        raw_path=figdir.joinpath(Path(out_key + "-raw" + \
                ("-restricted" if not upper_bound is None else ""))),
        plot_spec=plot_spec,
        target_time=target_time if use_target_time else None,
        grid_center=grid_center if get_subgrid else None,
        grid_aspect=grid_aspect if get_subgrid else None,
        gif_fps = gif_fps,
        )

