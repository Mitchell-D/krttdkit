import krttdkit as kk
from pathlib import Path
from datetime import datetime as dt
from krttdkit import ABIManager
import numpy as np

""" Script for generating b2-b5 channel differences """

pkldir = Path("/home/krttd/uah/22.f/aes572/hw2take2/data/pkls")
figdir = Path("/home/krttd/uah/22.f/aes572/hw2take2/data/figures/ian/diff")
buffdir = Path("/home/krttd/uah/22.f/aes572/hw2take2/data/buffer")

gm = kk.GridManager(buffer_dir=buffdir.as_posix())

# Choose whether to make a static image, and select a time
use_target_time = True
target_time = dt(year=2022, month=9, day=27, hour=22) # t1

# Define a subgrid of the full pkl to use
get_subgrid = True
#grid_center = (26.5, -81) # look 3/4
grid_center = (23.9, -83.6) # look 5
grid_aspect = (1,1)

gif_fps = 10 # Only applies if use_target_time is False
time_str = f" ({target_time})" if use_target_time else ""

inf_sat_key = "ian_ref_b05_1km-goes16"
vis_sat_key = "ian_ref_b02_p5km-goes16"
out_key = "ian-look5_diff_b2-5_1km-goes16"
raw_path = figdir.joinpath(Path(out_key+"-raw"))
mpl_path = figdir.joinpath(Path(out_key))

plot_spec = {
    "title":"ABI Band 2-5 Difference"+time_str,
    "cb_label":"Reflectance",
    "border_width":0.2,
    "dpi":200
    }

# Open the pkl files and regrid band 2
gm.load_pkl(pkldir.joinpath(inf_sat_key+".pkl"))
gm.load_pkl(pkldir.joinpath(vis_sat_key+".pkl"))
geom = gm.subgrids["b2ref"]["am"].geom
gm.stride("b2ref", 2)
band2 = gm.subgrids["b2ref"]["am"].data["b2ref"]
band5 = gm.subgrids["b5ref"]["am"].data["b5ref"].isel(x=slice(0,930))
diff_am = gm.subgrids["b2ref"]["am"]
gm.clear()

#diff_am.data["diff"] = band2-band5
#diff_am.data["b2ref"].data = gm.norm_to_unit(np.clip(band2.data,0,1)) - \
#        np.clip(gm.norm_to_unit(band5.data),0,1)
diff_am.data["b2ref"].data = np.clip(band2.data,0,1) - \
        np.clip(band5.data,0,1)

# Load an ABIManager with the data
#am = kk.ABIManager().from_data_and_geom(
#        data=band2-band5, geom=geom, dataset_label="diff")
#diff_am.data["b2ref"].data = gm.norm_to_uint8(
#        diff_am.data["b2ref"].data)

kk.geo_plot.get_scalar_graphics(
        am=diff_am,
        matplotlib_path=mpl_path,
        raw_path=raw_path,
        plot_spec=plot_spec,
        target_time=target_time if use_target_time else None,
        grid_center=grid_center if get_subgrid else None,
        grid_aspect=grid_aspect if get_subgrid else None,
        gif_fps=8
        )
