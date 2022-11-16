import krttdkit as kk
from pathlib import Path
import datetime as dt
from krttdkit import RecipeBook as RB
import numpy as np

pkldir = Path("/home/krttd/uah/22.f/aes572/hw2take2/data/pkls")
figdir = Path("/home/krttd/uah/22.f/aes572/hw2take2/data/figures/pyroCb_red")

gm = kk.GridManager()
#sat_key = "pyroCb_tb_b13_2km-goes16"
#sat_key = "pyroCb_tb_b13_2km-goes17"
#sat_key = "pyroCb_ref_b02_2km-goes16"
#sat_key = "pyroCb_ref_b02_2km-goes17"
#sat_key = "pyroCb_ref_b02_2km-goes17"
sat_key = "pyroCb_ref_b02_1km-goes17"
label = "ref"
gm.load_pkl(pkldir.joinpath(sat_key+".pkl"))

plot_spec = {
    "title":"ABI Band 2 Reflectance (1km)",
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

red_da = gm.subgrids[label]["am"].data[label]

# Plot a scalar projection with matplotlib
#"""
kk.geo_plot.geo_scalar_plot(
    data=red_da.data,
    lat=red_da["lat"],
    lon=red_da["lon"],
    fig_path=figdir.joinpath(Path(sat_key+".gif")),
    plot_spec=plot_spec,
    animate=True
    )
#"""

# Get an array version of the scalar data values normalized to [0,1]
darr = kk.GridManager.norm_to_unit(
        np.stack((red_da.data,red_da.data,red_da.data), axis=3))

grad = kk.ColoGrad(base_color="#09331d")
grad.set_colorstop(.2, "#d1c492")
grad.set_colorstop(.5, "#7ab8bf")
grad.set_colorstop(.8, "#CECA8C")
grad.set_colorstop(1., "#E67D30")

"""
grad.data = red_da.data
grad.symmetric_norm()
grad.to_rgb(method="linear")
"""

#print(grad.stops)

# Plot a "raw" unprojected gif at full resolution
kk.geo_plot.generate_raw_image(
        RGB=darr,
        image_path=figdir.joinpath(Path(sat_key+"-raw.gif")),
        gif=True,
        fps=15
        )

print(darr.shape)

# Plot a "raw" unprojected png at full resolution
kk.geo_plot.generate_raw_image(
        RGB=darr[:,:,20],
        image_path=figdir.joinpath(Path(sat_key+"-raw.png")),
        )
