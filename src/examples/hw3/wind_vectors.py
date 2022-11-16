"""
Helper script for generating truecolor RGBs using already-generated
ABIManager pkls
"""
import datetime as dt
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import gc

from pathlib import Path

from ABIManager import ABIManager
from geo_plot import geo_rgb_plot
from make_truecolor import make_truecolor

# Todo: Document all plot_spec options.
plot_spec = {
    "title_size":8,
    "gridline_color":"gray",
    "fig_size":(12,9),
    "dpi":1200,
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
    "vmin":None,
    "vmax":None,
    "marker_char":"x",
    "marker_color":"blue"
    }

class IceLook:
    """
    Describes separate 2d subsets of an RGB data array with respect to time,
    used for characterizing vertical and horizontal variation of RGB values.
    """
    def __init__(self, x:int, y:int, x_range:float, y_range:float,
                 name:str=None):
        """
        Initialize an IceLook with a
        """
        self.x = x # x pixel index of bottom left point
        self.y = y # y pixel index of bottom left point
        self.x_range = x_range # horizontal pixel range of subset
        self.y_range = y_range # vertical pixel range of subset
        self._name = name # attribute not currently used internally
        self.rgb_x = None
        self.rgb_y = None

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, name):
        self._name = name

    def get_vert_look(self, R:xr.DataArray, G:xr.DataArray, B:xr.DataArray):
        """
        :return: 2d-arrays
        """
        # Need to swap indeces here. I'm not sure why...
        y_idx = self.x
        x_idx = self.y
        self.rgb_y = (
            R.isel(x=x_idx).isel(y=slice(y_idx, y_idx+self.y_range)).data,
            G.isel(x=x_idx).isel(y=slice(y_idx, y_idx+self.y_range)).data,
            B.isel(x=x_idx).isel(y=slice(y_idx, y_idx+self.y_range)).data,
            )
        return self.rgb_y

    def get_horiz_look(self, R:xr.DataArray, G:xr.DataArray, B:xr.DataArray):
        """
        :return: (R, G, B) 2d arrays
        """
        # Need to swap indeces here. I'm not sure why...
        y_idx = self.x
        x_idx = self.y
        self.rgb_x = (
            R.isel(y=y_idx).isel(x=slice(x_idx, x_idx+self.x_range)).data,
            G.isel(y=y_idx).isel(x=slice(x_idx, x_idx+self.x_range)).data,
            B.isel(y=y_idx).isel(x=slice(x_idx, x_idx+self.x_range)).data,
            )
        return self.rgb_x


if __name__=="__main__":
    # Locations and pixel ranges to investigate ice movement
    looks = [
        IceLook(x=62, y=45, x_range=10, y_range=10, name="look1"),
        IceLook(x=45, y=30, x_range=10, y_range=10, name="look2"),
        IceLook(x=127, y=41, x_range=10, y_range=10, name="look3"),
        IceLook(x=70, y=60, x_range=10, y_range=10, name="look4"),
        IceLook(x=90, y=60, x_range=10, y_range=10, name="look5")
        ]

    # Get paths to all the data pkls and figure outputs.
    pkl_dir = Path("/home/krttd/uah/22.f/aes572/hw3/data/pkls/rgb")
    am01_pkl = pkl_dir.joinpath(Path("michigan_ref_b01_1km.pkl")) # Band 1
    am02_pkl = pkl_dir.joinpath(Path("michigan_ref_b02_1km.pkl")) # Band 2
    am03_pkl = pkl_dir.joinpath(Path("michigan_ref_b03_1km.pkl")) # Band 3

    # Output figure paths
    ice_markers_path = Path("/home/krttd/uah/22.f/aes572/hw3/data/figures/ice_markers.png")
    ice_look_path = "/home/krttd/uah/22.f/aes572/hw3/data/figures/final/ice_looks/q2d_ice_{name}.png"

    # Get the Dataset for each band
    am_B = ABIManager().load_pkl(am01_pkl)
    am_R = ABIManager().load_pkl(am02_pkl)
    am_G = ABIManager().load_pkl(am03_pkl)

    # Get the data arrays
    B = am_B.data["b1ref"]
    R = am_R.data["b2ref"][{"y":slice(0,156)}] # Drop extra latitude coordinate
    G = am_G.data["b3ref"]

    # Time of static reference image (used for displaying IceLook locations)
    target_idx = am_B.index_at_time(dt.datetime(
        year=2018, month=2, day=14, hour=17, minute=2))

    # Update the R/G/B arrays with the truecolor data
    # (exclusively bounded (0,1))
    R.data, G.data, B.data = make_truecolor(R=R.data, G=G.data, B=B.data)

    # Get each of the "ice looks." These are vertical and horizontal subsets of
    # the data plotted in RGB wrt time
    plot_spec["use_ticks"] = True
    for lk in looks:
        Rx, Gx, Bx = lk.get_horiz_look(R, G, B)
        # x distance ticks
        vert_ticks = [(i, am_B.data["lon"][lk.y, lk.x+i].values.item())
                  for i in range(lk.x_range)]
        geo_rgb_plot(Rx, Gx, Bx,
                fig_path=Path(ice_look_path.format(name=lk.name+"_x")),
                yticks=vert_ticks, plot_spec=plot_spec,
                extent=[0,3,0,1])

        Ry, Gy, By = lk.get_vert_look(R, G, B)
        # y distance ticks
        vert_ticks = [(j, am_B.data["lat"][lk.y+j, lk.x].values.item())
                  for j in range(lk.y_range)]
        geo_rgb_plot(Ry, Gy, By,
                fig_path=Path(ice_look_path.format(name=lk.name+"_y")),
                yticks=vert_ticks, plot_spec=plot_spec,
                extent=[0,3,0,1])

        del lk
        gc.collect()

    # Plot a image marking the loc
    plot_spec["use_ticks"] = True
    plot_spec.update({"markers":[(l.x, l.y) for l in looks]})
    geo_rgb_plot(
            R=R[:,:,target_idx], G=G[:,:,target_idx], B=B[:,:,target_idx],
            #fig_path=fig_path_gif,
            fig_path=ice_markers_path,
            plot_spec=plot_spec,
            animate=False,
            extent=None,
            )
