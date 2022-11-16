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

def make_truecolor(R:np.ndarray, G:np.ndarray, B:np.ndarray):
    """
    Performs RGB color correction on the provided red, green, and blue
    ndarrays, which may be 2d single-frame values from ABI bands 1, 2, and 3,
    or 3d time series. In either case, the arrays must have the same shape.

    :return: (R, G, B) tuple of ndarrays that have been color-corrected
        according to the CIMSS Natural True Color RGB recipe.

    Recipe reference:
    https://www.star.nesdis.noaa.gov/GOES/documents/ABIQuickGuide_CIMSSRGB_v2.pdf
    """

    # Normalize the reflectance to exclusive (0,1).
    # If the L1b radiances were actually converted, all the grid values
    # are probably already well within this range.
    B = np.clip(B, 0, 1)
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)

    # Gamma correction using gamma=2.2 for standard digital displays
    # (ref: https://en.wikipedia.org/wiki/Gamma_correction)
    gamma = 2.2
    B = np.power(B, 1/gamma)
    R = np.power(R, 1/gamma)
    G = np.power(G, 1/gamma)

    #print(type(B), type(R), type(G))
    #print(B.shape, R.shape, G.shape)

    # Get "True color" green according to CIMSS recipe, and collect the RGB
    # (ref: CIMSS Natural True Color Quick Guide)
    G_TC = np.clip(.45*R+.1*G+.45*B, 0, 1)

    return (R, G_TC, B)

if __name__=="__main__":
    plot_spec = {
        "title":"Truecolor RGB",
        "title_size":8,
        "gridline_color":"gray",
        #"fig_size":(48,24),
        "dpi":500,
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
        "xtick_size":3,
        "ytick_count":12,
        "ytick_size":3,
        }


    # If single_frame is True, selects the DataArray closest to the target
    # time and generates a single image based on it.
    single_frame=False
    #target_time = dt.datetime(year=2018, month=2, day=14, hour=17, minute=2)
    #target_time = dt.datetime(year=2020, month=8, day=27, hour=15)
    #target_length = dt.timedelta(hours=5)
    target_time = None
    target_length = None

    # Get paths to all the data pkls and figure outputs.
    """
    pkl_dir = Path("/home/krttd/uah/22.f/aes572/hw3/data/pkls/rgb")
    am01_pkl = pkl_dir.joinpath(Path("michigan_ref_b01_1km.pkl")) # Band 1
    am02_pkl = pkl_dir.joinpath(Path("michigan_ref_b02_1km.pkl")) # Band 2
    am03_pkl = pkl_dir.joinpath(Path("michigan_ref_b03_1km.pkl")) # Band 3
    """
    #"""
    pkl_dir = Path("/home/krttd/uah/22.f/aes572/hw3/data/pkls/laura_conus")
    am01_pkl = pkl_dir.joinpath(Path("laura-conus_ref_b01_1km.pkl")) # Band 1
    am02_pkl = pkl_dir.joinpath(Path("laura-conus_ref_b02_1km.pkl")) # Band 2
    am03_pkl = pkl_dir.joinpath(Path("laura-conus_ref_b03_1km.pkl")) # Band 3
    #"""

    # Output figure paths
    #fig_path = Path("/home/krttd/uah/22.f/aes572/hw3/data/figures/truecolor.gif")
    #fig_path = Path("/home/krttd/uah/22.f/aes572/hw3/data/figures/truecolor.png")
    #fig_path = Path("/home/krttd/uah/22.f/aes572/hw3/data/figures/ice_sample.png")
    #fig_path = Path("/home/krttd/uah/22.f/aes572/hw3/data/figures/laura-conus_truecolor-1km.png")
    #fig_path = Path("/home/krttd/uah/22.f/aes572/hw3/data/figures/laura-conus_truecolor-1km.gif")
    #fig_path = Path("/home/krttd/uah/22.f/aes572/hw3/data/figures/laura-conus_truecolor-1km-fullres.png")
    fig_path = Path("/home/krttd/uah/22.f/aes572/hw3/data/figures/laura-conus_truecolor-1km-fullres.gif")


    # Get the Dataset for each band
    am_B = ABIManager().load_pkl(am01_pkl)
    am_R = ABIManager().load_pkl(am02_pkl)
    am_G = ABIManager().load_pkl(am03_pkl)

    if single_frame:
        # Subset red channel (range selection left an extra lon column)
        B = am_B.array_from_time(target_time)["b1ref"]
        R = am_R.array_from_time(target_time)["b2ref"]#.isel(y=slice(0,156))
        G = am_G.array_from_time(target_time)["b3ref"]

    else:
        # Get the DataArray attribute from the ABIManager
        B = am_B.data["b1ref"]
        R = am_R.data["b2ref"]
        G = am_G.data["b3ref"]
        # Subset the time series to the provided range.
        if target_time and target_length:
            start_idx_b = am_B.index_at_time(target_time)
            end_idx_b = am_B.index_at_time(target_time+target_length)
            B = B.isel(time=slice(start_idx_b, end_idx_b))
            start_idx_r = am_R.index_at_time(target_time)
            end_idx_r = am_R.index_at_time(target_time+target_length)
            R = R.isel(time=slice(start_idx_r, end_idx_r))#[{"y":slice(0,156)}]
            start_idx_g = am_G.index_at_time(target_time)
            end_idx_g = am_G.index_at_time(target_time+target_length)
            G = G.isel(time=slice(start_idx_g, end_idx_g))

    # Get ndarrays for the truecolor RGB values of the grid.
    R, G, B = make_truecolor(R=R.data, G=G.data, B=B.data)

    #print(type(B), type(R), type(G))
    #print(R.shape, G.shape, B.shape)

    generate_raw_image(
            #RGB=np.dstack((R,G,B)),
            RGB=np.stack((R,G,B), axis=3),
            image_path=fig_path,
            gif=True,
            fps=30
            )
    """
    geo_rgb_plot(
            R=R, G=G, B=B,
            #lat=am_B.data["lat"], lon=am_B.data["lon"],
            fig_path=fig_path,
            plot_spec=plot_spec,
            animate=not single_frame
            )
    """
