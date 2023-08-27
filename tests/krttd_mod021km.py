"""
Primary driver script for the aes670 final project, now a relic
that will (hopefully) remain backward-compatible with krttdkit.
"""

from pathlib import Path
from datetime import datetime as dt
from datetime import timedelta as td
import pickle as pkl
from pprint import pprint as ppt
import numpy as np
import argparse
import subprocess
import shlex

import krttdkit.visualize.guitools as gt
import krttdkit.operate.enhance as enh
import krttdkit.visualize.geoplot as gp
from krttdkit.visualize import TextFormat as TFmt
from krttdkit.products import MOD021KM
from krttdkit.operate import Recipe

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--hour", dest="hour", type=str,
                        help="UTC hour of observation. If no day is " + \
                                "provided, defaults to today.",
                        default=None)
    parser.add_argument("-M", "--minute", dest="minute", type=str,
                        help="Minute of observation. If no hour is "+ \
                                "provided, this value is ignored.",
                        default=None)
    parser.add_argument("-D", "--day", dest="day", type=str,
                        help="Day in YYYYMMDD format",
                        default=None)
    parser.add_argument("-r", "--recipe", dest="recipe", type=str,
                        help="Imagery recipe to use; defaults to truecolor",
                        default="TC")
    parser.add_argument("--center", dest="center", type=str,
                        help="lat/lon center, formatted '\d+.\d+,\d+.\d+",
                        default=None)
    parser.add_argument("--aspect", dest="aspect", type=str,
                        help="Grid aspect ratio in pixels, " + \
                                "formatted '\d+.\d+,\d+.\d+",
                        default="1080,1920")
    parser.add_argument("--sat", dest="sat", type=str,
                        help="Satellite to query data from",
                        default="terra")
    raw_args = parser.parse_args()

    if not raw_args.hour is None:
        hour = int(raw_args.hour)%24
        # Only regard the minutes argument if an hour is provided
        target_tod = td( hours=hour,
                minutes=0 if raw_args.minute is None \
                        else int(raw_args.minute)%60)
        # If no day is provided, default to the last occurance of the
        # provided time.
        if raw_args.day is None:
            target_time = (dt.utcnow()-target_tod).replace(
                    hour=0, minute=0, second=0, microsecond=0)+target_tod
        else:
            try:
                target_day = dt.strptime(raw_args.day, "%Y%m%d")
                target_time = target_day+target_tod
            except:
                raise ValueError("Target day must be in YYYYmmdd format.")
    # Only accept a day argument if an hour is also provided
    # If no day argument or hour argument is provided, default to now.
    else:
        if raw_args.day is None:
            target_time = dt.utcnow()-td(days=1)
        else:
            target_time = dt.strptime(raw_args.day, "%Y%m%d")
    grid_center = None
    grid_aspect = None
    if raw_args.aspect:
        grid_aspect = tuple(map(float, raw_args.aspect.split(",")))
    if raw_args.center:
        grid_center = tuple(map(float, raw_args.center.split(",")))

    satellite = raw_args.sat
    assert satellite in ("terra", "aqua")
    recipe = raw_args.recipe
    return (satellite, target_time, grid_center, grid_aspect, recipe)

def get_rgbs(subgrid, fig_dir, gamma_scale:int=2, choose=False):
    for r in subgrid.rgb_recipes.keys():
        print(TFmt.GREEN("Choose RGB gamma for ")+TFmt.WHITE(r, bold=True))
        rgb = subgrid.get_rgb(r, choose_gamma=choose, gamma_scale=gamma_scale)
        gp.generate_raw_image(rgb, fig_dir.joinpath(f"rgbs/rgb_{r}.png"))

if __name__=="__main__":
    """ Settings """
    token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUFMgT0F1dGgyIEF1dGhlbnRpY2F0b3IiLCJpYXQiOjE2Nzg0MDEyNzgsIm5iZiI6MTY3ODQwMTI3OCwiZXhwIjoxNjkzOTUzMjc4LCJ1aWQiOiJtZG9kc29uIiwiZW1haWxfYWRkcmVzcyI6Im10ZDAwMTJAdWFoLmVkdSIsInRva2VuQ3JlYXRvciI6Im1kb2Rzb24ifQ.gwlWtdrGZ1CNqeGuNvj841SjnC1TkUkjxb6r-w4SOmk"
    l1b_bands = (
        3,  # 459-479nm blue
        10, # 483-493nm teal/blue
        4,  # 545-565nm green
        1,  # 620-670nm near-red
        2,  # 841-876nm NDWI / land boundaries
        16, # 862-877nm NIR / aerosol distinction
        19, # 916-965nm H2O absorption
        5,  # 1230-1250nm optical depth
        26, # 1360-1390nm cirrus band
        6,  # 1628-1652nm snow/ice band
        7,  # 2106-2155nm cloud particle size
        20, # 3660-3840nm SWIR
        21, # 3929-3989 another SWIR
        27, # 6535-6895nm Upper H2O absorption
        28, # 7175-7475nm Lower H2O absorption
        29, # 8400-8700nm Infrared cloud phase, emissivity diff 11-8.5um
        31, # 10780-11280nm clean LWIR
        32, # 11770-12270nm less clean LWIR
        33, # 14085-14385nm dirty LWIR
        )
    debug = True
    hsv_params = {"hue_range":(.6,0),"sat_range":(.6,.9),"val_range":(.6,.6)}
    data_dir = Path("/tmp/rgb_bg/mod021km")
    hdf_path = data_dir.joinpath("tmp_mod021km.hdf")

    target_latlon = (32.416, 32.987)
    target_time = dt(year=2019, month=5, day=10, hour=8, minute=29)
    satellite = "terra"
    satellite, target_time, target_latlon, region_aspect, recipe = parse_args()
    # Transpose since MOD021KM are (2030,1354)
    region_width, region_height = region_aspect
    if input("Download new hdf? (Y/n) ").lower()=="y":
        tmp_path = MOD021KM.download_granule(
                data_dir=data_dir,
                raw_token = token,
                target_latlon=target_latlon,
                satellite=satellite,
                target_time=target_time,
                day_only=True,
                debug=debug,
                )
        print(f"Downloaded l1b: {tmp_path.name}")
        tmp_path.rename(hdf_path)


    M = MOD021KM.from_hdf(hdf_path, l1b_bands)
    if target_latlon is None:
        ctr_px = (int(M.shape[0]/2), int(M.shape[1]/2))
        target_latlon = (M.data("lat")[*ctr_px],
                         M.data("lon")[*ctr_px])
    subgrid = M.get_subgrid(target_latlon, region_height, region_width,
                            from_center=True, boundary_error=False)
    rgb_path = data_dir.joinpath(f"mod021km_{recipe}.png")
    gp.generate_raw_image(subgrid.get_rgb(recipe).transpose(1,0,2), rgb_path)
    subprocess.run(shlex.split(f"feh --bg-fill -q {rgb_path.as_posix()}"))
    #subgrid.make_pkl(pkl_path)
    #subgrid = MOD021KM.from_pkl(pkl_path)
    #get_rgbs(subgrid, fig_dir, choose=True, gamma_scale=4)
