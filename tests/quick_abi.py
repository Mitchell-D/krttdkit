#!/home/krttd/.anaconda3/envs/aes/bin/python
from pathlib import Path
import argparse
from datetime import datetime
from datetime import timedelta
import numpy as np
from krttdkit.products import ABIL1b, FeatureGrid
from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp
from krttdkit.operate.geo_helpers import get_geo_range

def next_abi(data_dir:Path, pkl_path:Path, target_time:datetime=None,
             satellite="16", scan="C", resolution=2, target_latlon=None,
             dx_px=640, dy_px=512, recipe="truecolor", show=True):
    assert data_dir.is_dir()
    response = "n"
    if pkl_path.exists():
        response = input(f"Overwrite previous FeatureGrid? (Y/n): ")

    if response.lower() == "y" or not pkl_path.exists():
        nc_files = ABIL1b.get_l1b(
            data_dir=data_dir,
            satellite=satellite,
            scan=scan,
            bands=list(range(1, 17)),
            start_time=target_time,
            replace=False,
            )[0]
        fg = ABIL1b.from_l1b_files(
                path_list=nc_files,
                convert_tb=True,
                convert_ref=True,
                get_latlon=True,
                get_mask=True,
                get_scanangle=False,
                resolution=resolution,
                )
        fg.to_pkl(pkl_path)
        for p in nc_files:
            p.unlink()
    else:
        fg = ABIL1b.from_pkl(pkl_path)

    print(fg.labels)
    if target_latlon:
        latlon = np.dstack((fg.data("lat"), fg.data("lon")))
        yrange,xrange = get_geo_range(latlon, target_latlon, dx_px, dy_px,
                                      True, True, False)
        fg = fg.subgrid(vrange=yrange, hrange=xrange)
    if show:
        gt.quick_render(fg.data(recipe))
    return fg

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
                        default="truecolor")
    parser.add_argument("--center", dest="center", type=str,
                        help="lat/lon center, formatted '\d+.\d+,\d+.\d+",
                        default=None)
    parser.add_argument("--aspect", dest="aspect", type=str,
                        help="Grid aspect ratio in pixels, " + \
                                "formatted '\d+.\d+,\d+.\d+",
                        default="640,520")
    parser.add_argument("--sat", dest="sat", type=str,
                        help="Satellite to query data from (16, 17, or 18)",
                        default="16")
    parser.add_argument("--scan", dest="scan", type=str,
                        help="Satellite scan (M1, M2, C, or F)",
                        default="C")
    parser.add_argument("--res", dest="res", type=str,
                        help="Scan resolution in pixels.",
                        default="2")
    parser.add_argument("--save", dest="save", type=str,
                        help="File path of image to save",
                        default=None)
    raw_args = parser.parse_args()

    if not raw_args.hour is None:
        hour = int(raw_args.hour)%24
        # Only regard the minutes argument if an hour is provided
        target_tod = timedelta( hours=hour,
                minutes=0 if raw_args.minute is None \
                        else int(raw_args.minute)%60)
        # If no day is provided, default to the last occurance of the
        # provided time.
        if raw_args.day is None:
            target_time = (datetime.utcnow()-target_tod).replace(
                    hour=0, minute=0, second=0, microsecond=0)+target_tod
        else:
            try:
                target_day = datetime.strptime(raw_args.day, "%Y%m%d")
                target_time = target_day+target_tod
            except:
                raise ValueError("Target day must be in YYYYmmdd format.")
    # Only accept a day argument if an hour is also provided
    # If no day argument or hour argument is provided, default to now.
    else:
        if raw_args.day is None:
            target_time = datetime.utcnow()#-timedelta(days=1)
        else:
            target_time = datetime.strptime(raw_args.day, "%Y%m%d")
    grid_center = None
    grid_aspect = None
    if raw_args.aspect:
        grid_aspect = tuple(map(float, raw_args.aspect.split(",")))
    if raw_args.center:
        grid_center = tuple(map(float, raw_args.center.split(",")))

    recipe = raw_args.recipe
    sat = raw_args.sat
    assert str(sat) in ("16", "17", "18")
    scan = raw_args.scan
    assert scan.upper() in ("M1", "M2", "C", "F")
    res = raw_args.res
    assert res in (".5", "1", "2")
    save_path = Path(raw_args.save) if raw_args.save else None
    if save_path:
        assert save_path.parent.exists()
    return (sat,target_time,grid_center,grid_aspect,recipe,scan,res,save_path)

if __name__=="__main__":
    data_dir=Path("/tmp")
    sat,target_time,grid_center,grid_aspect,recipe,scan,res,save = parse_args()
    fg = next_abi(
        data_dir=data_dir,
        pkl_path=data_dir.joinpath("abil1b_tmp.pkl"),
        target_time=target_time,
        recipe=recipe,
        satellite=sat,
        scan=scan,
        resolution=res, # km
        target_latlon=grid_center,
        dx_px=grid_aspect[0],
        dy_px=grid_aspect[1],
        )
    if save:
        gp.generate_raw_image(fg.data(recipe), save)
