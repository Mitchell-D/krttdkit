from pathlib import Path
from datetime import datetime
import numpy as np
from krttdkit.products import ABIL1b, FeatureGrid
from krttdkit.visualize import guitools as gt
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

    if target_latlon:
        latlon = np.dstack((fg.data("lat"), fg.data("lon")))
        yrange,xrange = get_geo_range(latlon, target_latlon, dx_px, dy_px,
                                      True, True, False)
        fg = fg.subgrid(vrange=yrange, hrange=xrange)
    if show:
        gt.quick_render(fg.data(recipe))
    return fg

if __name__=="__main__":
    data_dir=Path("/tmp")
    fg = next_abi(
        data_dir=data_dir,
        pkl_path=data_dir.joinpath("abil1b_tmp.pkl"),
        #target_time=datetime(2023,8,14,22,30),
        target_time=datetime(2023, 5, 20, 18),
        recipe="lingamma wv",
        satellite="16",
        scan="C",
        resolution=2, # km
        target_latlon=(34.7, -86.6),
        #target_latlon=(40, -90),
        dx_px=640*2,
        dy_px=512*2,
        )
    print(fg)
