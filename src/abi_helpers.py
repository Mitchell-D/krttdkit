"""
basic helper method wrapping geographic range and data selection for
generating ABIManager pkls. This should probably be integrated directly
into the ABIManager class.

TODO:
 - Generalize "GeoLook class" for defining lat/lon boundaries and copying
   ABIManager axes onto each other.
 -

"""

from .ABIManager import ABIManager
from pathlib import Path
from datetime import datetime as dt
from datetime import timedelta as td
import argparse

def ncs_to_AM_subgrid(
        nc_dir:Path, dataset_label:str, field:str, grid_center:tuple=None,
        grid_aspect:tuple=None, convert_Tb:bool=False, convert_Ref:bool=False,
        stride:int=1, pkl_path:Path=None, ftype:str="noaa_nws", _debug:bool=False):
    """
    Wraps ABIManager.load_netCDFs with default options configured on the module
    level, returning an ABIManager object for a continuous time series of
    netCDF data in the provided directory.
    """
    sorted_files = list(nc_dir.iterdir())
    sorted_files.sort()

    get_subgrid = (grid_center is not None) and (grid_aspect is not None)
    lat_range = (grid_center[0]-grid_aspect[0]/2,
                 grid_center[0]+grid_aspect[0]/2) if get_subgrid else None
    lon_range = (grid_center[1]-grid_aspect[1]/2,
                 grid_center[1]+grid_aspect[1]/2) if get_subgrid else None

    AM = ABIManager()
    AM.load_netCDFs(nc_paths=sorted_files, ftype=ftype,
        dataset_label=dataset_label, field=field, lat_range=lat_range,
        lon_range=lon_range, buffer_arrays=buffer_arrays,
        buffer_dir=buffer_dir, buffer_keep_pkls=buffer_keep_pkls,
        buffer_append="bufpkl", convert_Tb=convert_Tb, convert_Ref=convert_Ref,
        stride=stride, _debug=_debug)
    if not pkl_path is None:
        AM.make_pkl(pkl_dir.joinpath(pkl_path))
    return AM

def align_arrays(A, B):
    if A.shape!=B.shape:
        ydiff = A.shape[0]-B.shape[0]
        xdiff = A.shape[1]-B.shape[1]
        if ydiff>0:# A has ydiff more elements in the y direction
            A = A[:-ydiff]
        if ydiff<0:# B has ydiff more elements in the y direction
            B = B[:ydiff,:]
        if xdiff>0:# A has xdiff more elements in the x direction
            A = A[:,:-xdiff]
        if xdiff<0:# B has xdiff more elements in the x direction
            B = B[:,:xdiff]
    return A, B

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
                        help="Grid aspect ratio, formatted '\d+.\d+,\d+.\d+",
                        default=None)
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
        # If a day and
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
            target_time = dt.utcnow()
        else:
            raise ValueError("You cannot specify a day without " + \
                    "also specifying an hour.")
    grid_center = None
    grid_aspect = None
    if raw_args.center and raw_args.aspect:
        grid_center = tuple(map(float, raw_args.center.split(",")))
        grid_aspect = tuple(map(float, raw_args.aspect.split(",")))
    elif raw_args.center or raw_args.aspect:
        raise ValueError("You must provide both a center and an aspect ratio")

    return raw_args.recipe, target_time, grid_center, grid_aspect

if __name__=="__main__":
    # Directory where temporary buffer pkls can be stored.
    buffer_dir = Path("/home/krttd/uah/22.f/aes572/hw3/data/buffer")
    # If False, buffer pkls are deleted after being incorporated
    buffer_keep_pkls = False
    # If True, pkl array buffering will be used to preserve memory while
    # loading a large netCDF time series.
    buffer_arrays = True,
    # File format; currently noaa_aws is the only supported option.
    ftype="noaa_aws"
    # Directory where "finished" pkls are stored for graphing
    pkl_dir = Path("/home/krttd/uah/22.f/aes572/hw3/data/pkls")

    # Great lakes netCDF data directories
    """
    gl_band1 = Path("/home/krttd/uah/22.f/aes572/hw3/data/great_lakes/band01")
    gl_band2 = Path("/home/krttd/uah/22.f/aes572/hw3/data/great_lakes/band02")
    gl_band3 = Path("/home/krttd/uah/22.f/aes572/hw3/data/great_lakes/band03")
    """
    subgrid_params = {
        "nc_dir":laura_b12,
        "dataset_label":"b12ref",
        "stride": 1,
        "pkl_path":Path("laura_conus/laura-conus_tb_b12_2km.pkl"),
        "field":"Rad",
        "grid_center":(30, -87.1), # laura_conus
        "grid_aspect":(25, 45),    # laura_conus
        #"grid_center":(42, -87.2), # S. Superior
        #"grid_aspect":(1.6, 2.4),  # S. Superior
        "convert_Ref":False,
        "convert_Tb":True,
        }

    am = ncs_to_AM_subgrid( **subgrid_params, _debug=True)

