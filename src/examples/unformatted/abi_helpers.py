"""
basic helper method wrapping geographic range and data selection for
generating ABIManager pkls. This should probably be integrated directly
into the ABIManager class.

TODO:
 - Generalize "GeoLook class" for defining lat/lon boundaries and copying
   ABIManager axes onto each other.
 -

"""

from ABIManager import ABIManager
from pathlib import Path

def ncs_to_AM_subgrid(
        nc_dir:Path, dataset_label:str, field:str, grid_center:tuple=None,
        grid_aspect:tuple=None, convert_Tb:bool=False, convert_Ref:bool=False,
        stride:int=1, pkl_path:Path=None, _debug:bool=False):
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
    gl_band5 = Path("/home/krttd/uah/22.f/aes572/hw3/data/great_lakes/band05")
    gl_band7 = Path("/home/krttd/uah/22.f/aes572/hw3/data/great_lakes/band07")
    """

    laura_b1 = Path("/mnt/warehouse/data/abi/aes572_hw3/laura_conus/band01")
    laura_b2 = Path("/mnt/warehouse/data/abi/aes572_hw3/laura_conus/band02")
    laura_b3 = Path("/mnt/warehouse/data/abi/aes572_hw3/laura_conus/band03")

    laura_b8 = Path("/mnt/warehouse/data/abi/aes572_hw3/laura_conus/band08")
    laura_b10 = Path("/mnt/warehouse/data/abi/aes572_hw3/laura_conus/band10")
    laura_b12 = Path("/mnt/warehouse/data/abi/aes572_hw3/laura_conus/band12")
    laura_b13 = Path("/mnt/warehouse/data/abi/aes572_hw3/laura_conus/band13")

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

