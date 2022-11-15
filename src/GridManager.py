from dataclasses import dataclass
import datetime as dt
from pathlib import Path
import gc
import xarray as xr
import numpy as np
from .ABIManager import ABIManager

class GridManager:
    def __init__(self, buffer_dir:Path=None):
        self._subgrids = {}
        self._buffer_dir = buffer_dir
        # Functionally-evaluated attribute providing an iterable generator
        # that collects all current subgrid dataarrays, under the hood of
        # their respective ABIManagers. List of (key:str, grid:xr.DataArray)
        self._dss = lambda: (self._subgrids[k]["am"].data[k]
                             for k in self.labels)

    @staticmethod
    def _get_grid_bounds(grid_center:tuple, grid_aspect:tuple):
        get_subgrid = (grid_center is not None) and (grid_aspect is not None)
        lat_range = (grid_center[0]-grid_aspect[0]/2,
                grid_center[0]+grid_aspect[0]/2) if get_subgrid else None
        lon_range = (grid_center[1]-grid_aspect[1]/2,
                grid_center[1]+grid_aspect[1]/2) if get_subgrid else None
        return (lat_range, lon_range)

    def get_abi_grid(self, data_dir:Path, pkl_path:Path=None,
            grid_center:tuple=None, grid_aspect:tuple=None, label:str="band",
            stride:int=1, convert_Ref:bool=False, convert_Tb:bool=False,
            field:str="Rad", ftype:str="noaa_aws", ti:dt.datetime=None,
            tf:dt.datetime=None, _debug:bool=False):

        buffer_arrays = not self._buffer_dir is None
        lat_range, lon_range = GridManager._get_grid_bounds(
                grid_center, grid_aspect)

        # Get in-range files ordered chronologically
        sorted_files = ABIManager.paths_in_time_range(
                data_dir, ftype, ti, tf)

        am = ABIManager().load_netCDFs(nc_paths=sorted_files, ftype=ftype,
            dataset_label=label, field=field, lat_range=lat_range,
            lon_range=lon_range, buffer_arrays=buffer_arrays,
            buffer_dir=self._buffer_dir, buffer_keep_pkls=False,
            buffer_append="bufpkl", convert_Tb=convert_Tb,
            convert_Ref=convert_Ref, stride=stride, _debug=_debug)

        if not pkl_path is None:
            am.make_pkl(pkl_path)

        self._subgrids.update({
            label:{
                "data_dir":data_dir,
                "pkl_path":pkl_path,
                "am":am
                }
            })

    '''
    def align(self, keys:tuple, method="left"):
        """
        Aligns the ABIManager Datasets corresponding to the
        provided keys along a common dimensional axes.
        """
        aligned = list(zip(keys, xr.align(
                *(self._subgrids[k]["am"].data for k in keys),
                join=method)))
        for al in aligned:
            key, sg = al
            self._subgrids[key]["am"] =
    '''

    @staticmethod
    def norm_to_unit(arr:np.ndarray, unit:int=1):
        return unit*(arr-np.min(arr))/(np.max(arr)-np.min(arr))
    @staticmethod
    def norm_to_uint8(arr:np.ndarray, resolution:int=256):
        """
        Normalize data to linear 0,1 then scale by the provided resolution
        before converting to integer.
        """
        norm = GridManager.norm_to_unit(arr)
        return (resolution*norm).astype(np.uint8)

    def stride(self, key:str, intv:int):
        """
        Subset the grid corresponding to the provided key by parsing every nth
        index out of the x and y dimension coordinate arrays and data
        """
        self._subgrids[key]["am"].set_stride(intv)


    def clear(self, key:str=None):
        """
        Unload an ABIManager from memory using its key.
        If no key is provided, all subgrids are deleted.
        """
        if not key is None and key not in self.labels:
            raise ValueError(f"Provided key {key} is not one of " + \
                    f"{list(self.labels)}")

        for sg in list(self.labels):
            del self._subgrids[sg]
            print(f"removed subgrid {sg}")
        gc.collect()

    def load_pkl(self, pkl_path:Path):
        """ Load an ABIManager pkl """
        am = ABIManager().load_pkl(pkl_path)
        if am._label in self.labels:
            print("Warning: subgrid labeled {am._label} is already one " + \
                    "of this GridManager's subgrids")
        self._subgrids.update({am._label:{"am":am, "pkl_path":pkl_path}})
        print(f"loaded pkl {pkl_path.as_posix()}")
        return self

    @property
    def coords(self):
        """
        Returns None if the subgrids aren't uniform, or a xarray coordinate
        hashable if all subgrids are using the same coordinate grid.
        """
        uniform = self.uniform
        return None if not uniform else list(self._dss())[0].coords

    @property
    def labels(self):
        """ Returns a tuple of all current subgrid labels (keys) """
        return tuple(self._subgrids.keys())

    @property
    def subgrids(self):
        return self._subgrids

    @property
    def uniform(self):
        """
        Functionally-evaluated property, True when all loaded subgrids have
        the same shape and coordinate axes
        """
        #"""
        try:
            xr.align(*tuple(self._dss()), join="exact")
        # If exact alignment raises an error, coordinates are mismatched.
        except ValueError as ve:
            return False
        return True

    def grid_shapes(self):
        """
        returns a tuple of all unique grid shapes loaded as subgrids. This is
        useful for checking if all grids are uniform
        """
        return tuple(set((ds.shape for ds in self._dss())))

    def snap_to_coords(self, coords):
        """
        Set the coordinates of all ABIManager data to the provided mapping.
        This works under the hood of subgrid ABIManagers, which is a somewhat
        unstable operation.

        Raises an error if all loaded subgrids aren't uniform
        (see GridManager.uniform property)
        """
        if not len(self.grid_shapes())==1:
            raise ValueError("Not all loaded subgrids have the same shape: "+\
                    f"{self.grid_shapes()}")
        for k in self.labels:
            assigned = self._subgrids[k]["am"].data.assign_coords(coords)
            self._subgrids[k]["am"]._data = assigned
