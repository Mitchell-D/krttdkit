from pathlib import Path
from datetime import datetime
import numpy as np
import pickle as pkl
import skimage as sk
import json

from krttdkit.products import FeatureGrid
from krttdkit.acquire import abi
from krttdkit.acquire.get_goes import parse_goes_path
#from krttdkit.operate.geo_helpers import get_geo_range
from krttdkit.operate.recipe_book import abi_recipes

class ABIL1b:
    @staticmethod
    def from_pkl(pkl_path:Path):
        """
        Load a pkl generated with abil1b.to_pkl() by a former a instance of the
        class, returning a new ABIL1b object identical to the original.

        :@param pkl_path: Valid path to existing ABIL1b pkl
        :@return: ABIL1b instance with the data arrays and information loaded
        """
        data, fg_dict = pkl.load(pkl_path.open("rb"))
        abil1b = ABIL1b(fg_dict["labels"], data, fg_dict["info"])
        #abil1b.meta.update(fg_dict["meta"])
        return abil1b

    @staticmethod
    def get_l1b(data_dir:Path, satellite:str, scan:str, bands:list,
                start_time:datetime, end_time:datetime=None,
                replace:bool=False):
        """
        Download L1b data from the NOAA S3 bucket to a data directory given
        time constraints and a list of bands.

        Basic wrapper on krttdkit.acquire.abi.download_l1b

        :@param data_dir: Directory to store downloaded netCDF files
        :@param satellite: GOES satellite number ('16', '17', or '18')
        :@param scan: Satellite scan type: 'C' for CONUS, 'F' for Full Disc,
            and 'M1' or 'M2' for Mesoscale 1 or 2.
        :@param bands: List of ABI bands, [1-16]. The order of this list
            corresponds to the order of the returned netCDF files of each band.
        :@param start_time: Inclusive initial time of the time range to search
            if end_time is defined. Otherwise if end_time is None, start_time
            stands for the target time of returned files.
        :@param end_time: If end_time isn't None, it defines the exclusive
            upper bound of a time range in which to search for files.

        :@return: a list of paths to the downloaded files.
        """
        bands = [abi.valid_band(b) for b in bands]
        return abi.download_l1b(
                data_dir=data_dir, satellite=satellite, scan=scan, bands=bands,
                start_time=start_time, end_time=end_time, replace=replace)

    @staticmethod
    def from_l1b_files(path_list:list, convert_tb:bool=False,
                   convert_ref:bool=False, get_latlon:bool=True,
                   get_mask=True, get_scanangle:bool=False, resolution=None,
                   reduce_method=np.max):
        """
        Parse a list of netCDF files with identical start times as an ABIL1b
        object. Each netCDF must be L1b, correspond to a unique band, and have
        the same stime as all others in the list

        :@param path_list: List of paths to valid netCDF files following the
            above constraints.
        :@param convert_tb: If True, converts thermal bands to brightness temp
        :@param convert_ref: If True, converts reflective bands to reflectance
        :@param get_latlon: If True, adds channels for latitude and longitude.
        :@param get_scanangle: If True, adds  channels for radian scan angles
        :@param resolution: Resolution of data (in km). The provided resolution
            can not be higher than the highest-resolution channel provided in
            path_list (ie if resolution=.5, channel 2 must be included). Grids
            will be up-sampled or down-sampled as needed.
            NOTE: resolution must be one of {.5, 1, 2}

        :@return: ABIL1b object with channels corresponding to the parsed
            files and any additional requested datasets.
        """
        assert all(p.exists() for p in path_list)
        files = sorted([parse_goes_path(f) for f in path_list])
        # Make sure this is an ABI L1b file
        assert all(f.product.sensor=="ABI" and f.product.level=="L1b"
                   for f in files)
        file_labels = [f.label for f in files]
        # All files must have the same stime
        assert len(set(f.stime for f in files))==1
        # All files must have a unique label
        assert len(set(file_labels)) == len(file_labels)
        bands = [(abi.valid_band(f.label[-2:]), f) for f in files]
        bands, files = zip(*sorted(bands, key=lambda t:t[0]))

        data = []
        info = []
        labels = []
        resolutions = [abi.bands[b]["default_res"] for b in bands]
        # Grid resolution is the lowest by default, or the provided resolution
        # if the provided resolution isn't
        target_res = float(resolution) if not resolution is None \
                else max(resolutions)
        # Provided resolution must be one of the file resolutions
        assert not all(r!=target_res for r in resolutions)
        for i in range(len(bands)):
            tmp_info = abi.bands[bands[i]]
            tmp_info.update({
                "band":bands[i],
                "path":files[i].path,
                "stime":files[i].stime.strftime("%Y%m%d-%H%M"),
                })
            info.append(tmp_info)

            # Currently 3.9um has no kappa0 value for reflectance conversion
            # so reflectance and brightness temp are mutually exclusive.
            # Calculate reflectance as a recipe if you want to use your own k0
            if convert_ref and abi.is_reflective(bands[i]):
                labels.append(f"{bands[i]}-ref")
                data.append(abi.get_abi_l1b_ref(Path(files[i].path)))
            elif convert_tb and abi.is_thermal(bands[i]):
                labels.append(f"{bands[i]}-tb")
                data.append(abi.get_abi_l1b_Tb(Path(files[i].path)))
            else:
                labels.append(f"{bands[i]}-rad")
                data.append(abi.get_abi_l1b_radiance(Path(files[i].path)))

            # Downscale finer resolutions  using reduce_method
            tmp_res = resolutions[i]
            while tmp_res<target_res:
                tmp_res *= 2
                data[i] = sk.measure.block_reduce(
                        data[i], (2,2), reduce_method)
            # Upscale coarser resolutions by tiling data.
            while tmp_res>target_res:
                tmp_res /= 2
                data[i] = np.repeat(np.repeat(data[i], 2, axis=0), 2, axis=1)
        if get_latlon or get_scanangle:
            # Get a netCDF with the targeted resolution
            sample = Path(files[resolutions.index(target_res)].path)
            # At least one resolution must be the target resolution.
            if get_latlon:
                lats, lons = abi.get_abi_l1b_latlon(sample)
                data += [lats, lons]
                labels += ["lat", "lon"]
                info += [{"units":"deg n"}, {"units":"deg e"} ]
            if get_scanangle:
                ns, ew = abi.get_abi_l1b_scanangle(sample)
                data += [ns, ew]
                labels += ["sa-ns", "sa-ew"]
                info += [{"units":"radians"}, {"units":"radians"}]
        fg = FeatureGrid(labels, data, info)
        for label, recipe in abi_recipes.items():
            fg.add_recipe(label, recipe)
        #abil1b.update_meta("stime",files[0].stime.strftime("%s"))
        return fg

    @staticmethod
    def from_pkl(pkl_path:Path):
        fg = FeatureGrid.from_pkl(pkl_path)
        for label, recipe in abi_recipes.items():
            print(label)
            fg.add_recipe(label, recipe)
        return fg


    """ --- START OF OBJECT METHODS --- """

    def __init__(self, labels:list, data:list, info:list, meta:dict={}):
        # Most of the functionality of this class is driven by its maintenance
        # of the FeatureGrid representing its bands
        self._fg = FeatureGrid(labels, data, info, meta=meta)
        #for label, recipe in abi_recipes.items():
        #    self._fg.add_recipe(label, recipe)

        # Add abi-specific recipes from recipe_book module

    def to_json(self, indent=None):
        """
        Returns the critical information about this ABIL1b grid as a string
        JSON file. This includes labels, shape, etc so that the full ABIL1b
        object can be recovered given a (M,N,F) shaped unlabeled numpy array
        """
        return self._fg.to_json

    def to_pkl(self, pkl_path:Path, overwrite=False):
        """
        Stores this ABIL1b object as a pkl recoverable by the ABIL1b.from_pkl
        static method.

        :@param pkl_path: Location to save this ABIL1b instance
        :@param overwrite: If True, overwrites pkl_path if it already exits
        """
        self._fg.to_pkl(pkl_path, overwrite)

    '''
    def subgrid(self, labels, target_latlon:tuple, dx_px:int, dy_px:int,
                from_center:bool=False, boundary_error:bool=True, debug=False):
        """
        Return a new ABIL1b
        """
        latlon = np.dstack((self._fg.data("lat"), self._fg.data("lon")))
        yrange,xrange = get_geo_range(
                latlon, target_latlon, dx_px, dy_px,
                from_center, boundary_error, debug)
        new_fg = self._fg.subgrid(labels, yrange, xrange)
        return ABIL1b(
                labels=self._fg.labels,
                data=[X[slice(*yrange),slice(*xrange)] for X in self._data],
                info=self._info
                )
    '''
