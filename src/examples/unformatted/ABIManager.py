#/usr/bin/python

import copy
import datetime as dt
import xarray as xr
import numpy as np
import math as m
import pickle as pkl
import gc
import re
from pathlib import Path
from dataclasses import dataclass

from GeosGeometry import GeosGeometry

class ABIManager:
    ftypes = {
            "noaa_aws":"netCDF files with formats and naming schemes " + \
                    "matching the NOAA AWS S3 bucket standard."
            }
    def __init__(self):
        """
        Initialize ABIManager to handle a chronological xarray Dataset of
        netCDF datasets with a uniform spatial domain (lat/lon coordinates
        for all data are extracted from the first file).

        Time parsing relies on the file naming scheme, so the ftype much
        match a supported scheme. See the ftypes static property.
        """
        # GeosGeometry object representing the spatial domain of the provided
        # data. We make the assumption that the lat/lon or fixed-grid
        # coordinates associated with the provided file apply to all files
        # loaded at init.
        self._geom = None
        # DataArray representing a chronological lat/lon grid
        self._data = None
        self._label = None

    @property
    def geom(self):
        """
        Returns the GeosGeometry object describing this ABIManager's data.
        """
        return self._geom

    @property
    def data(self):
        """
        Returns the ABIManager-style xarray value stored in the data attribute
        """
        return self._data

    def _get_geom(self, dataset:xr.Dataset):
        """
        Returns a GeosGeometry object based on the provided xarray Dataset,
        which is parsed from a GOES-style netCDF.
        """
        _proj = dataset.goes_imager_projection
        sa_ew, sa_ns = np.meshgrid(dataset.x.data, dataset.y.data)
        return GeosGeometry(
            # Nadir longitude
            lon_proj_origin=_proj.longitude_of_projection_origin,
            e_w_scan_angles=sa_ew, # Horizontal FGC (m)
            n_s_scan_angles=sa_ns, # Vertical FGC (m)
            satellite_alt=_proj.perspective_point_height, # radius (m)
            # Earth spheroid equitorial radius (m)
            r_eq=_proj.semi_major_axis,
            # Earth spheroid polar radius (m)
            r_pol=_proj.semi_minor_axis,
            sweep=_proj.sweep_angle_axis,
            )

    @staticmethod
    def dt64_to_datetime(dt64:np.datetime64):
        """
        Convert numpy datetime64 to datetime vie epoch time.
        numpy datetime64s provide epoch time in integer attoseconds,
        so need to divide by 1e9 to convert to decimal epoch in seconds.
        """
        return dt.datetime.utcfromtimestamp(int(dt64)/1e9)

    def load_netCDFs(self, nc_paths:list, ftype:str="noaa_aws",
                     dataset_label:str, field:str, lat_range:tuple=None,
                     lon_range:tuple=None, buffer_arrays:bool=False,
                     buffer_dir:Path=None, buffer_keep_pkls:bool=False,
                     buffer_append:str="", convert_Tb:bool=False,
                     convert_Ref:bool=False, stride:int=1, _debug:bool=False):
        """
        Loads many chronological netCDF files as xarray DataArrays in
        a xarray Dataset

        :param nc_paths: list of Path objects to existing netCDF files.
        :param ftype: netCDF file format identifier
        :param dataset_label: attribute label for this data series.
        :param field: netCDF scalar field to parse (for example "Rad").
        :param lat_range: (lat_i, lat_f) tuple of initial and final latitude
                values in degrees. Defaults to full size
        :param lon_range: (lon_i, lon_f) tuple of initial and final longitude
                values in degrees. Defaults to full size
        :param buffer_arrays: If True, loaded datasets will be buffered as pkls
                in the provided buffer_dir in order to save on memory. If this
                option is True, buffer_dir must be provided.
        :param buffer_dir: Directory where buffer pkls of DataArrays are stored
        :param buffer_keep_pkls: If True, buffer pkls for each netCDF are
                kept in the buffer directory instead of being deleted.
        :param buffer_append: String to append to the end of buffered pkl
                file names. Cannot contain any underscores
        :param convert_Tb: If True, uses planck function coefficients stored
                in valid netCDF files to convert scalar values to brightness
                temperatures. If this isn't an infrared band converting to
                brightness temperatures doesn't make sense, so the program
                will exit with an error.
        :param convert_Ref: If True, uses kappa coefficient to convert
                radiance to reflectance. This is only valid for ABI bands 1-7
        :param stride: Subset the data by choosing a pixel at the given
                interval.  GOES-R series data can be reasonably downscaled in
                this manner since lower-resolution pixels (.5 or 1km) are
                centered on the 2km
        :param _debug: If True, prints useful information as arrays are
                processed.
        """
        self._label = dataset_label
        if "_" in buffer_append:
            ABIManager._FAIL("Buffer append strings cannot contain " + \
                    f"underscores! (provided {buffer_append})")
        if ftype not in self.ftypes.keys():
            ABIManager._FAIL(
                reason="The provided ftype doesn't match any of " + \
                        "the file format options. Valid options " + \
                        "include:\n"+str(list(self.ftypes.keys())))

        # Collect a list of 3-tuples containing the file stime, the
        # reported netCDF time, and 2d ndarrays of scalar radiance
        # values for each netCDF file.
        rad_arrays = []
        if _debug: print("Loading netCDFs.")
        for i in range(len(nc_paths)):
            # Throw an error if the data can't be imported
            nc = nc_paths[i]
            if not nc.exists() or nc.is_dir() or not nc.suffix==".nc":
                ABIManager._FAIL(
                    f"{nc.as_posix()} isn't a valid netCDF path!")

            stime = ABIManager._parse_stime(nc, ftype=ftype)
            if _debug: print(f"Loading data from {nc.name}")
            ds = xr.load_dataset(nc)

            """
            If the data needs to be subset by equal-interval index skipping,
            construct a new dataset with the appropriate axes. I had to make
            some frustrating assumptions here for generality: since Dataset
            data_vars share coordinate arrays, every data_var that reports
            "y" and "x" dimensions is subset along both axes; data_vars not
            reporting both dimensions are ignored, which will be problematic
            if data is only a defined along either the "x" or "y" dimension.
            Furthermore, the indexing order is assumed to be y/x corresponding
            to row/column.
            """
            if stride != 1:
                new_vars = {}
                new_coords = dict(ds.coords)
                new_coords["x"] = new_coords["x"][::stride]
                new_coords["y"] = new_coords["y"][::stride]

                for k in ds.data_vars.keys():
                    # Subset any data_vars along y/x dimensions
                    if set(["y", "x"]).issubset(
                            set(ds.data_vars[k].coords.keys())):
                        new_data = ds.data_vars[k].data[::stride, ::stride]
                        new_vars.update({k:(["y", "x"], new_data)})
                    # Ignore and copy any other data_vars
                    else:
                        new_vars.update({k:ds.data_vars[k]})

                ds = xr.Dataset(
                        data_vars=new_vars,
                        coords=new_coords,
                        attrs=ds.attrs
                        )
                gc.collect()

            nc_time = ABIManager.dt64_to_datetime(ds.coords["t"].values)

            # Initialize a GeosGeometry object describing the geographic
            # properties of the first dataset, which will be used to calculate
            # lat/lon, zenith angles, and viewing angles based on the first
            # netCDF loaded. These geometry values are then used to label
            # the entire series of data being loaded.
            if not self._geom:
                # Get index ranges from the GeosGeometry object
                self._geom = self._get_geom(ds)
                lat_ind_range, lon_ind_range = self._geom.get_subgrid_indeces(
                        lat_range=lat_range,
                        lon_range=lon_range,
                        _debug=_debug)

                if _debug:
                    print(f"Setting coordinates using observation at {stime}")
                    print("raw ds nancount: ",
                          np.count_nonzero(np.isnan(ds["Rad"])))
                    print("raw ds size:     ",ds[field].size)
                    print("raw ds shape:     ",ds[field].shape)

                # Get planck coefficient values from the netCDF if requested.
                if convert_Tb:
                    try:
                        fk1 = ds["planck_fk1"].data
                        fk2 = ds["planck_fk2"].data
                        bc1 = ds["planck_bc1"].data
                        bc2 = ds["planck_bc2"].data
                    except:
                        ABIManager._FAIL("Provided netCDF doesn't have" + \
                            "planck coefficients. Is it actually infrared?")
                elif convert_Ref:
                    try:
                        k0 = ds["kappa0"].data
                    except:
                        ABIManager._FAIL("Provided netCDF doesn't have" + \
                            " kappa coefficients. Is it a reflectance band?")

            # Subset the data to the lat/lon boundaries and format it as a
            # tuple along with its time values.
            ds_subset = ds[field].data[lat_ind_range[0]:lat_ind_range[1]+1,
                                       lon_ind_range[0]:lon_ind_range[1]+1]

            # If requested, convert radiances to brightness temperatures
            # using Planck's function and the previously-parsed coeffs.
            if convert_Tb:
                if _debug: print("Converting to Tb")
                ds_subset = ABIManager.convert_to_Tb(
                        ds_subset, fk1, fk2, bc1, bc2)
            # If requestes, convert radiance values to reflectance by
            # multiplying by the previously-parsed kappa coefficient
            # (See PUG v3 revision 2.2 page 27/28)
            elif convert_Ref:
                if _debug: print("Converting to Reflectance")
                ds_subset = ds_subset * k0

            ds_tuple = (stime, nc_time, ds_subset)
            # If the user wants to buffer arrays as pkls prior to assimilating
            # them into a 3d arrays, load them.
            if buffer_arrays:
                ABIManager._make_buffer_pkl(ds_tuple, buffer_dir, i,
                        self._label, field, stime, buffer_append)
            # Otherwise, just keep a list of the dataset tuples.
            else:
                rad_arrays.append(ds_tuple)
            # Delete the dataset object and run the garbage collector
            # to save memory.
            del ds, ds_subset, ds_tuple
            gc.collect()

        if buffer_arrays:
            rad_arrays = list(ABIManager._load_buffer_pkls(
                    buffer_dir, self._label, field, buffer_append,
                    delete=not buffer_keep_pkls))


        # Construct a 3d DataArray with a chronological time coordinate based
        # on the file stime and lat/lon axes based projected from the
        # fixed-grid coordinate values provided in the first netCDF.
        rad_arrays.sort(key=lambda tup: tup[0])
        nc_data = xr.Dataset(
            data_vars={self._label:( ["x", "y", "time"],
                    np.dstack((tuple(ra[2] for ra in rad_arrays))))
                    },
            coords={
                "lat":( ("x","y"),
                    self._geom.lats[lat_ind_range[0]:lat_ind_range[1]+1,
                         lon_ind_range[0]:lon_ind_range[1]+1]),
                "lon":( ("x","y"),
                    self._geom.lons[lat_ind_range[0]:lat_ind_range[1]+1,
                         lon_ind_range[0]:lon_ind_range[1]+1]),
                "time":[ ra[0] for ra in rad_arrays ],
                },
            attrs={
                "ftype":ftype,
                "field":field,
                "dataset":self._label,
                "lat_range":lat_ind_range,
                "lon_range":lon_ind_range,
                # We're keeping the full lat/lon grid in the _geom attribute,
                # which means it will need to be referenced wrt the lat_range
                # and lon_range of indices to access the requested subset.
                "geom":self._geom,
                "nctime":[ ra[1] for ra in rad_arrays ],
                }
            )
        self._data = nc_data

    @staticmethod
    def convert_to_Tb(rads:np.array, fk1:np.array, fk2:np.array,
                      bc1:np.array, bc2:np.array):
        """
        Use the provided planck constants to convert an ndarray of scalar
        radiance values to brightness temperatures.
        """
        return ( fk2/np.log(fk1/rads+1) - bc1 ) / bc2

    @staticmethod
    def _make_buffer_pkl(data, buffer_dir:Path, index:int, dataset_label:str,
                         field:str, stime:str, append:str="",
                         _debug:bool=False):
        """
        Generate a pkl of a 2d numpy array using the standard naming scheme.
        Generally data is a

        For buffer_dir, dataset_label, and field see the parameter definitions
        in ABIManager.load_netCDFs.

        :param data: Data to load into the pkl. ABIManager.load_netCDFs() uses
                a format like ( stime, nctime, np.ndarray )
        :param append: String to append to the end of the file name. This
                should probably be preceded by an underscore.
        :param _debug: If True, prints when buffer pickles are generated
        """
        if not buffer_dir:
            ABIManager._FAIL(reason="No buffer directory provided!")
        fname = Path(f"{int(stime.timestamp())}_{dataset_label}_{field}" + \
                (f"_{append}" if append else "")+".pkl")
        if _debug: print(f"Generating buffer pickle {fname}")
        with open(buffer_dir.joinpath(fname).as_posix(), "wb") as pklfp:
            pkl.dump(data, pklfp)

    @staticmethod
    def _load_buffer_pkls(buffer_dir:Path, dataset_label:str, field:str,
                          append:str=None, delete:bool=False,
                          _debug:bool=False):
        """
        Loads all buffer pkls from provided buffer_dir that match the provided
        dataset_label and field, and returns them as a tuple of formerly-
        -pickled objects in the order of their stime.

        :param buffer_dir: Directory where pkls are stored
        :param dataset_label: Label matching second underscore-separated field
                of all desired pkls in the buffer_dir
        :param field: Label matching third underscore-separated field of all
                desired pkls in the buffer_dir
        :param delete: If True, successfully loaded buffer pickles are
                immediately deleted.
        :param _debug: If True, prints when pkls are loaded or deleted.
        """
        regex = r"\d+_\w[^_]+_\w+(_\w+)?"
        if not (buffer_dir.exists() and buffer_dir.is_dir()):
            ABIManager._FAIL(f"Buffer directory {buffer_dir} doesn't exist!")

        buffer_pkls = []
        for p in buffer_dir.iterdir():
            # Check if the file name matches the buffer format
            match = re.match(regex, p.name)
            if not match:
                continue
            # Parse field values and check if they match the label/field
            f_split = tuple(p.stem.split("_"))
            if append and len(f_split) == 4:
                f_stime, f_ds_label, f_field, f_append = f_split
                if append != f_append:
                    continue
            elif not append and len(f_split==3):
                f_stime, f_ds_label, f_field = f_split
            else:
                continue
            if not f_ds_label==dataset_label and f_field==field:
                continue
            # Load pkls that are part of the desired dataset
            if _debug: print(f"Loading buffered pickle {p.name}")
            with open(p.as_posix(), "rb") as pklfp:
                buffer_pkls.append((int(f_stime), pkl.load(pklfp)))
            if delete:
                if _debug: print(f"Deleting buffer pickle {p.as_posix()}")
                p.unlink()
        buffer_pkls.sort(key=lambda p: p[0])
        return tuple((b[1] for b in buffer_pkls))

    def index_at_time(self, time:dt.datetime, mode:str="close",
                      nctime:bool=False):
        """
        Returns the index along the time dimension of the time closest to
        the provided UTC datetime.

        :param time: UTC time of the desired array. Defaults to "time"
                dimension, which the ABIManager parses from filename stime.
        :param mode: Specifies how to choose the time index. Options include:
                - "less": closest value less than the provided time
                - "close": closest value to the provided time
                - "more": closest value greater than the provided time
        :param nctime: Use the netCDF-reported time in the "nctime" attribute
                field instead of the stime field "time" extracted from the
                netCDF file name.
        """
        valid_modes = ("less", "close", "more")
        if self._data is None:
            ABIManager._FAIL("Can't find index; no data has been loaded yet!")
        if mode not in valid_modes:
            ABIManager._FAIL(f"{mode} is not a valid mode.\n" + \
                    f"Try one of: {valid_modes}")

        # Collect an array of differences between the requested time
        # and the selected time coordinate axis.
        file_times = self._data.attrs["nctime"] if nctime else list(map(
                        ABIManager.dt64_to_datetime,
                        self._data.coords["time"].values))
        file_epochs = np.asarray(list(map(
                lambda t: t.timestamp(), file_times)))
        target_epoch = time.timestamp()
        diffs = file_epochs-target_epoch

        # Choose the index matching the selected mode.
        if mode=="less":
            negative_diffs = diffs[diffs<=0]
            if not len(negative_diffs):
                raise ValueError(
                    f"There are no times in range less than {time}...  "+ \
                    f"Loaded initial time: {min(file_times)}  " + \
                    f"Loaded final time: {max(file_times)}")
            return np.where(diffs==np.amax(negative_diffs))[0][0]
        elif mode=="more":
            positive_diffs = diffs[diffs>=0]
            if not len(positive_diffs):
                raise ValueError(
                    f"There are no times in range greater than {time}"+ \
                    f"Loaded initial time: {min(file_times)}\n" + \
                    f"Loaded final time: {max(file_times)}")
            return np.where(diffs==np.amin(positive_diffs))[0][0]
        elif mode=="close":
            return np.where(abs(diffs)==np.amin(abs(diffs)))[0][0]

    def array_from_time(self, time:dt.datetime, mode:str="close",
                        nctime:bool=False):
        """
        Returns a 2d DataArray corresponding to the provided time.

        :param time: UTC time to select an index near
        :param mode: String argument; one of "less", "close", or "more".
                See ABIManager.index_at_time for the options.
        :param nctime: (optional) Select from nctimes instead of stimes
        """
        return self._data.isel(time=self.index_at_time(time, mode, nctime))

    def load_pkl(self, pkl_path:Path, dataset_label:str=None):
        """
        Load a Dataset pkl that was previously formatted by an ABIManager
        instance.

        :param pkl_path: Path to pickle to load
        :param dataset_label: data_vars label of the datset in the pkl that
                should be loaded. Currently only one data_var can be loaded
                at a time per ABIManager.
        :return self: Although this is an inplace method, returns a reference
                to this ABIManager to allow method cascading.
                For example, it's valid to load a pkl on init like:
                ABIManager().load_pkl(Path("path/to/data.pkl"))
        """
        with open(pkl_path.as_posix(), "rb") as pklfp:
            self._data = pkl.load(pklfp)
        #self._data.attrs = full_dataset.attrs
        self._geom = self._data.attrs["geom"]
        self._label = dataset_label

        labels = list(self._data.data_vars.keys())
        if len(labels)==1:
            if dataset_label in labels:
                self._label = dataset_label
            elif dataset_label is None:
                self._label = labels[0]
        else:
            raise ValueError(f"invalid key: {dataset_label}; " + \
                    "options include: {labels}")
        return self

    def make_pkl(self, pkl_path:Path):
        """
        Serialize the data into a pkl file at the provided path.

        :pkl_path: Path object where pickle should be written
        """
        print(f"Generating pickle {pkl_path.as_posix()}")
        with open(pkl_path.as_posix(), "wb") as pklfp:
            pkl.dump(self._data, pklfp)

    def solar_zenith_angles(self, utc_datetime:dt.datetime,
                            geom:GeosGeometry=None):
        """
        Generates and returns solar zenith angles at the provided UTC
        datetime for the lat/lon ranges represented by the provided geometry.

        :param utc_datetime: Time at which zenith angles are calculated for
                each grid point
        :param geom: Geometry object, defaulting to the object attribute.
        :return Solar zenith angle array in degrees:
        """
        geom = geom if geom else self._geom
        return astronomy.sun_zenith_angle(utc_datetime, geom.lons, geom.lats)

    def restrict_data(self, dataset_label:str, bounds:list=[None, None],
                      copy:bool=False, replace_val:list=[None, None]):
        """
        Restrict data values to the provided inclusive boundaries.

        :param dataset_label: The xarray data_var label of the data to subset.
        :param bounds: 2-list like [lower, upper] describing the desired
                scalar boundaries of the data array. If only a lower bound or
                an upper bound is needed, use None in the list instead of
                the unrestricted boundary.
        :param copy: Return a new ABIManager with the constrained data
                instead of replacing the data in this ABIManager.
        :param replace_val: 2-list of values like [lower, upper] that will
                replace any data that is out of bounds. If replace_val is None
                (or default), replaces all out-of-bounds values with the
                "bounds" values.

                NOTE: if you want to replace with null-like values, use
                      np.nan instead of None.
        """
        if copy:
            arr_copy = copy.deepcopy(self._data)
            self._data = None
            gc.collect()
        else:
            arr_copy = self._data

        if len(bounds)!=2:
            ABIManager._FAIL(reason="Data boundaries must be a 2-list "+ \
                    f"(provided {bounds})")
        # Establish usable data boundaries and replacements.
        bounds[0] = bounds[0] if not bounds[0] is None \
                else np.amin(arr_copy[dataset_label].data)
        bounds[1] = bounds[1] if not bounds[1] is None \
                else np.amax(arr_copy[dataset_label].data)

        replace_val[0] = replace_val[0] if not replace_val[0] is None \
                else bounds[0]
        replace_val[1] = replace_val[1] if not replace_val[1] is None \
                else bounds[1]

        # Restrict upper and lower values, replacing out-of-range data
        # with the provided replacement value or default.
        arr_copy = arr_copy.where(
                arr_copy[dataset_label]>=bounds[0], replace_val[0])
        arr_copy = arr_copy.where(
                arr_copy[dataset_label]<=bounds[1], replace_val[1])

        if copy:
            # If the user wants a new ABIManager, initialize and return it
            am = ABIManager().from_data_and_geom(self, arr_copy, geom)
            return am
        # Otherwise, replace the data attribute and run the garbage collector.
        self._data = arr_copy
        del arr_copy
        gc.collect()
        return None


    def from_data_and_geom(self, data:xr.Dataset, geom:GeosGeometry,
                           dataset_label:str):
        """
        Loads the provided data and geometry into this ABIManager's attributes.
        This can be used to initialize an ABIManager object instead of loading
        from a pkl or from netCDFs.

        :param data: Dataset derived from another ABIManager object.
        :param geom: GeosGeometry object describing the provided data.
        """
        if dataset_label not in self._data.data_vars.keys():
            raise ValueError(f"Dataset label {dataset_label} isn't " + \
                    "in the provided data's data_vars! The provided data" + \
                    f" has these variables: {data.data_vars.keys()}")
        self._data = data
        self._geom = geom
        self._label = dataset_label
        return self

    @staticmethod
    def _parse_stime(ncfile:Path, ftype:str):
        """
        Parse the observation start time from the netCDF file path

        :param ncfile: File path of the netCDF to parse
        :param ftype: File format type. Must be supported by this method.
        """
        # List of supported file types
        if ftype=="noaa_aws":
            # In the NOAA file naming scheme for GOES netCDF, the start
            # time is a UTC YYYYjjjHHMMSSf formatted string in the 3rd-to-last
            # field separated by underscores, and starts with a 's'.
            sformat="%Y%j%H%M%S%f"
            timestring = ncfile.stem.split("_")[-3].replace("s","")
            return dt.datetime.strptime(timestring, sformat)

    def replace_nans(self, replace_value=0):
        """
        Replace all nan values in the dataset with the provided value
        """
        self._data.data = np.nan_to_num(self._data.data, replace_value)

    @staticmethod
    def _FAIL(reason:str):
        """
        Generic failure that doesn't raise an exception traceback (exits 1).
        """
        print(f"ABIManager failed at runtime! Here's why:\n\n{reason}")
        exit(1)

if __name__=="__main__":
    # Collect paths to all netCDF files in the data directory
    datadir = Path("/home/krttd/uah/22.f/aes572/hw3/data/great_lakes/band02")
    #datadir = Path("/home/krttd/uah/22.f/aes572/hw3/data/great_lakes/band07")
    sorted_files = list(datadir.iterdir())
    sorted_files.sort()

    #pkl_path = Path("/home/krttd/uah/22.f/aes572/hw3/data/pkls/michigan_bt_band07.pkl")
    #pkl_path = Path("/home/krttd/uah/22.f/aes572/hw3/data/pkls/supwater_bt_band07.pkl")
    #pkl_path = Path("/home/krttd/uah/22.f/aes572/hw3/data/pkls/supwater_vis_band02.pkl")
    pkl_path = Path("/home/krttd/uah/22.f/aes572/hw3/data/pkls/supwater_vis_band02-stride4.pkl")

    buffer_dir = Path("/home/krttd/uah/22.f/aes572/hw3/data/buffer")

    dataset_label = "g16b02"
    #dataset_label = "g16b07"

    AM = ABIManager()

    #"""
    # load subsets of a list of chronological netCDF files and make a pkl file
    # to store the complete dataset as a DataArray

    #"""
    # Load netCDFs in a specified geographic range.
    #geo_center = (42, -85)
    geo_center = (42, -87.2)
    #aspect = (8, 12)
    aspect = (1.6, 2.4)
    stride = 4

    #"""
    AM.load_netCDFs(
            nc_paths=sorted_files,
            #nc_paths=sorted_files[45:55],
            #nc_paths=sorted_files[35:65],
            ftype="noaa_aws",
            dataset_label=dataset_label,
            field="Rad",
            lat_range=(geo_center[0]-aspect[0]/2, geo_center[0]+aspect[0]/2),
            lon_range=(geo_center[1]-aspect[1]/2, geo_center[1]+aspect[1]/2),
            buffer_arrays=True,
            buffer_dir=buffer_dir,
            buffer_keep_pkls=False,
            buffer_append="full",
            convert_Tb=False,
            stride=stride,
            _debug=True,
           )
    #"""
    #AM.replace_nans()
    AM.make_pkl(pkl_path)
    #AM.load_pkl(pkl_path, dataset_label=dataset_label)
    #print(AM.data)

    #tmp_latlon = { "lats": AM.geom.lats, "lons": AM.geom.lons }
    #with open("data/latlons.pkl", "wb") as pklfp:
    #    pkl.dump(tmp_latlon, pklfp)

    """
    # Testing constraining data to boundaries.
    restricted_data_path = Path(
            "/home/krttd/uah/22.f/aes572/hw2/data/pkls/bt_b02_BigPicture-restricted.pkl")
    AM.restrict_data(
            bounds=[273, None],
            copy=False,
            replace_val=[None, None]
            )
    AM.make_pkl(restricted_data_path)
    """
    """
    #AM.load_pkl(restricted_data_path)
    print(f"Data nan count: {np.count_nonzero(np.isnan(AM.data.data))}")
    print(f"Data size:      {AM.data.size}")
    print(f"Width (px): {AM.data['lon'].size}")
    print(f"Height (px): {AM.data['lat'].size}")

    print(f"lat range: ({min(AM.data.coords['lat'])}, " + \
            f"{max(AM.data.coords['lat'])}")
    print(f"lat size: {AM.data.coords['lat'].size}")
    print(f"lon range: ({min(AM.data.coords['lon'])}, " + \
            f"{max(AM.data.coords['lon'])}")
    print(f"lon size: {AM.data.coords['lon'].size}")

    """
    """
    print(AM.data)
    print(AM.data.coords)
    print(AM.data.dims)
    print(f"time range: ({min(AM.data.coords['time'])}, " + \
            f"{max(AM.data.coords['time'])}")
    print(f"time size: {AM.data.coords['time'].size}")
    """

    """
    sample_time = dt.datetime(year=2021, month=6, day=29, hour=21, minute=40)
    print(f"closest index at {sample_time}: " + \
            f"{AM.index_at_time(sample_time, mode='close')}")
    print(AM.array_from_time(sample_time))
    """
