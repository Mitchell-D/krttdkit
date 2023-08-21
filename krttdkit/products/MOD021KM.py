"""
Powerfully abstract class for downloading, analyzing, and generating
recipes with MODIS L1b data from the LAADs DAAC.

(the bands I use most often)
band  range          reason
3     459-479nm      blue
4     545-565nm      green
1     620-670nm      near-red
16    862-877nm      NIR / aerosol distinction
19    916-965nm      H2O absorption
5     1230-1250nm    optical depth
26    1360-1390nm    cirrus band
6     1628-1652nm    snow/ice band
7     2106-2155nm    cloud particle size
20    3660-3840nm    SWIR
21    3929-3989      another SWIR
27    6535-6895nm    Upper H2O absorption
28    7175-7475nm    Lower H2O absorption
29    8400-8700nm    Infrared cloud phase, emissivity diff 11-8.5um
31    10780-11280nm  clean LWIR
32    11770-12270nm  less clean LWIR
33    14085-14385nm  dirty LWIR
"""

from pathlib import Path
from datetime import datetime as dt
from datetime import timedelta as td
from pprint import pprint as ppt
from copy import deepcopy
import multiprocessing as mp
import numpy as np
import pickle as pkl

from krttdkit.acquire import laads, modis
from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp
from krttdkit.visualize import TextFormat as TFmt
from krttdkit.operate import classify, geo_helpers, Recipe
from krttdkit.operate import enhance as enh
from krttdkit.products import PixelCat

class MOD021KM:
    """
    Wrapper class for applying recipes to equal-resolution grids of MODIS l1b
    data. Stores each band of data as a 2d array with uniform shape, and
    allows for easy application of RGB recipes,

    Raw band data indeces are internally stored as integers, and other data
    is referenced with a string key. Several RGB recipes are supported by
    default, but new recipes can be added with a list of keys that
    correspond to arguments to a provided function.

    This class fundamentally relies on a standard format of data consisting of
    a 4-tuple (data, info, geo, sunsat) such that:

     - data: n-tuple of (MxN) 2d arrays for a uniform height/width (M,N)
     - info: dict with information about the stored arrays, coefficients, units
     - geo: 3-tuple of (MxN) 2d arrays like (lat, lon, height). Height can
       usually be passed as "None"
     - sunsat: 4-tuple of (MxN) 2d arrays like (sza, saa, vza, vaa) where
       sza and saa are the solar zenith/azimuth angles and vza/vaa are
       the viewing zenith/azimuth angles.

    Ideas:
     - quick-render RGB recipes without making new arrays, optionally
       save the images.
     - Save data with user-applied parameters as new labeled array

    Hopefully this will develop into a generalized parent class in the future.
    """
    valid_bands = {i for i in range(1,34)}
    @staticmethod
    def ctr_wl(band:int):
        return modis.band_to_wl(band)
    @staticmethod
    def from_pkl(pkl_path:Path):
        """ Load a pkl with the standard format, returned as new object """
        return MOD021KM(*pkl.load(pkl_path.open("rb")))
    @staticmethod
    def find_granule(target_latlon:tuple=None, satellite:str=None,
                     target_time:dt=None, day_only:bool=True,
                     time_radius:td=None, return_in_range:bool=False,
                     debug:bool=False):
        """
        Query the LAADS DAAC for MOD021KM data.

        :@param target_latlon: lat/lon coordinates that must be included in
            the array. Defaults to no geographic constraint.
        :@param satellite: "terra" or "aqua". If None, queries both satellites.
        :@param target_time: Closest time to look for a pass filling all
            the other parameters. Defaults to now-time_radius
        :@param day_only: Only returns daytime passes. Defaults to True.
        :@param time_radius: Amount of time before and after target_time to
            search for passes. Defaults to 1 day.
        :@param return_in_range: Returns all products in range of
            target_time +/- time_radius instead of just the closest product
            to target_time.
        """
        valid_sats = []
        if satellite is None:
            sats = ("terra", "aqua")
        elif satellite in ("terra", "aqua"):
            sats = (satellite,)
        else:
            raise ValueError(f"satellite must be one of {valid_sats}")
        time_radius = time_radius if time_radius else td(days=1)
        ctr_time = (target_time, dt.utcnow()-time_radius)[target_time is None]

        # Query the LAADS DAAC for products fitting the query
        products = []
        for s in sats:
            product_key = ("MOD021KM", "MYD021KM")[s=="aqua"]
            products += modis.query_modis_l1b(
                product_key=product_key,
                start_time=ctr_time-time_radius,
                end_time=ctr_time+time_radius,
                latlon=target_latlon,
                day_only=day_only,
                debug=debug
                )

        # If the user wants the full range, return as a list. Otherwise,
        # select the product clusest to the center time.
        if return_in_range:
            return products
        return list(sorted(
            products,
            key=lambda p: abs(ctr_time.timestamp()-p["atime"].timestamp())
            ))[0]
        return closest_product

    @staticmethod
    def download_granule(data_dir:Path, raw_token:str,
                         target_latlon:tuple=None, replace:bool=True,
                         satellite:str=None, target_time:dt=None,
                         day_only:bool=True, time_radius:td=None,
                         debug:bool=False):
        """
        Downloads a MODIS granule HDF fitting the parameters.
        See find_granule for parameter descriptions.

        :@param data_dir: directory to download MODIS hdf(s) to.
        :@param raw_token: API token string for the LAADS DAAC.
        :@param replace: If True, replaces existing HDFs with the same name.
        :@return: Path object to the downloaded l1b HDF file.
        """
        gran = MOD021KM.find_granule(target_latlon, satellite, target_time,
                     day_only, time_radius, return_in_range=False, debug=debug)
        return laads.download(gran["downloadsLink"], data_dir, replace=replace,
                              raw_token=raw_token, debug=debug)
    @staticmethod
    def from_hdf(l1b_hdf:Path, l1b_bands:tuple):
        """
        Returns a new full-resolution MOD021KM object from the provided l1b
        HDF file, probably acquired from the LAADS DAAC.
        """
        return MOD021KM(*modis.get_modis_data(l1b_hdf, bands=l1b_bands))

    def mask_to_color(array:np.ndarray, mask:np.ndarray, color:tuple=(1,0,0),
                      radius:int=0):
        """
        Applies an RGB color to a (M,N,3) RGB or (M,N) scalar array where
        the provided mask is True. If the provided array is (M,N)-shaped,
        the returned array will be an (M,N,3) RGB.

        :@param array: (M,N) scalar or (M,N,3) RGB array. Values are normalized
            from minimum to maximum in the returned RGB.
        :@param mask: (M,N) shaped boolean mask, True where pixels should be
            marked with the provided color.
        :@param color: 3-tuple of [0,1] RGB float values for the color to
            mark True-masked pixels with. Defaults to RED.

        :@return: (M,N,3) shaped RGB array of [0,1] float values with masked
            pixels marked using the provided color.
        """
        assert len(mask.shape) == 2
        assert tuple(mask.shape[:2]) == tuple(array.shape[:2])
        assert len(color) == 3
        assert all([0<=v<=1 for v in color])
        assert radius >= 0
        if len(array.shape) == 2:
            array = np.dstack([array for i in range(3)])
        array = enh.linear_gamma_stretch(array)
        if not radius:
            array[np.where(mask)] = np.asarray(color)
            return array
        for ji in zip(*np.where(mask)):
            array = enh.linear_gamma_stretch(gt.label_at_index(
                (array*255).astype(np.uint8),ji,size=radius,
                color=[255*c for c in color]))
        return array

    @staticmethod
    def mask_to_idx(mask:np.ndarray, samples:int=None):
        """
        Converts a 2d boolean mask to a a list of (i,j) indeces of True values.
        """
        mpx = tuple(zip(*np.where(mask)))
        spx = []
        return [ mpx[j] for j in np.random.choice(len(mpx), size=samples) ]

    @staticmethod
    def idx_to_mask(samples:list, shape:tuple):
        """
        Converts a list of (j,i) integer pixel indeces to a 2d boolean mask
        array with the provided shape
        """
        mask = np.full(shape, False, dtype=bool)
        mask[tuple(map(np.asarray,zip(*samples)))] = True
        return mask
        '''
        for j,i in samples:
            mask[j,i] = True
        return mask
        '''

    @staticmethod
    def ints_to_masks(intarr:np.ndarray):
        """
        Converts an array of [0,M) integers representing M classes to a list
        of boolean truth arrays for the corresponding classes, each boolean
        array having the same shape as the input array.
        """
        intarr = intarr.astype(int)
        masks = [np.full_like(intarr, False, dtype=bool)
                 for i in range(np.amax(intarr)+1)]
        for j in range(intarr.shape[0]):
            for i in range(intarr.shape[1]):
                masks[intarr[j,i]][j,i] = True
        return masks

    def __init__(self, data:tuple, info:dict,
                 geo:tuple=None, sunsat:tuple=None):
        """
        Minimally provide data tuple and info dict as returned by
        aes670hw2.modis.get_modis_data() with L1b files.
        """
        # Equal-size MODIS data arrays; immutable
        if not len(data):
            raise ValueError(f"data parameter needs at least 1 2d array.")
        # Equal-size MODIS data arrays; immutable
        self._shape = tuple(data[0].shape)
        self._data = tuple([ self._valid_array(X) for X in data ])

        # Assure all arguments are the valid types and have a nonzero length.
        self._info = tuple(info)
        self._geo = tuple(geo) if geo else None
        self._sunsat = tuple(sunsat) if sunsat else None

        # Dictionary of data generated with a recipe. Since the underlying
        # data can't change you should never have to re-generate one of
        # the recipe arrays. If you need to enhance, make a new recipe/label.
        self._recipe_data = {
                "sza":self._sunsat[0],
                "saa":self._sunsat[1],
                "vza":self._sunsat[2],
                "vaa":self._sunsat[3],
                "lat":self._geo[0],
                "lon":self._geo[1],
                "height":self._geo[2],
                }
        self._rgb_recipe_data = {}

        # available MODIS band numbers; immutable
        self._bands = tuple([info[i]["band"] for i in range(len(self._data))])

        # RGB recipes are assumed to return a (M,N) array
        self._scalar_recipes = {
                "ndvi":Recipe(
                    args=(1,2),
                    func=lambda A,B:(B-A)/(B+A),
                    name="Normalized Difference Vegetation Index",
                    ),
                "ndsi":Recipe(
                    args=(4,6),
                    func=lambda A,B:(A-B)/(A+B),
                    name="Normalized Difference Snow Index",
                    ),
                "ndwi":Recipe(
                    args=(2,7),
                    func=lambda nir,swir:enh.linear_gamma_stretch(
                        (nir-swir)*(nir+swir)),
                    name="Normalized Difference Water Index",
                    ),
                "nddi":Recipe(
                    args=("ndvi","ndwi"),
                    func=lambda ndvi,ndwi:(ndvi-ndwi)/(ndvi+ndwi),
                    desc=("ndvi", "ndwi"),
                    name="Normalized Difference Drought Index",
                    ref="https://doi.org/10.1029/2006GL029127"
                    ),
                }
        # RGB recipes are assumed to return a (M,N,3) array
        self._rgb_recipes = {
                "TCeq":Recipe(
                    args=(1,4,3),
                    func=lambda r,g,b: np.dstack([
                        enh.linear_gamma_stretch(
                            enh.histogram_equalize(X, 512)[0])
                        for X in [r,g,b]]),
                    name="Equalized Truecolor RGB"
                    ),
                "TC":Recipe(
                    args=(1,4,3),
                    func=lambda r,g,b:np.dstack([enh.linear_gamma_stretch(X)
                                                   for X in [r,g,b]]),
                    name="True-color RGB",
                    desc=("red","green","blue"),
                    ),
                "DUST":Recipe(
                    args=(29,31,32),
                    func=lambda r,g,b:np.dstack(list(map(
                        enh.linear_gamma_stretch, [ b-g, g-r, g ]))),
                    desc=("depth","phase","temp"),
                    name="Dust RGB",
                    ref="https://weather.ndc.nasa.gov/sport/training/quickGuides/rgb/QuickGuide_DustRGB_MSG_NASA_SPoRT.pdf"
                    ),
                "DCP":Recipe(
                    args=(26,1,6),
                    func=lambda r,g,b:np.dstack([enh.linear_gamma_stretch(X)
                                                   for X in [r,g,b]]),
                    name="Day cloud phase RGB",
                    desc=("cirrus","red","phase"),
                    ),
                "FIRE":Recipe(
                    args=(1, 20, 31),
                    func=lambda r,g,b: np.dstack(list(map(
                        enh.linear_gamma_stretch,
                        [ g-b, b, 1-enh.linear_gamma_stretch(r) ]))),
                    name="Fire RGB",
                    desc=("swlwdiff","lwir","red"),
                    ),
                "DLC":Recipe(
                    args=(6, 2, 1),
                    func=lambda r,g,b: np.dstack(list(map(
                        lambda X: np.clip(X,0,1), [r,g,b]))),
                    name="Day land cloud RGB",
                    desc=("smallwater","iceveg","ice"),
                    ref="https://weather.ndc.nasa.gov/sport/training/quickGuides/rgb/QuickGuide_DayLandCloudRGB_GOESR_NASA_SPoRT.pdf",
                    ),
                "CUSTOM":Recipe(
                    args=(31,"ndvi", 29),
                    func=lambda lwir,ndvi,emiss:np.dstack([
                        enh.linear_gamma_stretch(lwir),
                        enh.linear_gamma_stretch(ndvi),
                        1-enh.linear_gamma_stretch(emiss),
                        ]),
                    name="Custom RGB",
                    desc=("lwir","ndvi","emiss"))
                }

    @property
    def rgb_recipes(self):
        """ Returns a copy of the dictionary of loaded RGB Recipe objects """
        return self._rgb_recipes

    @property
    def recipes(self):
        """ Returns a copy of the dictionary of loaded 2d Recipe objects """
        return self._scalar_recipes

    @property
    def shape(self):
        """ returns the list of data band numbers """
        return self._shape

    @property
    def bands(self):
        """ returns the list of data band numbers """
        return self._bands

    @property
    def labels(self):
        """
        returns a tuple of all labels currently supported, including string
        labels for recipes and band number labels, which should be
        interchangable as recipe function arguments.
        """
        return list(set(list(self._scalar_recipes.keys()) + \
                list(self._bands) + list(self._recipe_data.keys()) + \
                list(self._rgb_recipe_data.keys())))

    def radiance(self, band:int):
        """
        Returns the band array converted to radiance instead of brightness
        temperature or reflectance.
        This is a lossy process for brightness temperature since it requires
        un-inverting the planck function of radiance calculated after download.
        """
        if self.info(label)["is_reflective"]:
            counts = self.data(label)*np.cos(np.deg2rad(self._sunsat[0])) \
                    / self.info(label)["ref_scale"] + self.info("ref_offset")
            rad = (counts-self.info(label)["rad_offset"]) \
                    * self.info["rad_scale"]
        else:
            T = self.data(label)
            wl = self.info(label)["ctr_wl"]
            c1 = 1.191042e8 # W / (m^2 sr um^-4)
            c2 = 1.4387752e4 # K um
            rad = c1/(wl**5 * (np.exp(c2/(wl*T)-1)))
        return rad

    def raw_reflectance(self, band:int):
        """
        returns the raw reflectance array of the provided reflectance band,
        ie the observed reflectance not scaled by the solar zenith angle.
        """
        if not self.info(band)["is_reflective"]:
            raise ValueError(f"Band {band} isn't a reflectance band!")
        return np.cos(np.deg2rad(self.data("sza")))*self.data(band)

    def area(self, mask=None, nadir_merid:float=1, nadir_zonal:float=1):
        """
        Returns the total area of this subgrid's domain based on viewing zenith
        angle and nadir pixel dimensions. If a 2d boolean mask is provided,
        returns only the sum of the areas of the True pixels.

        :@param mask: 2d boolean array with the same shape as this subgrid.
            True values correspond to pixels counted in the returned area.
        :@param nadir_merid: Meridional (y) width of the nadir pixel.
        :@param nadir_merid: Zonal (x) width of the nadir pixel.
        """
        if mask is None:
            mask = np.full(self.shape, True, dtype=bool)
        assert mask.shape == self._shape
        distortion = (np.cos(np.deg2rad(
            self.data("vza")[np.where(mask)])))**(-3)
        return np.sum(nadir_merid*nadir_zonal*distortion)

    def data(self, label, choose_contrast:bool=False,
             choose_gamma:bool=False, _rdepth:int=0):
        """
        Functionally evaluates and returns a copy of the array associated with
        the requested data band or scalar recipe referenced by the band integer
        or recipe string label.

        This method is intentionally limited to 2d bands and recipes. If you
        need to evaluate an RGB recipe, see MOD021KM.get_rgb().

        This method can functionally evaluate the "call tree" of a scalar
        recipe that involves composition of multiple other scalar recipes
        referenced by strings. It limits recursive depth of function calls
        to 5, so if a function has too many embedded calls to another recipe
        in it's call tree, it may fail.

        NOTE: if choose_gamma or choose_contrast are True, the returned data
        will be normalized from [min,max] to [0,1].

        :@param label: integer of loaded band or string label of loaded Recipe.
        :@param choose_contrast: If True, prompts the user to choose a scalar
                contrast value using an interactive window and returns the
                array resulting from the value they choose.
        :@param choose_gamma: If True, prompts the user to choose a scalar
                gamma exponent value using an interactive window and returns
                the array resulting from the value they choose.
        :@param _rdepth: Recipes are allowed to nest, so there is danger of
                accidentally creating a recursive loop. Here we keep a
                count of the recursive depth of a call and keep it under 5.
        """
        self._validate_label(label)
        if _rdepth>5:
            raise RuntimeError(f"Exceeded recursive depth of 5")
        if type(label) is str and label not in self._recipe_data.keys():
            try:
                self._recipe_data[label] = self._scalar_recipes[label].func(
                        *tuple([self.data(L, _rdepth+1) for L in \
                                self._scalar_recipes[label].args]))
            except RuntimeError:
                raise RuntimeError(f"Exceeded maximum recursive depth " + \
                        "while getting data: {label}")

        if type(label) is str:
            data = self._recipe_data[label]
        else:
            data = self._data[self._b2idx(label)]
        if not choose_contrast or not choose_gamma:
            return data
        data = enh.linear_gamma_stretch(data)
        pc = PixelCat([data], [label])
        if choose_contrast is True:
            print(TFmt.WHITE(f"Choose saturating contrast for {label}",
                  bold=True))
            pc.pick_linear_contrast(label, set_band=True)
        if choose_gamma is True:
            print(TFmt.WHITE(f"Choose gamma for {label}",bold=True))
            pc.pick_gamma(label, set_band=True)
        print("returning Pixel Cat")
        return pc.band(label)

    def add_data(self, label:str, data:np.ndarray):
        """
        Add a labeled scalar array to the data stored by this subgrid
        so that it can be referenced with its label string
        """
        assert data.shape == self.shape
        assert label not in self.labels
        self._recipe_data[label] = data

    def add_rgb_data(self, label:str, data:np.ndarray):
        """
        Add a labeled scalar array to the data stored by this subgrid
        so that it can be referenced with its label string
        """
        assert data.shape == (*self.shape,3)
        assert label not in self.labels
        self._rgb_recipe_data[label] = data


    def values(self, band, mask:np.ndarray=None):
        """
        Returns the MASKED data values in the requested band as a flattened
        array. If no mask is provided, returns all data values as a flat array.

        If band is an interable of scalar recipes or available band numbers,
        returns a list of all flattened masked data values for the array
        with corresponding indeces
        """
        # If band is iterable, return a stacked array of values for all bands
        if hasattr(band, "__iter__") and not isinstance(band, str):
            return [self.values(b) for b in list(band)]
        if band not in self.labels:
            raise ValueError(f"{band} is not a band or label in {self.labels}")
        array = self.data(band)
        if mask is None:
            mask = np.full_like(array, True, bool)
        return array[np.where(mask)].shape

    def add_rgb_recipe(self, label:str, rgb_recipe:Recipe):
        """ Add a new scalar recipe """
        if label in self.rgb_recipes.keys():
            raise ValueError(f"Label {label} is already a scalar recipe!")
        self._rgb_recipes.update({label:rgb_recipe})

    def add_recipe(self, label:str, recipe:Recipe):
        """ Add a new scalar recipe """
        if label in self._scalar_recipes.keys():
            raise ValueError(f"Label {label} is already a scalar recipe!")
        self._scalar_recipes.update({label:recipe})

    def _validate_rgb_label(self, rgb_label:str):
        """
        Verifies that the provided argument is a valid string label for
        the Recipe of a loaded RGB.
        """
        rgb_keys = list(self._rgb_recipes.keys())
        if not rgb_label in rgb_keys:
            raise ValueError(
                    f"RGB label {rgb_label} is not one of {rgb_keys}")
        return rgb_label

    def _validate_label(self, band_or_label):
        """
        Verifies that the provided argument is either a valid band int or a
        string label for a supported scalar recipe
        """
        if not band_or_label in self.labels:
            raise ValueError(
                    f"Label {band_or_label} is not one of {self.labels}")
        return band_or_label

    def get_rgb(self, label, as_pixelcat:bool=False, choose_gamma:bool=False,
                choose_contrast:bool=False, gamma_scale:float=1.):
        """
        Evaluates a loaded RGB recipe referenced by label and returns a (M,N,3)
        array copy of the RGB.

        :@param label: valid string label for the loaded RGB recipe to generate
        """
        if label in self._rgb_recipe_data.keys():
            rgb =  np.copy(self._rgb_recipe_data[label])
            if label in self._rgb_recipes.keys():
                args = self._rgb_recipes[label].args
            else:
                args = ["RED", "GREEN", "BLUE"]

        else:
            self._validate_rgb_label(label)
            f = self._rgb_recipes[label].func
            args = self._rgb_recipes[label].args
            rgb = f(*[self.data(a) for a in args])
        if not as_pixelcat and not choose_gamma:
            return rgb
        rgb_pc = PixelCat([rgb[:,:,i] for i in range(rgb.shape[2])], args)
        if choose_contrast:
            for l in args:
                print(TFmt.WHITE(f"Choose saturating contrast for {label}"))
                rgb_pc.pick_linear_contrast(l, set_band=True)
        if choose_gamma:
            for l in args:
                print(TFmt.WHITE(f"Choose gamma for {label}"))
                rgb_pc.pick_gamma(l, gamma_scale=gamma_scale, set_band=True)
        if as_pixelcat:
            return rgb_pc
        return rgb_pc.get_rgb(args, normalize=True)

    def from_recipe(self, recipe_name:str):
        """
        Returns an RGB or 2d scalar array with a Recipe that has already been
        loaded into this MOD021KM object, and subsequently updates the internal
        storage of recipe-generated arrays.
        """
        assert (recipe_name in self._rgb_recipes.keys() or \
                recipe_name in self._scalar_recipes.keys())

    def __repr__(self):
        rdict = {
                "bands":', '.join(map(str,self._bands)),
                "nbands":len(self._data),
                "shape":self._shape,
                "geo":not self._data is None,
                "sunsat":not self._sunsat is None,
                }
        bstr = ""
        stats = ["\n    ".join([f"{k}:{float(v):.3f}"
                           for k,v in enh.array_stat(X).items()
                           if k != "shape"])+"\n"
                 for X in self._data]
        for i in range(len(self._bands)):
            bstr+=f"\n\nBand {self._bands[i]} " + \
                    f"({self._info[i]['ctr_wl']:.3f}um):\n    {stats[i]}"
            #bstr+=stats[i]
        rstr = "\n".join([f"{k:8}:{v}" for k,v in rdict.items()])
        return bstr+"\n"+rstr

    def info(self, label=None):
        """ Returns the info dictionary associated with the provided band """
        if label is None:
            return self.__repr__
        if type(label)==int:
            return self._info[self._b2idx(label)]
        return enh.array_stat(self.data(label))

    def _b2idx(self, band:int):
        """
        Helper function to convert MODIS band number to internal array index
        """
        return self._bands.index(self._validate_bands(int(band))[0])

    def _valid_array(self, X:np.ndarray):
        """
        Validates that X is a 2d array that matches the shape of the
        current dataset. If it isn't, raises an error.
        """
        if not X.shape == self._shape:
            raise ValueError(
                    f"Provided array doesn't match shape {self._shape}")
        return X

    def make_pkl(self, pkl_path:Path):
        """
        Generates a 4-tuple pkl with the standard format, identical to the
        output of modis.get_modis_data(): (data, info, geo, sunsat)
        """
        with pkl_path.open("wb") as pklfp:
            pkl.dump((self._data, self._info, self._geo, self._sunsat), pklfp)
            print(f"Generated pkl at {pkl_path.as_posix()}")

    def _validate_bands(self, band):
        """
        Validates that the provided argument is a MODIS band, and
        returns it as a list.

        :@param band: integer MODIS band or list of bands
        """
        bands = band if hasattr(band, "__iter__") else [band]
        invalid = [b for b in MOD021KM.valid_bands
                   if int(b) not in MOD021KM.valid_bands]
        if len(invalid):
            raise ValueError(f"Invalid bands: {invalid}")
        return bands

    def get_closest_pixel(self, target_latlon:tuple):
        """
        Uses euclidean distance to estimate the closest pixel to the provided
        geographic coordinate, and returns the (ax1,ax2) indeces of the pixel
        as a 2-tuple. Wrapper on geo_helpers.get_geo_range method.
        """
        return geo_helpers.get_closest_pixel(
                np.asarray(self._geo[:2]), target_latlon)

    def quick_render(self, label, as_hsv:bool=False, hsv_ranges:dict={}):
        """
        Evaluate and quick-render an RGB recipe or 2d grayscale array with cv2.

        Optionally use cv2 to render an RGB of 2d data arrays by scaling the
        hue to data coordinates.

        You can provide HSV range keyword arguments with hsv_ranges such that
        all ranges are 2-tuples of [0,1]-scaled initial and final values...
        hsv_ranges = {
            "hue_range":(0,.66), # Hue range, blue->red by default.
            "sat_range":(1,1),   # Saturation range, 1 by default
            "val_range":(1,1),   # Value range, 1 by default
            }
        """
        assert type(label) in (str, int)
        # If this is a grayscale array...
        if label in self.labels:
            array = self.data(label)
            # render as an HSV with any provided arguments if requested
            if as_hsv:
                gt.quick_render(gt.scal_to_rgb(array, **hsv_ranges))
            else:
                # Otherwise just render it as a grayscale.
                gt.quick_render(array)
            return
        # If this is an RGB...
        self._validate_rgb_label(label)
        gt.quick_render(self.get_rgb(label))

    def get_subgrid(self, target_latlon:tuple, dy_px:int, dx_px:int,
                    from_center:bool=True, boundary_error:bool=True):
        """
        Returns a new MOD021KM object for a geographic subgrid of this one.

        :@param target_latlon: 2-Tuple of float values specifying the"anchor"
                point of the boundary box. By default, this is the top left
                corner of the rectangle if dx_px and dy_px are positive. If
                from_center is True, the closest pixel will be the center of
                the rectangle.  :@param dx_px: Horizontal pixels from anchor.
                Positive values correspond to
                an increase in the second axis of the latlon ndarray, which is
                usually rendered as "rightward", or increasing longitude
        :@param dy_px: Vertical pixels from anchor. Positive values correspond
                to an increase in the first axis of the latlon ndarray, which
                is usually rendered as "downward", or decreasing longitude.
        :@param from_center: If True, target_latlon describes the center point
                of a rectangle with width dx_px and height dy_px.
        :@param boundary_error: If True, raises a ValueError if the requested
                pixel boundary extends outside of the latlon domain. Otherwise
                returns a boundary at the closest valid pixel. This means if
                boundary_error is False and the grid overlaps a boundary, the
                returned array will not have the requested shape.
        """
        dy, dx = geo_helpers.get_geo_range(
                latlon=np.dstack(self._geo[:2]), target_latlon=target_latlon,
                dx_px=dx_px, dy_px=dy_px, from_center=from_center,
                boundary_error=boundary_error, debug=True)
        tmp_data = [self.data(b)[dy[0]:dy[1],dx[0]:dx[1]] for b in self._bands]
        tmp_geo = ([G[dy[0]:dy[1],dx[0]:dx[1]] for G in self._geo],
                   None)[self._geo is None]
        tmp_sunsat = ([S[dy[0]:dy[1],dx[0]:dx[1]] for S in self._sunsat],
                   None)[self._sunsat is None]
        return MOD021KM(tmp_data, self._info, tmp_geo, tmp_sunsat)

    def _label_or_array_to_array(self, label_or_array):
        """
        Helper function that takes a value which may either be a (M,N,3) RGB
        array, a (M,N) grayscale array, a string RGB recipe, a string scalar
        recipe, or an integer band
        """
        is_label = lambda k: type(k) in (str, int) and \
                (k in self.labels or k in self._rgb_recipes.keys())
        if not is_label(label_or_array):
            try:
                X = label_or_array
            except:
                raise ValueError(f"Provided array is not a " + \
                        "valid (M,N) or M,N,3 numpy array")
        else:
            if label_or_array in self._rgb_recipes.keys():
                X = self.get_rgb(label_or_array)
            else:
                X = self.data(label_or_array)
        return X

    def histogram_match(self, from_label_or_array, to_label_or_array,
                        nbins=256, equalize:bool=False, show:bool=False,
                        fig_path:Path=None):
        """
        Histogram matchs the 2d array referenced by from_label_or_array to the
        2d array referenced by to_label_or_array, and returns the
        histogram-matched 2d array without updating internal data or labels.

        Consider generating a normal distribution of greyscale values with
        guitools.get_normal_array and providing a mean/stdev of your choice.

        :@param from_label_or_array: int (band), str (label of a scalar or RGB
            Recipe) or label of a valid 2d array loaded into this object. This
            array's brightness distribution is remapped to the
            to_label_or_array array.
        :@param to_label_or_array: int (band) or str (2d array from recipe)
            label of a valid 2d array loaded into this object. This array's
            cumulative brightness distribution histogram is used as a reference
            to adapt the distribution of the  from_label_or_array array.
        :@param nbins: Number of brightness bins to use for matching
        :@param nbins: If True, histogram-equalizes the proveded label's array
            and returns it in the analysis dictionary.
        :@param show: if True, shows a generated matplotlib plot.
        :@param fig_path: if not None, saves histogram plot as a png
        :@param plot_spec: geo_plot.plot_lines plot_spec specification
        """
        # Evaluate the Distribution to match and the reference distribution
        # as either RGB or 2d scalar arrays.
        F = self._label_or_array_to_array(from_label_or_array)
        T = self._label_or_array_to_array(to_label_or_array)

        # Do histogram matching on the 2d or RGB arrays
        matched = enh.histogram_match(F,T,nbins=nbins)
        if show:
            gt.quick_render(enh.linear_gamma_stretch(matched))
        if fig_path:
            gp.generate_raw_image(enh.linear_gamma_stretch(matched), fig_path)
        return matched

    def rgb_histogram_analysis(self, rgb_label, nbins:int=256,
                               equalize:bool=False, show:bool=False,
                               fig_path:Path=None, data_range:tuple=None,
                               plot_spec:dict={}):
        """
        Perform histogram analysis on a loaded RGB recipe with
        enhance.do_hisogram_analysis. Separate method from histogram_analysis
        because the generalization would be dirty.

        :@param rgb_label: Valid loaded RGB label.
        :@param nbins: Number of brightness bins to use in analyzing frequency.
        :@param equalize: If True, histogram-equalizes the provided label's
            array and returns it in the analysis dictionary.
        :@param show: if True, shows a generated matplotlib plot.
        :@param fig_path: if not None, saves histogram plot as a png
        :@param plot_spec: geo_plot.plot_lines plot_spec specification
        """
        hists = []
        labels = ["RED", "GREEN", "BLUE"]
        array = self.get_rgb(rgb_label)
        rgb = [array[:,:,i] for i in range(3)]
        for i in range(3):
            if not data_range is None:
                tmp_data = np.clip(rgb[i], *tuple(data_range))
            hists.append(enh.do_histogram_analysis(rgb[i], nbins, equalize))
        if show or fig_path:
            gp.plot_lines(
                domain=[np.arange(nbins)*h["bin_size"]+h["Xmin"]
                        for h in hists],
                ylines=[h["hist"] for h in hists],
                labels=labels,
                plot_spec = plot_spec,
                image_path=fig_path,
                show=show
                )
        return hists

    def histogram_analysis(self, labels, nbins:int=256, equalize:bool=False,
                           show:bool=False, fig_path:Path=None,
                           data_range:tuple=None, plot_spec:dict={}):
        """
        Perform histogram analysis with enhance.do_hisogram_analysis on one
        or more integer band or string-labeled 2d arrays

        :@param labels: int, str, or iterable of valid 2d array labels.
        :@param nbins: Number of brightness bins to use in analyzing frequency.
        :@param equalize: If True, histogram-equalizes the proveded label's
            array and returns it in the analysis dictionary.
        :@param show: if True, shows a generated matplotlib plot.
        :@param fig_path: if not None, saves histogram plot as a png
        :@param plot_spec: geo_plot.plot_lines plot_spec specification
        """
        labels = [labels] if single_label else labels
        hists = []
        for l in labels:
            if not data_range is None:
                tmp_data = np.clip(self.data(l), *tuple(data_range))
            hists.append(enh.do_histogram_analysis(
                self.data(l), nbins, equalize))
        if show or fig_path:
            gp.plot_lines(
                domain=[np.arange(nbins)*h["bin_size"]+h["Xmin"]
                        for h in hists],
                ylines=[h["hist"] for h in hists],
                labels=labels,
                plot_spec = plot_spec,
                image_path=fig_path,
                show=show
                )
        return hists[0] if single_label else hists

    def get_mask(self, label_or_array, lower=False, upper=False,
                 show:bool=False, fig_path:Path=None,
                 select_resolution:int=512, choose_rgb_params:bool=False,
                 use_hsv:bool=False, rgb_type:str=None,rgb_match=False,
                 mask_color:list=[0,0,0], debug:bool=False):
        """
        Ask the user to choose bounds on a 2d array, and return a mask on
        values outside the chosen threshold. Optionally render a RGB with
        the mask, which can be modified with user-selected parameters

        :@param label: integer or string label of the scalar dataset
        :@param lower: If True, gets lower bound. Upper bound otherwise.
        :@param upper: If True, gets upper bound. Upper bound otherwise.
        :@param show: If True, quick-renders the thresholded RGB
        :@param select_resolution: point resolution of mask selection trackbar.
                Better masks can be selected with a higher number, but
                computational load increases linearly with resolution.
        :@param choose_rgb_params: If True, user is asked to select RGB gammas.
        :@param use_hsv: If True and data is 2d, scale to hsv bounds if a
            base rgb isn't being used to choose the bounds.
        :@param rgb_type: RGB recipe label or (M,N,3) RGB array to use as a
            base layer for selecting the mask on the data specified by
            label_or_array.
        :@param rgb_match: If True, histogram-matches unmasked values to the
            brightness distribution of the RGB, which can be configured by
            hand if choose_rgb_params is True. This helps enhance hidden
            features in the specific class being masked. Returns a tuple with
            the mask and enhanced array instead of just the mask.
        :@param mask_color: [0,1] normalized RGB color of mask in renders
        """
        array = self._label_or_array_to_array(label_or_array)
        if not lower and not upper:
            raise ValueError(
                    f"Either lower or upper must be true to choose a bound.")
        # Set the selection base-layer RGB to requested recipe or array.
        rgb_mask = []
        if rgb_type is None:
            if len(array.shape)==3:
                if use_hsv:
                    raise ValueError("Cannot use HSV mapping on an RGB array")
                base_rgb = np.copy(array)
            else:
                if use_hsv:
                    base_rgb = gt.scal_to_rgb(np.copy(array))
                    print(enh.array_stat(base_rgb))
                else:
                    base_rgb = np.dstack([np.copy(array) for i in range(3)])
        else:
            base_rgb = self._label_or_array_to_array(rgb_type)
        if not base_rgb.shape[2]==3:
            raise ValueError(f"base_rgb must be (M,N,3) RGB recipe or array")
        # Let the user choose gamma parameters for each RGB band.
        if choose_rgb_params:
            rgb_labels = list(range(3))
            rgb_pc = PixelCat(
                    [base_rgb[:,:,i] for i in range(base_rgb.shape[2])],
                    rgb_labels)
            for l in rgb_labels:
                rgb_pc.pick_gamma(l, gamma_scale=5, set_band=True)
            base_rgb = rgb_pc.get_rgb(rgb_labels)

        # If the provided array is an RGB, get the mask for each independently
        if len(array.shape)==3 and array.shape[2]==3:
            base_rgb = enh.linear_gamma_stretch(base_rgb)
            rgb_mask = [
                self.get_mask(array[:,:,i], lower=lower, upper=upper,
                              select_resolution=select_resolution,
                              rgb_type=base_rgb)
                for i in range(array.shape[2])]
            # Mask any pixel that is OOB for one channel
            mask = np.logical_or(np.logical_or(
                rgb_mask[0],rgb_mask[1]),rgb_mask[2])
            gt.quick_render(mask.astype(np.uint8)*255)
            rgb_mask = np.dstack([mask for i in range(3)])
            if rgb_match:
                # Get the mask, and a [0,1] array 'stretch' that contains
                # a version of the base RGB histogram-stretched between
                # the bounds
                masked_rgb = np.ma.masked_where(rgb_mask, base_rgb)
                base_rgb = enh.linear_gamma_stretch(self.histogram_match(
                    masked_rgb, base_rgb, nbins=1024, show=True))
            # Mask the out-of-bounds values in the RGB
            base_rgb[np.where(rgb_mask)] = 0
            if show:
                gt.quick_render(base_rgb)
            if fig_path:
                gp.generate_raw_image(base_rgb, fig_path)
            if rgb_match:
                return mask, base_rgb
            return mask

        # If the requested array is 2d, get data-scale bounds for each.
        if lower:
            lb = self.get_bound(
                    array,lower=True,select_resolution=select_resolution,
                    base_arr=base_rgb,debug=debug)
        else:
            lb = np.amin(array)
        if upper:
            ub = self.get_bound(
                    array,upper=True,select_resolution=select_resolution,
                    base_arr=base_rgb,debug=debug)
        else:
            ub = np.amax(array)

        # Create a mask using the data-scale bounds
        mask_color = np.asarray(mask_color)
        bottom_mask = array < lb
        top_mask = array > ub
        mask = np.logical_or(bottom_mask, top_mask)
        # Go ahead and return the mask if we aren't visualizing.
        if not fig_path and not show and not rgb_match:
            return mask
        #masked_rgb = np.copy(base_rgb)
        #masked_rgb[mask] = mask_color
        # Histogram-match the unmasked values to the RGB
        if rgb_match:
            matched_rgb = np.ma.masked_where(
                    np.dstack([np.copy(mask) for i in range(3)]),base_rgb)
            matched_rgb = self.histogram_match(matched_rgb, base_rgb,
                                               nbins=1024, show=False)
            base_rgb = matched_rgb.data
        base_rgb = enh.linear_gamma_stretch(base_rgb)
        base_rgb[mask] = mask_color
        if show:
            gt.quick_render(base_rgb)
        if fig_path:
            gp.generate_raw_image(base_rgb, fig_path)
        if rgb_match:
            #return mask, masked_rgb
            return mask, base_rgb
        return mask

    def get_bound(self, label_or_array, lower=False, upper=False,
                  base_arr=None, select_resolution:int=512, debug:bool=False):
        """
        returns a numpy mask and data value for a user-selected lower bound.
        :@param label: loaded integer band or string recipe label
        :@param lower: if True, ask the user to choose a lower bound.
        :@param upper: if True, ask the user to choose an upper bound.
        :@param base_arr: band number, str recipe label, str RGB recipe label,
                (M,N,3) RGB array, or (M,N) grayscale array to use as a
                reference while choosing the parameter refered to by label.
                By default, the label array is used.
        :@param select_resolution: point resolution of bound selection trackbar
                Better masks can be selected with a higher number, but
                computational load increases linearly with resolution.
        """
        if not (lower or upper):
            raise ValueError(f"Must choose either lower or upper bound")
        global _bound_base_arr

        # Get an array from the provided data label or array
        if type(label_or_array) not in (str, int):
            label = ""
        else:
            label = deepcopy(label_or_array)
        original = self._label_or_array_to_array(label_or_array)
        # Get the background array
        if not base_arr is None:
            base_arr = self._label_or_array_to_array(base_arr)
        else:
            base_arr = np.copy(original)

        # Evaluate each channel independently if an RGB is provided.
        if len(original.shape)==3 and original.shape[2]==3:
            print(TFmt.RED(
                "Getting bounds for all 3 channels of RGB", bold=True))
            return [self.get_bound(original[:,:,i], lower=lower, upper=upper,
                              base_arr=np.copy(base_arr), debug=debug
                              ) for i in range(3)]

        # Assign a global variable with the base array choice for the callback
        global _bound_base_arr
        global _sel_res
        _bound_base_arr = base_arr
        _sel_res = select_resolution

        def pick_lbound(X,v):
            """ Callback function for rendering the user's l-bound choice """
            global _bound_base_arr
            global _sel_res
            Xnew = enh.linear_gamma_stretch(np.copy(X))
            mask = Xnew<v/(_sel_res-1)
            if _bound_base_arr is None or _bound_base_arr.size==0:
                Xnew[np.where(mask)] = 0
                _bound_base_arr = enh.linear_gamma_stretch(Xnew)
            bba = np.copy(np.asarray(_bound_base_arr))
            if len(bba.shape) == 3 and bba.shape[2]==3:
                bba[np.where(mask)] = np.array([0,0,0])
                bba = bba[:,:,::-1]
            elif len(bba.shape) == 2:
                bba[np.where(mask)] = 0
            else:
                raise ValueError(
                        f"Shape of base array is invalid: {bba.shape}")
            return bba
        def pick_ubound(X,v):
            """ Callback function for rendering the user's u-bound choice """
            global _bound_base_arr
            global _sel_res
            Xnew = enh.linear_gamma_stretch(X)
            mask = Xnew>v/(_sel_res-1)
            if _bound_base_arr is None:
                Xnew[np.where(mask)] = np.amin(Xnew)
                _bound_base_arr = enh.linear_gamma_stretch(Xnew)
            bba = np.copy(np.asarray(_bound_base_arr))
            if len(bba.shape) == 3 and bba.shape[2]==3:
                bba[np.where(mask)] = np.array([0,0,0])
                bba = bba[:,:,::-1]
            elif len(bba.shape) == 2:
                bba[np.where(mask)] = 0
            else:
                raise ValueError(
                        f"Shape of base array is invalid: {bba.shape}")
            return bba

        # Ask the user for lower and/or upper bounds.
        Xnew, Xmin, Xrange = enh.linear_gamma_stretch(
                np.copy(original), report_min_and_range=True)
        lb, ub = None, None
        if lower:
            if debug:
                print(TFmt.WHITE(f"Choose lower bound for {label}"))
            lb = gt.trackbar_select(
                X=np.copy(original),
                func=pick_lbound,
                label=label,
                resolution=select_resolution,
                )
            lb = lb/(select_resolution-1)*Xrange+Xmin
            if debug:
                print(TFmt.WHITE("Selected"),TFmt.YELLOW(str(lb)))
        if upper:
            if debug:
                print(TFmt.WHITE(f"Choose upper bound for {label}"))
            ub = gt.trackbar_select(
                X=np.copy(original),
                func=pick_ubound,
                label=label,
                resolution=select_resolution,
                )
            ub = ub/(select_resolution-1)*Xrange+Xmin
            if debug: print(TFmt.WHITE("Selected"),TFmt.YELLOW(str(ub)))
        _bound_base_arr = None
        if not lb is None and not ub is None:
            return lb, ub
        elif not lb is None:
            return lb
        return ub

    def spectral_analysis(self, array_labels:list=None, mask_labels:list=None,
                          masks:list=None, fig_path:Path=None, show:bool=False,
                          bar_sigma:float=1, shade_sigma:float=1/3,
                          yscale="linear", plot_spec:dict={}):
        """
        Do spectral analysis of the mean and standard deviation of each
        provided band or scalar array label, and return a dict of means
        and standard deviations in data coordinates for each mask label.
        If no masks or labels are provided, defaults to all bands, unmasked.
        Optionally plot the spectral response of the masked pixels.

        The returned dictionary is formatted like:
        data_dict = {
            "Class 1":{"means":[9,8,7], "stdevs":[1,2,3]}
            "Class 2":{"means":[9,8,7], "stdevs":[1,2,3]}
            "Class 3":{"means":[9,8,7], "stdevs":[1,2,3]}
            }

        :@param array_labels: 2+ valid band numbers or scalar recipe labels
        :@param mask_labels: Class labels corresponding to each mask
        :@param mask: 2d boolean array for each class with the analysis domain
            set to True
        """
        data_dict = {}
        def get_xlabel(alabel):
            if type(alabel)==str:
                return alabel.upper()
            return f"{self.ctr_wl(alabel):.3f} $\mu m$\nBand {alabel}"

        # If no labels provided, plot all bands in order of wavelength
        if array_labels is None:
            array_labels = sorted(
                    [b for b in self.bands if self.info(b)["is_reflective"]],
                    key=lambda b: self.ctr_wl(b))

        # If all band numbers, sort by wavelength.
        if all([type(s) is int for s in array_labels]):
            array_labels.sort(key=lambda b: self.ctr_wl(b))
        if masks:
            assert len(mask_labels)==len(masks)
            for i in range(len(masks)):
                values = [self.data(label)[np.where(masks[i])]
                          for label in array_labels]
                means = [np.average(V) for V in values]
                stdevs = [np.std(V) for V in values]
                data_dict[mask_labels[i]] = {"means":means, "stdevs":stdevs}
        else:
            values = [np.ravel(self.data(label)) for label in array_labels]
            means = [np.average(V) for V in values]
            stdevs = [np.std(V) for V in values]
            data_dict["Spectral Response"] = {"means":means, "stdevs":stdevs}
        xlabels = [ get_xlabel(label) for label in array_labels ]
        gp.stats_1d(data_dict=data_dict, band_labels=xlabels,
                    fig_path=fig_path, show=show, class_space=.2,
                    bar_sigma=bar_sigma, shade_sigma=shade_sigma,
                    yscale=yscale, plot_spec=plot_spec)
        return data_dict

    def geo_scalar_plot(self, label, show:bool=True, fig_path:Path=None,
                        plot_spec:dict={}):
        """
        Wrapper on geo_plot.geo_scalar_lot for 2d arrays.

        :@param label: int band or str label of a 2d recipe
        :@param show: If True, show matplotlib histogram in qt window
        :@param fig_path: If Path is valid, saves the figure there.
        :@param plot_spec: geo_plot.geo_scalar_plot style plot_spec dict.
        """
        gp.geo_scalar_plot(self.data(label), self._geo[0], self._geo[1],
                           show=show,fig_path=fig_path, plot_spec=plot_spec)

    def get_kmeans(self, labels_or_arrays:list=None, class_count:int=4,
                   tolerance=1e-3, return_as_ints:bool=False, debug=False):
        """
        Perform K-means classification on the requested scalar arrays and
        return a list of masks with length class_count.

        :@param labels_or_arrays: list of loaded scalar recipes or band
            numbers, or 2d ndarrays with the same shape as this subgrid. By
            default all loaded bands are used.
        :@param class_count: Number of K-means classes to acquire
        :@param tolerance: Floating-point tolerance for mean vector equality
            in [0,1]-normalized data coordinates.
        :@param return_as_ints: if True, returns the classified array as a
            2d array of integers in [0,class_count) corresponding to each
            identified k-means class.

        :@return: list of a 2d boolean mask for each k-means class.
        """
        if labels is None:
            labels = self.bands
        labels = list(labels)
        arrays = [None for i in range(len(labels))]
        for i in range(len(labels)):
            if type(labels[i]) not in [int,str]:
                try:
                    arrays[i] = np.asarray(labels[i])
                    assert arrays[i].shape == self._shape
                except:
                    raise ValueError(
                            f"Labels that aren't a recipe or band number" + \
                             " must be a array with this subgrid's shape")
            else:
                arrays[i] = self.data(labels[i])

        norm = lambda X: 10*enh.linear_gamma_stretch(X)
        km = classify.k_means(
                X=np.dstack([norm(arrays[i]) for i in range(len(arrays))]),
                cluster_count=class_count,
                tolerance=tolerance,
                get_sse=False,
                debug=debug
                )
        masks = []
        for c in range(len(km)):
            tmp_mask = np.full_like(arrays[0], False, dtype=bool)
            for i,j in km[c]:
                tmp_mask[i,j] = True
            masks.append(tmp_mask)
        if not return_as_ints:
            return masks
        arr = np.full_like(masks[0], 0, np.uint8)
        for i in range(len(masks)):
            arr[np.where(masks[i])] = i
        return arr

    def get_mlc(self, sample_dict:dict, labels_or_arrays:list=None,
                thresh=None):
        """
        Perform MLC classification using the scalar arrays corresponding to
        the provided list of labels, using the class samples defined in
        the required sample_dict.

        :@param sample_dict:Dictionary mapping class names to a list of (i,j)
            array coordinates of pixel samples for that class indexing the
            first (y) and second (x) dimensions, respectively. Boolean masks
            can be converted to pixels indeces with MOD021KM.mask_to_idx
        :@param labels_or_arrays: list of loaded scalar recipes or band
            numbers, or 2d ndarrays with the same shape as this subgrid. By
            default all loaded bands are used.
        :@param thresh: [0,1] confidence percentile for inclusion in a class.
            Values outside the threshod are labeled with new "uncertain" class.
        """
        #print(sample_dict.keys())
        if labels_or_arrays is None:
            labels_or_arrays = self.bands
        labels_or_arrays = list(labels_or_arrays)
        arrays = [None for i in range(len(labels_or_arrays))]
        for i in range(len(labels_or_arrays)):
            if type(labels_or_arrays[i]) not in [int,str]:
                try:
                    arrays[i] = np.asarray(labels_or_arrays[i])
                    assert arrays[i].shape == self._shape
                except:
                    raise ValueError(
                            f"Labels that aren't a recipe or band number" + \
                             " must be a array with this subgrid's shape")
            else:
                arrays[i] = self.data(labels_or_arrays[i])

        norm = lambda X: 10*enh.linear_gamma_stretch(X)
        classified, keys = classify.mlc(np.dstack(arrays), sample_dict, thresh)
        return classified, keys
