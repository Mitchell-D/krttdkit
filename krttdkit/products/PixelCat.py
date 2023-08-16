""" Helper class with gui methods for pixel classification tasks. """
from dataclasses import dataclass
import pickle as pkl
from pathlib import Path
import numpy as np
import math as m

from krttdkit.operate import enhance
from krttdkit.visualize import guitools as gt

class PixelCat:
    """
    Wrapper class for supervised and unsupervised classification of arrays.

    Initialized with an set of arrays and labels. Any enhancements
    must be done as pre or post processing, but show_rgb() allows for
    temporary visualization of provided arrays with a rgb recipe.

    Several methods enable you to select a parameter value for a band
    or set of bands. If you generate a new array with a method like
    pick_linear_contrast(), you can subsequently add it as a new 'band'
    with set_band()
    """
    def __init__(self, arrays:list, bands:list):
        """
        :@param arrays: List of uniformly-shaped arrays of data
        :@param bands: List of unique labels corresponding to each array
        """
        # Arrays must be uniformly-shaped, and each one must have a unique
        # band label.
        assert all([ X.shape==arrays[0].shape and len(arrays[0].shape)==2
                    for X in arrays ])
        assert len(set(bands)) == len(arrays) and len(arrays)
        self._arrays = list(arrays)
        self._bands = list(bands)
        self._shape = arrays[0].shape

    @property
    def bands(self):
        return self._bands

    def band(self, band:str):
        """ Returns the array associated with the provided band label """
        assert band in self._bands
        return self._arrays[self._bands.index(band)]

    def pick_staurated_contrast(self, band:str, offset:float=0,
                             set_band:bool=False, debug=False):
        """ """
        assert band in self._bands
        slope_scale = 50
        contrast = m.e**((gt.trackbar_select(
                X=self.band(band),
                func=lambda X, v: enhance.linear_contrast(
                        X, m.e**((v-100)/slope_scale), offset),
                label=f"Band {band} contrast slope: ",
                debug=debug
                )-100)/slope_scale)
        if set_band:
            print("Setting band")
            band_idx = self._bands.index(band)
            self._arrays[band_idx] = enhance.linear_contrast(
                    self.band(band), contrast, offset)
        return contrast

    def pick_band_scale(self, band:str, set_band:bool=False):
        assert band in self._bands
        choice = gt.trackbar_select(
                X=self.band(band),
                func=lambda X, v: X*v/255
                )/255
        if set_band:
            self.set_band(self.band(band)*choice, band, replace=True)
        return choice

    def pick_linear_contrast(self, band:str, offset:float=0,
                             set_band:bool=False, debug=False):
        """ """
        assert band in self._bands
        slope_scale = 50
        contrast = m.e**((gt.trackbar_select(
                X=self.band(band),
                func=lambda X, v: enhance.linear_contrast(
                        X, m.e**((v-100)/slope_scale), offset),
                label=f"Band {band} contrast slope: ",
                debug=debug
                )-100)/slope_scale)
        if set_band:
            print("Setting band")
            band_idx = self._bands.index(band)
            self._arrays[band_idx] = enhance.linear_contrast(
                    self.band(band), contrast, offset)
        return contrast

    def pick_band_scale(self, band:str, set_band:bool=False):
        assert band in self._bands
        choice = gt.trackbar_select(
                X=self.band(band),
                func=lambda X, v: X*v/255
                )/255
        if set_band:
            self.set_band(self.band(band)*choice, band, replace=True)
        return choice

    '''
    def get_cats(bands:list, cat_count:int cat_names:list=None,
                 show_pool=False, debug=False):
        """
        """
        if type(bands)==str:
            bands = [ bands for i in range(3) ]
        assert len(bands) == 3

        return gt.get_category_series(X=np.dstack(bands), cat_count=cat_count,
                category_names=cat_names, show_pool=show_pool, debug=debug)
    '''
    def pick_linear_gamma(self, band:str, gamma_scale=.1, set_band:bool=False,
                          hist_equalize:bool=False, hist_nbins:int=256,
                          debug=False):
        """
        Choose a gamma value for a single using the gui trackbar selector.
        :@param band: Band ID string
        :@param gamma_scale: Scales the sensitivity of the gamma function to
                trackbar movement.
        :@param set_band: if True, the PixelCat band will be updated.
        :@param hist_equalize: If True, histogram-equalizes input before
                prompting the user for gamma selection.
        :@param hist_nbins: Number of bins to use if hist_equalize is True
        """
        assert band in self._bands
        sigmoid = lambda a: 1/(1+m.e**(-a*gamma_scale))
        if hist_equalize:
            choice = sigmoid(gt.trackbar_select(
                    X=self.band(band),
                    func=lambda X, v: enhance.linear_gamma_stretch(
                        enhance.histogram_equalize(X, nbins=hist_nbins)[0],
                        gamma=sigmoid(v-128)),
                    label=f"Band {band} linear gamma: ",
                    debug=debug
                    )-128)
        else:
            choice = sigmoid(gt.trackbar_select(
                    X=self.band(band),
                    func=lambda X, v: enhance.linear_gamma_stretch(
                        X, gamma=sigmoid(v-128)),
                    label=f"Band {band} linear gamma: ",
                    debug=debug
                    )-128)
        if set_band:
            if hist_equalize:
                self.set_band(
                        enhance.linear_gamma_stretch(
                            enhance.histogram_equalize(
                                self.band(band), nbins=hist_nbins)[0],
                            gamma=choice),
                        band=band, replace=True)
            else:
                self.set_band(enhance.linear_gamma_stretch(
                    self.band(band), choice), band=band, replace=True)
        return choice

    def pick_gamma(self, band:str, gamma_scale=1, set_band:bool=False,
                   debug=False):
        """
        Choose a gamma value for a single using the gui trackbar selector.
        :@param band: Band ID string
        :@param gamma_scale: Scales the sensitivity of the gamma function to
                trackbar movement.
        :@param set_band: if True, the PixelCat band will be updated.
        """
        assert band in self._bands
        choice = (gamma_scale/255)*(1+gt.trackbar_select(
                X=self.band(band),
                func=lambda X, v: enhance.gamma(X, (1+v)*(gamma_scale/255)),
                label=f"Band {band} gamma: ",
                debug=debug
                ))
        if set_band:
            self.set_band(enhance.gamma(
                self.band(band), choice), band=band, replace=True)
        return choice

    def get_rgb(self, bands:list, recipes:list=None, show=False,
                normalize:bool=False, debug=False):
        """
        Compile a (M,N,3)-shaped RGB numpy array of bands corresponding to
        the provided list of labels.
        By design, this class doesn't make any in-place modifications to its
        arrays, so unless a recipe is provided
        """
        # Verify 3 valid band labels were provided in the list
        assert len(bands) == 3 and all([ b in self._bands for b in bands ])
        # Verify that if recipes were provided, they are 3 functions.
        assert not recipes or len(recipes) == 3 and \
                all([ type(r)==type(lambda m:1) for r in recipes  ])
        recipes = recipes if recipes else [ lambda m:m for i in range(3) ]
        if debug: print(f"Generating RGB using bands {bands}")
        RGB = np.dstack([ recipes[i](self.band(bands[i]))
                          for i in range(len(bands)) ])
        if normalize:
            for i in range(3):
                RGB[:,:,i] = enhance.linear_gamma_stretch(RGB[:,:,i])
        if show:
            gt.quick_render(RGB)
        return RGB

    def set_band(self, array:np.ndarray, band:str, replace=False):
        """
        Add a new band array, which must have the same shape as the others,
        or set an existing band array to a new one.

        :@param array: Numpy ndarray with same shape as other arrays.
        :@param band: Unique string label for the new array. If replace=False
                and the band key already exists, and error is raised.
                Otherwise, the new array will replace the old one.
        """
        assert replace or band not in self._bands
        assert array.shape == self._shape

        if band in self._bands:
            if replace:
                self._arrays[self._bands.index(band)] = array
            else:
                raise ValueError(f"Band key {band} already exists")
            return None
        self._arrays.append(array)
        self._bands.append(band)
        return None

    def get_edge_filter(self, band:str, high_frequency:bool=True):
        """
        Apply an edge filter to both dimensions of the band brightness
        array by setting low (optionally high) frequency wave intervals
        to zero.

        :@param high_frequency: If True, remove high frequencies.
        :return: User-selected edge value and filtered array as a 2-tuple
        """
        # Get phase space image
        P = enhance.dft2D(
                np.copy(self.band(band)),
                inverse=False,
                use_scipy=True,
                )
        # Filter according to user specification
        my_filter = lambda X,v: enhance.norm_to_uint(np.abs(
            enhance.dft2D(enhance.border_mask(
                    X, cutoff=int(((v+1)/256)*X.shape[0]),
                    fill=0, high_freq=high_frequency),
                inverse=True, use_scipy=True)), 256, np.uint8)
        # Get a scaled filter value from the user
        user_value = gt.trackbar_select(P, my_filter)
        Q = enhance.border_mask(P, int(((user_value+1)/256)*P.shape[0]),
                                fill=0, high_freq=high_frequency)
        Y = enhance.linear_gamma_stretch(np.abs(enhance.dft2D(
                Q, inverse=True, use_scipy=True)))
        # Filter the phase array with the user-selected radius value.
        return (int(((user_value+1)/256)*P.shape[0]), my_filter(P, user_value))

    def get_box_filter(self, band:str, outside:bool=False, roll:bool=True):
        # Get phase space image
        P = enhance.dft2D(
                np.copy(self.band(band)),
                inverse=False,
                use_scipy=True,
                )
        if roll:
            P = np.roll(P, int(P.shape[0]/2), axis=0)
            P = np.roll(P, int(P.shape[1]/2), axis=1)
        yb, xb = gt.region_select(enhance.norm_to_uint(
            np.log(1+np.abs(P)), 256, np.uint8))
        if outside:
            P[:yb[0]] = 0
            P[:xb[0]] = 0
            P[yb[1]:] = 0
            P[xb[1]:] = 0
        else:
            P[yb[0]:yb[1],xb[0]:xb[1]] = 0

        if roll:
            P = np.roll(P, -int(P.shape[0]/2), axis=0)
            P = np.roll(P, -int(P.shape[1]/2), axis=1)
        Y = enhance.linear_gamma_stretch(np.abs(enhance.dft2D(
            P, inverse=True, use_scipy=True)))
        return Y

    def get_filter(self, band:str, low_pass:bool=True):
        """
        Apply a low-pass (by default) or high-pass filter to the band array
        and return an identically-shaped result normalized to [0,1].

        :return: User-selected radius value and filtered array as a 2-tuple
        """
        # Get phase space image
        P = enhance.dft2D(
                np.copy(self.band(band)),
                inverse=False,
                use_scipy=True,
                )
        # Filter according to user specification
        my_filter = lambda X,v: enhance.norm_to_uint(np.abs(
            enhance.dft2D(enhance.radius_mask(
                    X, radius=int(((v+1)/256)*X.shape[0]),
                    fill=0, true_inside=low_pass),
                inverse=True, use_scipy=True)), 256, np.uint8)
        # Get a scaled filter value from the user
        user_value = gt.trackbar_select(P, my_filter)
        Q = enhance.radius_mask(P, int(((user_value+1)/256)*P.shape[0]),
                                fill=0, true_inside=low_pass)
        Y = enhance.linear_gamma_stretch(np.abs(enhance.dft2D(
                Q, inverse=True, use_scipy=True)))
        # Filter the phase array with the user-selected radius value.
        return (int(((user_value+1)/256)*P.shape[0]), my_filter(P, user_value))


if __name__=="__main__":
    debug=True
    nbins=1024

    region_pkl = Path("/home/krttd/uah/23.s/aes670/aes670hw2/data/pkls/" + \
            "my_region.pkl")
    if not region_pkl.exists():
        from restore_pkl import restore_pkl
        restore_pkl(region_pkl, debug=debug)
    region, info, _, _ = pkl.load(region_pkl.open("rb"))

    pc = PixelCat(arrays=region, bands=[ b for b in info["bands"] ])

    truecolor_bands = ["M05", "M04", "M03"]
    tc_recipe = lambda X: enhance.norm_to_uint(
            enhance.histogram_equalize(X, nbins)[0], 256, np.uint8)
    noeq_recipe = lambda X: enhance.norm_to_uint(X, 256, np.uint8)

    # Show unequalized RGB
    pc.get_rgb(bands=truecolor_bands, recipes=[noeq_recipe for i in range(3)])
    # Show histogram-equalized RGB
    pc.get_rgb(bands=truecolor_bands, recipes=[tc_recipe for i in range(3)])

    pc.set_band(
        array=(pc.band("M07")-pc.band("M05"))/(pc.band("M07")+pc.band("M05")),
        band="NDVI"
        )

    pc.set_band(
        array=(pc.band("M10")-pc.band("M04"))/(pc.band("M10")+pc.band("M04")),
        band="NDSI"
        )

    custom_bands = ["M15", "NDVI", "NDSI"]
    pc.get_rgb(bands=custom_bands, recipes=[tc_recipe for i in range(3)])
