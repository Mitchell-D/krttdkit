"""
Methods for plotting geographic data on meshed axes
"""

import cartopy.crs as ccrs
import cartopy.feature as cf
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle as pkl
import xarray as xr
import metpy
import imageio

from pathlib import Path
from PIL import Image

from matplotlib.ticker import LinearLocator, StrMethodFormatter, NullLocator
from cartopy.mpl.gridliner import LongitudeLocator, LatitudeLocator

from ABIManager import ABIManager
from make_ndsii1 import get_ndsii1


class ColoGrad:
    def __init__(self, data:np.ndarray):
        """
        Facilitates binning and RGB coloring operations for scalar
        2d or 3d data arrays. This class is designed so that the
        "data" attribute can be arbitrarily changed. This means it's easy
        to do apply functions or do math with the data attribute.

        :param data: 2d or 3d array of numbers. If the data is 3d,
            it's implicitly assumed that the 3rd dimension is a time
            dimension, NOT a series of RGB values.
        """
        # Use the setter method to check dimensionality
        self._data = None
        self._depth = None
        self._data_min = None
        self._data_max = None
        self._stops = None# list of [stop_value:(r,g,b)]
        self.data = data
        self._base_color

    @property
    def stops(self):
        return self._stops
    @stops.setter

    def set_colorstop(self, value, color):
        """
        Add an inclusive color "stop" in the gradient.

        :param value: Inclusive data-scale scalar value at which point the
                extent of the provided RGB vector will be maximum.
        """
        # Iteration depth, so we can sort efficiently
        if not self._depth:
            self._depth = 0
        self.depth+=1
        if ColoGrad._isnumeric(value):
            color = ColoGrad._parse_color(color)
            self._stops.append((value,color))
        elif hasattr(value, '__iter__'):
            for s in value:
                set_colorstop(*s)
        self._depth-=1
        return self._sort_stops() if self._depth==0 else None

    def _sort_stops(self):
        """ Sorts the stops by increasing value """
        self._stops = list(sorted(self._stops, key=lambda s:s[0]))
        return self

    @staticmethod
    def _isnumeric(value):
        """
        Basic static method returning True if given value can convert to an int
        """
        return hasattr(value, '__int__')

    @staticmethod
    def _parse_color(color):
        """ Parses a hex or 3-element iterable color into a numeric tuple """
        # convert a "#\d{6}" formatted hex string into a rgb
        if type(color) is str:
            rgb = tuple(int(h.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        elif hasattr(color, '__iter__'):
            if len(color)!=3:
                raise ValueError(f"Color tuple must have 3 elements (r,g,b)")
            rgb = tuple(color)
        # Lenth is zero if none of the elements are numeric
        if not filter(ColoGrad._isnumeric, rgb):
            raise ValueError("All elements in RGB array must be numeric")
        return rgb

    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, data:np.ndarray):
        """
        :param data: 2d or 3d array of numbers. If the data is 3d,
            it's implicitly assumed that the 3rd dimension is a time
            dimension, NOT a series of RGB values.
        """
        if not data.shape in {2,3}:
            raise ValueError("Scalar data must be 2 or 3-dimensional")
        self._data = data
        self._data_min = np.amin(self._data)
        self._data_max = np.amax(self._data)

    def to_int8(self, inplace:bool=True):
        """ Round the scalar data to the nearest integer"""
        rounded = np.rint(self._data)
        if inplace:
            self._data = rounded
            return self
        else:
            return rounded

if __name__=="__main__":
    b02_pkl = Path("/home/krttd/uah/22.f/aes572/hw3/data/pkls/ndsii1/michigan_ref_b02_1km.pkl")
    b05_pkl = Path("/home/krttd/uah/22.f/aes572/hw3/data/pkls/ndsii1/michigan_ref_b05_1km.pkl")
    am2 = ABIManager().load_pkl(b02)
    am5 = ABIManager().load_pkl(b05)
    cg = ColoGrad()
