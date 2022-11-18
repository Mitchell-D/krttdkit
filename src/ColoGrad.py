"""
Methods for plotting geographic data on meshed axes
"""

import numpy as np
from pathlib import Path

from .ABIManager import ABIManager
#from .RecipeBook import ndsii1

class ColoGrad:
    class InterpMethods:
        @staticmethod
        def bezier(data:np.ndarray, stops:list):
            print(stops)
        @staticmethod
        def linear(data:np.ndarray, stops:list):
            points, colors = zip(*stops)
            print(points, colors)

    def __init__(self, data:np.ndarray=None, base_color=None):
        """
        Facilitates binning and RGB coloring operations for scalar
        2d or 3d data arrays. This class is designed so that the
        "data" attribute can be arbitrarily changed. This means it's easy
        to apply functions or do math with the data attribute.

        :param data: 2d or 3d array of numbers. If the data is 3d,
            it's implicitly assumed that the 3rd dimension is a time
            dimension, NOT a series of RGB values.
        :param base_color: hex value or 3-tuple of color values to use
            as the default or least-value-interval color. The minimum
            value in the domain maps closest to this color.
        """
        # Use the setter method to check dimensionality
        self._data = None
        self._depth = None
        self._data_min = None
        self._data_max = None
        self.interp = ColoGrad.InterpMethods()
        self._stops = []# list of [stop_value:(r,g,b)]
        self._methods = {
                "linear":self.interp.linear,
                "bezier":self.interp.bezier,
                }
        if not data is None:
            self.data = data
        self._base_color = self._parse_color(
                (0,0,0) if base_color is None else base_color)

    def symmetric_norm(self):
        """ Normalize the data to [-1,1] """
        self._data /= np.amax(np.abs(self._data))
        self._data = np.clip(self._data, -1, 1)
        return self

    def to_rgb(self, method:str=None):
        """
        Map the scalar data to an RGB using a supported method

        :param method: One of: [ "bezier", "linear" ]
        """
        if method is None:
            method = self._methods.keys()[0]
        if method not in self._methods.keys():
            raise ValueError(
                "Method {method} must be one of {self._methods.keys()}")
        return self._methods[method](data=self._data, stops=self._stops)

    @property
    def stops(self):
        return self._stops

    def set_colorstop(self, value, color):
        """
        Add an inclusive color "stop" in the gradient.

        :param value: Data-scale float value where data should be closest
                to the provided RGB color vector
        """
        # Iteration depth, so we can sort efficiently
        if not self._depth:
            self._depth = 0
        self._depth+=1
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
            rgb = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        elif hasattr(color, '__iter__'):
            if len(color)!=3:
                raise ValueError(f"Color tuple must have 3 elements (r,g,b)")
            rgb = tuple(color)
        # Lenth is zero if none of the elements are numeric
        if not filter(ColoGrad._isnumeric, rgb):
            raise ValueError("All elements in RGB array must be numeric")
        return rgb

    def __repr__(self):
        return str(self._base_color)

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
        if not len(data.shape) in {2,3}:
            raise ValueError("Scalar data must be 2 or 3-dimensional;" + \
                    f" current shape: {data.shape}")
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
    """
    b02_pkl = Path("/home/krttd/uah/22.f/aes572/hw3/data/pkls/ndsii1/michigan_ref_b02_1km.pkl")
    b05_pkl = Path("/home/krttd/uah/22.f/aes572/hw3/data/pkls/ndsii1/michigan_ref_b05_1km.pkl")
    am2 = ABIManager().load_pkl(b02)
    am5 = ABIManager().load_pkl(b05)
    """
    cg = ColoGrad()
