"""
The FeatureGrid class provides an abstraction on a set of 2d scalar arrays on
a uniform-grid, enabling the user to easily access, visualize, manipulate, and
store the scalar feature arrays along with labels and metadata.
"""
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from pprint import pprint as ppt
from copy import deepcopy
import multiprocessing as mp
import numpy as np
import pickle as pkl

class FeatureGrid:
    def __init__(self, labels:list, data:list, info:list=None):
        # Make sure there is a label for every dataset
        assert len(labels) == len(data) != 0
        # If an info dict is provided, make sure there's one for each feature
        if info:
            assert len(info) == len(labels)
        else:
            info = [{} for i in range(len(labels))]

        self._labels = []
        self._data = []
        self._info = []
        self._recipes = {}
        self._shape = None

        for i in range(len(labels)):
            self.add_data(labels[i], data[i], info[i])

    def add_data(self, label:str, data:np.ndarray, info:dict=None):
        if self._shape is None:
            assert len(data.shape)==2
            self._shape = data.shape
        # Make sure the data array's shape matches this grid's
        elif self._shape != data.shape:
            raise ValueError(
                    f"Cannot add {label} array with shape {data.shape}. Data"
                    f"must match this FeatureGrid's shape: {self._shape}")

        # Make sure the new label is unique and valid
        if self._label_exists(label):
            raise ValueError(f"A feature with label {label} is already added.")

        self._labels.append(label)
        self._data.append(data)
        self._info.append(dict(info) if info else {})

    def _label_exists(self, label:str):
        """
        Returns True if the provided case-insensitive label matches either a
        currently-loaded scalar feature array or an added recipe. Accepts
        any object that implements __str__(), ie integer band numbers.
        """
        label = str(label).lower()
        return label in self._labels or label in self._recipes.keys()

