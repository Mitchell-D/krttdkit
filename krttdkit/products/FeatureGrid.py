"""
The FeatureGrid class provides an abstraction on a set of 2d scalar arrays on
a uniform-grid, enabling the user to easily access, visualize, manipulate, and
store the scalar feature arrays along with labels and metadata.
"""
from pathlib import Path
from datetime import datetime
from datetime import timedelta
import pickle as pkl
from copy import deepcopy
import numpy as np
import json

from krttdkit.operate import enhance as enh
from krttdkit.operate.recipe_book import transforms
from krttdkit.operate import Recipe

class FeatureGrid:
    @staticmethod
    def from_pkl(pkl_path:Path):
        """
        Recovers a FeatureGrid from a pkl object expected to contain a 2-tuple
        like (fg_dict, data) where fg_dict is a dictionary following the
        FeatureGrid standard with keys for 'labels', 'info', and 'meta', and
        data is a list of uniform (M,N,F) shaped arrays corresponding to the
        F features in the 'labels' and 'info' arrays.
        """
        fg_dict, data = pkl.load(pkl_path.open("rb"))
        return FeatureGrid(
                labels=fg_dict["labels"],
                data=data,
                info=fg_dict["info"],
                meta=fg_dict["meta"]
                )

    def __init__(self, labels:list, data:list, info:list=None, meta:dict={}):
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
        # Freely-accessible meta-data dictionary. 'shape' is set by default.
        self._meta = meta
        # The shape parameter is set dynamically at __init__. If the meta
        # dictionary contains a shape from a previous iteration, get rid of it.
        self._meta["shape"] = None
        self._recipes = {}
        self._shape = None

        for i in range(len(labels)):
            self.add_data(labels[i], data[i], info[i])

    @property
    def labels(self):
        return tuple(self._labels)

    @property
    def shape(self):
        return self._shape

    @property
    def meta(self):
        return self._meta

    def to_dict(self):
        """
        All the information needed to recover the FeatureGrid given an
        appropriately-shaped data array.
        """
        return {"labels":self._labels, "info":self._info, "meta":self._meta}

    def info(self, label):
        assert label in self._labels
        return self._info[self._labels.index(label)]

    def data(self, label:str):
        """
        Return the array or evaluated recipe associated with the label
        """
        sequence = label.split(" ")
        base_label = sequence[-1]
        tran = sequence[:-1]
        print(f"Getting {label}")
        if not self._label_exists(base_label):
            raise ValueError(f"Label {base_label} not recognized")
        X = self._evaluate_recipe(base_label)
        for tranfunc in [transforms[s] for s in tran[::-1]]:
            X = tranfunc(X)
        return X

    def add_recipe(self, label:str, recipe:Recipe):
        if self._label_exists(label) or label in transforms.keys():
            raise ValueError(f"Label {label} already exists.")
        assert type(recipe) is Recipe
        self._recipes[label] = recipe

    def _evaluate_recipe(self, recipe:str):
        """
        Return evaluated recipe or base feature from a label
        """
        if recipe in self.labels:
            return self._data[self.labels.index(recipe)]
        elif recipe in self._recipes.keys():
            args = tuple(self.data(arg) for arg in self._recipes[recipe].args)
            return self._recipes[recipe].func(*args)
        else:
            raise ValueError(f"{recipe} is not a valid recipe or label.")

    def to_pkl(self, pkl_path:Path, overwrite=True):
        """
        Stores this FeatureGrid object as a pkl recoverable by the
        FeatureGrid.from_pkl static method.

        :@param pkl_path: Location to save this ABIL1b instance
        :@param overwrite: If True, overwrites pkl_path if it already exits
        """
        if pkl_path.exists() and not overwrite:
            raise ValueError(f"pickle already exists: {pkl_path.as_posix()}")
        pkl.dump((self.to_dict(), self._data), pkl_path.open("wb"))

    def subgrid(self, labels:list=None, vrange:tuple=None, hrange:tuple=None):
        """
        Given array slices corresponding to the horizontal and vertical axes
        of this FeatureGrid, returns a new FeatureGrid with subsetted arrays.

        :@param labels:
        :@param vrange:
        :@param hrange:
        """
        vslice = slice(None) if vrange is None else slice(*vrange)
        hslice = slice(None) if hrange is None else slice(*hrange)
        labels = self.labels if labels is None else labels
        fg = FeatureGrid(
                labels=labels,
                data=[self.data(l)[vslice,hslice] for l in labels],
                info=[self.info(l) for l in labels],
                meta=self.meta
                )
        fg._recipes.update(self._recipes)
        return fg

    def to_json(self, indent=None):
        """
        Returns the dictionary version of this FeatureGrid as a json-formatted
        string. This includes labels, shape, etc so that the full FeatureGrid
        object can be recovered given a (M,N,F) shaped unlabeled numpy array
        """
        return json.dumps(self.to_dict(), indent=indent)

    def get_rgb(self, r:str, g:str, b:str):
        """
        Given 3 recipes, return an RGB after evaluating any transforms/recipe
        """
        return np.dstack(map(self.data, (r,g,b)))

    def add_data(self, label:str, data:np.ndarray, info:dict=None):
        """
        Add a new data field to the FeatureGrid with an equally-shaped ndarray
        and a unique label. If this FeatureGrid has no data, this method will
        set the object's immutable shape attribute.

        :@param label: Unique label to identify the data array
        :@param data: 2d numpy array with identical shape to this FeatureGrid
        :@param info: Optional dictionary of attributes corresponding to this
            dataset, which can be useful for storing information for
            downstream applications.

        :@return: None
        """
        if self._shape is None:
            assert len(data.shape)==2
            self._shape = data.shape
            self._meta["shape"] = self._shape
        # Make sure the data array's shape matches this grid's
        elif self._shape != data.shape:
            raise ValueError(
                    f"Cannot add {label} array with shape {data.shape}. Data"
                    f" must match this FeatureGrid's shape: {self._shape}")

        # Make sure the new label is unique and valid
        if self._label_exists(label):
            raise ValueError(f"A feature with label {label} is already added.")

        self._labels.append(label)
        self._data.append(data)
        self._info.append(dict(info) if info else {})

    def __repr__(self, indent=2):
        """
        Print the meta-info
        """
        return self.to_json(indent)

    def _label_exists(self, label:str):
        """
        Returns True if the provided case-insensitive label matches either a
        currently-loaded scalar feature array or an added recipe. Accepts
        any object that implements __str__(), ie integer band numbers.
        """
        label = str(label).lower()
        return label in self._labels or label in self._recipes.keys() \
                or label in transforms.keys()

