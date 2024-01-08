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

from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp
from krttdkit.visualize import TextFormat as TF
from krttdkit.operate import enhance as enh
from krttdkit.operate.recipe_book import transforms
from krttdkit.operate import classify
from krttdkit.operate import Recipe

class FeatureGrid:
    """
    A FeatureGrid is defined four attributes:

     - list of unique string labels
     - list of data arrays with uniform shapes corresponding to each label
     - list of JSON-serializable information dicts corresponding to each label
     - JSON-serializable dict of meta-information about the FeatureGrid

    and can be fully represented as a 3d array of features shaped like (M,N,F)
    and a JSON tree of labels, information, and meta-data like:

    json_tree = {"labels":[list of strings],
                 "info":[list of JSON-serializable dicts]}
                 "meta":{JSON-serializable dict}}

    The FeatureGrid object enables the user to access and operate on the data
    using the string labels and the data() object method. This method is also
    able to evaluate a hierarchy of recipes.

    Data recipes referencing the labels for specific instances can be loaded
    with the add_recipe() object method.

    'transforms' are recipes that map a single array to an array with the same
    shape (ie functions of the form T:A->A). Prepending a transform label to
    a data label will return the data after the transform has been applied.

    For example, with FeatureGrid instance fg, which has the "truecolor"
    recipe loaded along with the required recipe data, one can get a truecolor
    image normalized to integers in [0,255] using the norm256 transform with:

    fg.data('norm256 truecolor')

    Also note: The JSON-serializability of meta-dictionaries are checked at
    merge. key collisions for meta dictionaries are handled as follows:
        - If both values for the key are lists, merges them
        - If one is a list and the other is a value, adds value to the list
        - If both are values, combines them as a list.
    """
    @staticmethod
    def merge(fg1, fg2, drop_duplicates=False, debug=False):
        """
        Given 2 FeatureGrid object instances, returns a new FeatureGrid with
        all of their features and meta-dictionaries combined. If any labels
        are duplicated and drop_duplicates is True, fg2 duplicates are ignored.
        """
        assert fg1.shape==fg2.shape
        new_labels = list(fg1.labels)
        new_data = list(fg1._data)
        new_info = list(fg1._info)
        # Check json serializability of both meta-dictionaries. This also
        # has the effect of forcibly converting any iterables to lists.
        try:
            new_meta = dict(json.loads(fg1.to_json()))
        except:
            raise ValueError(f"Provided FeatureGrid fg1 doesn't have JSON"+\
                    "-serializable meta-dict, labels, or info dicts.")
        try:
            fg2_meta = dict(json.loads(fg2.to_json()))
        except:
            raise ValueError(f"Provided FeatureGrid fg2 doesn't have JSON"+\
                    "-serializable meta-dict, labels, or info dicts.")


        # Merge the data, label, and info lists with fg2
        for l2 in fg2.labels:
            if l2 in new_labels:
                if debug: print(TF.RED(f"Duplicate label: {l2}"))
                if drop_duplicates:
                    continue
                raise ValueError(f"FeatureGrids have common label: {l2}")
            new_labels.append(l2)
            new_data.append(fg2.data(l2))
            new_info.append(fg2.info(l2))

        # Merge the meta-dictionaries.
        for k1 in fg2_meta.keys():
            if k1 in new_meta.keys():
                # Append to the list if already combined, or make a list
                if type(new_meta[k1]) is list:
                    new_meta[k1].append(fg2_meta[k1])
                else:
                    combined = [new_meta[k1], fg2_meta[k1]]
                    new_meta[k1] = combined

        return FeatureGrid(new_labels, new_data, new_info, new_meta)

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

    def __init__(self, labels:list=[], data:list=[],
                 info:list=[], meta:dict={}):
        # Make sure there is a label for every dataset
        assert len(labels) == len(data)# != 0
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

    def data(self, label:str=None, mask:np.ndarray=None, mask_value=0):
        """
        Return the array or evaluated recipe associated with the label

        :@param mask: if mask isn't None, applies the provided boolean mask
            with the same shape as this FeatureGrid to the base recipe before
            applying any transforms or recipes, such that any elements with a
            'True' value in the mask won't be included in the calculations.
        """
        if label is None:
            return np.copy(np.dstack(self._data))
        label = str(label)
        sequence = label.split(" ")
        base_label = sequence[-1]
        tran = sequence[:-1]
        if not self._label_exists(base_label):
            raise ValueError(f"Label {base_label} not recognized")
        X = np.copy(self._evaluate_recipe(base_label))
        if not mask is None:
            assert mask.shape == self._shape
            mask = mask.astype(bool)
            X[mask] = mask_value
        for tranfunc in [transforms[s] for s in tran[::-1]]:
            X = tranfunc(X)
        return X

    def add_recipe(self, label:str, recipe:Recipe):
        if self._label_exists(label) or label in transforms.keys():
            raise ValueError(f"Label {label} already exists.")
        assert type(recipe) is Recipe
        self._recipes[label] = recipe

    def extract_values(self, pixels:list, labels:list=None):
        """
        Extracts a (P,F) array for P pixels and F features with the provided
        labels using the provided 2-tuple pixel indeces
        """
        tmp_sg = np.dstack(self.subgrid(labels)._data)
        return np.vstack([tmp_sg[p] for p in pixels])

    def get_pixels(self, recipe:str, labels:list=None, show=False,
                   plot_spec={}, fill_color:tuple=(0,255,255)):
        """
        Enables the user to choose a pixel or series of pixels using a recipe
        basemap. After selecting the pixels, the chosen set of values can be
        optionally visualized with a bar plot if show=True.

        :@param recipe: Base-map of recipe used for pixel selection
        :@param labels: Optional list of labels to include in the bar plot
            when show=True.

        :@return: 2-tuple including a tuple of pixel indeces and a 2d array
            of data values with shape (P,F) for P pixels and F features.
            If a list of labels is provided, only the requested data will be
            extracted, in the order of the provided labels list.
        """
        pixels = gt.get_category(self.data(recipe),fill_color=fill_color)
        labels = self._labels if labels is None else labels
        # (P,F) array of values for P pixels and F features
        values = self.extract_values(pixels, labels)
        if show:
            stdevs = list(np.std(values, axis=0))
            means = list(np.mean(values, axis=0))
            gp.basic_bars(labels,means,err=stdevs, plot_spec=plot_spec)
        return tuple(pixels), values

    def do_mlc(self, select_recipe:str, categories:list, labels:list=None,
               threshold:float=None):
        """
        If this raises linear algebra errors, try selecting more samples.

        :@param select_recipe: String label of the feature or recipe available
            from this FeatureGrid to use to pick category pixel candidates
        :@param categories: List of unique string categories for each class of
            pixels you want to train mlc to recognize
        :@param labels: List of valid string labels for input arrays to include
            in maximum-likelihood classification.
        :@param threshold: Float confidence level in [0,1] below which pixels
            will be placed into a new 'uncertain' category.

        :@return: 2-tuple like (classified_ints, labels)
        """
        samples = {}
        for cat in categories:
            print(TF.BLUE("Select for category: ", bright=True)+
                  TF.WHITE(cat, bright=True, bold=True))
            samples[cat], _ = self.get_pixels(select_recipe, labels)
        class_ints, class_keys = classify.mlc(
                np.dstack(self.subgrid(labels=labels).data()),
                categories=samples,
                thresh=threshold
                )
        return class_ints, labels, samples

    def do_mdc(self, select_recipe:str, categories:list, labels:list=None):
        """
        :@param select_recipe: String label of the feature or recipe available
            from this FeatureGrid to use to pick category pixel candidates
        :@param categories: List of unique string categories for each class of
            pixels you want to train mlc to recognize
        :@param labels: List of valid string labels for input arrays to include
            in maximum-likelihood classification.
        """
        samples = {}
        for cat in categories:
            print(TF.BLUE("Select for category: ", bright=True)+
                  TF.WHITE(cat, bright=True, bold=True))
            samples[cat], _ = self.get_pixels(select_recipe, labels)
        classified, labels = classify.minimum_distance(
                np.dstack(self.subgrid(labels=labels).data()),
                categories=samples)
        return classified, labels, samples

    def _evaluate_recipe(self, recipe:str, mask:np.ndarray=None, mask_value=0):
        """
        Return evaluated recipe or base feature from a label

        :@param mask: if mask isn't None, applies the provided boolean mask
            with the same shape as this FeatureGrid to the base recipe before
            applying any transforms or recipes, such that any elements with a
            'True' value in the mask won't be included in the calculations.
        """
        if recipe in self.labels:
            if not mask is None:
                tmp_data = self._data[self.labels.index(recipe)]
                tmp_data[mask] = mask_value
                return tmp_data
            return self._data[self.labels.index(recipe)]
        elif recipe in self._recipes.keys():
            args = tuple(self.data(arg) for arg in self._recipes[recipe].args)
            if not mask is None:
                for a in args:
                    a[mask] = mask_value
            return self._recipes[recipe].func(*args)
        else:
            raise ValueError(f"{recipe} is not a valid recipe or label.")

    def get_nd_hist(self, labels:list, nbin=256):
        """
        Basic wrapper on the krttdkit.operate.enhace module tool for getting
        a sparse histogram in multiple dimensions. See the documentation for
        that method for details.

        :@param labels: Labels of each arrays to generate a histogram axis of.
        :@param nbin: Number of brightness bins in each array. This is
            a fixed number (256) by default, but you may provide a list of
            integer brightness bin counts corresponding to each label.

        :@return: 2-tuple like (H, coords) such that H and coords are arrays.

            H is a N-dimensional integer array of counts for each of the N
            provided arrays. Each dimension represents a different input
            array's histogram, and indeces of a dimension mark brightness
            values for that array.

            You can sum along axes in order to derive subsequent histograms.

            The 'coords' array is a length N list of numpy arrays. These arrays
            associate the corresponding dimension in H with actual brightness
            values in data coordinates. They may have different sizes since
            different bin_counts can be specified for each dimension.
        """
        return enh.get_nd_hist(
                arrays=tuple(self.data(l) for l in labels),
                bin_counts=nbin)

    def heatmap(self, label1, label2, nbin1=256, nbin2=256, show=False,
                ranges:list=None, fig_path:Path=None,
                plot_spec:dict={}):
        """
        Get a heatmap of the 2 arrays' values along with axis labels in data
        coordinates using the tool from krttdkit.enhance, which extracts

        :@param label1: Label of array for heatmap vertical axis. You can
            provide an array as well, just make sure it is the same size as
            the other label or array.
        :@param label2: Label of array for heatmap horizontal axis. As
            mentioned above, you can provide an array as well as long as the
            size is uniform.
        :@param nbin1: Number of brightness bins to use for the first dataset
        :@param nbin2: Number of brightness bins to use for the second dataset
        :@param ranges:
        :@param ranges: If ranges is defined, it must be a list of 2-tuple
            float value ranges like (min, max). This sets the boundaries for
            discretization in coordinate units, and thus sets the min/max
            values of the returned array, along with the mins if provided.
            Defaults to data range.
        :@param mins: If mins is defined, it must be a list of numbers for the
            minimum recognized value in the discretization. This sets the
            boundaries for discretization in coordinate units, and thus
            determines the min/max values of the returned array, along with
            any ranges. Defaults to data minimum
        :@param fig_path: Path to save the generated figure automatically
        :@param plot_spec: geoplot plot_spec dictionary with configuration
            options for geoplot.plot_heatmap when show=True
        """
        # Allow array arguments instead of labels.
        A1 = label1 if type(label1) == np.ndarray else self.data(label1)
        A2 = label2 if type(label2) == np.ndarray else self.data(label2)
        label1 = "ax1" if type(label1) == np.ndarray else label1
        label2 = "ax2" if type(label2) == np.ndarray else label2
        # Use the enhance tool to get a (nbin1, nbin2) integer array of counts
        # and axis coordinates in data values.
        M, coords = enh.get_nd_hist(
                arrays=(A1, A2),
                bin_counts=(nbin1, nbin2),
                ranges=ranges,
                )
        vcoords, hcoords = tuple(coords)
        if show or fig_path:
            # default plot_spec can be overwritten by parameter dict values
            def_ps = {
                    "ylabel":label1,
                    "xlabel":label2,
                    "cb_label":"counts",
                    "title":f"Brightness heatmap ({label2} vs {label1})",
                    "cmap":"gist_ncar",
                    "imshow_norm":"log",
                    "imshow_extent":(min(hcoords),max(hcoords),
                                     min(vcoords),max(vcoords)),
                    #"imshow_aspect":1,
                    }
            def_ps.update(plot_spec)
            gp.plot_heatmap(heatmap=M, plot_spec=def_ps, show=show,
                            fig_path=fig_path)
        return M, vcoords, hcoords


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
        of this FeatureGrid, returns a new FeatureGrid with subsetted arrays
        and subsetted labels if a labels array is provided.

        :@param labels: Ordered labels of arrays included in the subgrid
        :@param vrange: Vertical range in pixel index coordinates
        :@param hrange: Horizontal range in pixel index coordinates
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
        return np.dstack(tuple(map(self.data, (r,g,b))))

    def drop_data(self, label:str):
        """
        Drop the dataset with the provided label from this FeatureGrid.
        """
        i = self._labels.index(label)
        self._labels = list(self._labels[:i])+list(self._labels[i+1:])
        self._data = list(self._data[:i])+list(self._data[i+1:])
        self._info = list(self._info[:i])+list(self._info[i+1:])
        return self

    def get_bound(self, label, upper=True, lower=False, bg_recipe=None):
        """
        Use a trackbar to select an inclusive lower and/or upper bound for a
        feature. The feature must be 2d.
        """
        # Make sure the base array is a valid RGB
        base_arr = label if bg_recipe is None else bg_recipe
        if len(base_arr.shape)==2:
            base_arr = gt.scal_to_rgb(base_arr)
        def pick_lbound(X,v):
            """ Callback function for rendering the user's l-bound choice """
            global base_arr
            Xnew = enh.linear_gamma_stretch(np.copy(X))
            mask = Xnew<v/255
            if base_arr is None:
                Xnew[np.where(mask)] = 0
                base_arr = enh.linear_gamma_stretch(Xnew)
            bba = np.copy(np.asarray(base_arr))
            bba[np.where(mask)] = np.array([0,0,0])
            #bba = bba[:,:,::-1]
            return bba
        def pick_ubound(X,v):
            """ Callback function for rendering the user's u-bound choice """
            global base_arr
            Xnew = enh.linear_gamma_stretch(X)
            mask = Xnew>v/255
            if base_arr is None:
                Xnew[np.where(mask)] = np.amin(Xnew)
                base_arr = enh.linear_gamma_stretch(Xnew)
            bba = np.copy(np.asarray(base_arr))
            bba[np.where(mask)] = np.array([0,0,0])
            #bba = bba[:,:,::-1]
            return bba
        X = self.data(label)
        xmin = np.amin(X)
        xrange = np.amax(X)-np.amin(X)
        X = (X-xmin)/xrange
        if upper:
            bound = gt.trackbar_select(
                    X=X,
                    func=pick_ubound,
                    label=label,
                    ) * xrange + xmin
        if lower:
            bound = gt.trackbar_select(
                    X=X,
                    func=pick_lbound,
                    label=label,
                    ) * xrange + xmin


    def add_data(self, label:str, data:np.ndarray, info:dict=None,
                 extract_mask:bool=True):
        """
        Add a new data field to the FeatureGrid with an equally-shaped ndarray
        and a unique label. If this FeatureGrid has no data, this method will
        set the object's immutable shape attribute.

        :@param label: Unique label to identify the data array
        :@param data: 2d numpy array with identical shape to this FeatureGrid
        :@param info: Optional dictionary of attributes corresponding to this
            dataset, which can be useful for storing information for
            downstream applications.
        :@param extract_mask: if True and if the provided data is a
            MaskedArray, the mask will be

        :@return: None
        """
        label = str(label)
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

        if type(data) == np.ma.core.MaskedArray:
            if extract_mask:
                # Add the mask as a new feature array
                mask = np.ma.getmask(data).astype(bool)
                if np.any(mask):
                    self.add_data(label+"_mask", mask.astype(bool),
                                  {"name":"Boolean mask for "+label},
                                  extract_mask=False)
            # get rid of the mask
            data = np.asarray(data.data)
        self._labels.append(label)
        self._data.append(data)
        self._info.append(dict(info) if info else {})
        return self

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
        #label = str(label).lower()
        return label in self._labels or label in self._recipes.keys() \
                or label in transforms.keys()

