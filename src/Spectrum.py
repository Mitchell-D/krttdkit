import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import copy
import math as m
import re

from pathlib import Path
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d

class Spectrum:
    def __init__(self, figsize:tuple=(40,10), data_threshold:int=None):
        """
        :param figsize: numpy figure (width, height)
        """
        self._data = {}
        self._lines = {}
        self._figsize = figsize
        self._xrange = None
        self._yrange = None
        self._thresh = data_threshold
        # Spectal data
        self.clear_figure()
        # Universal x and y range of figure, determined by first data loaded.

    def get_xrange(self, data_key:str):
        self._check_key(data_key, warn=True, should_exist=True)
        return (self._data[data_key][0,0], self._data[data_key][-1,0])

    def get_yrange(self, data_key:str):
        self._check_key(data_key, warn=True, should_exist=True)
        return (min(self._data[data_key][:,1]),
                max(self._data[data_key][:,1]))

    def copy_data(self, data_key_from:str, data_key_new:str):
        """
        Copy the data from one data key to a new one.
        """
        self._check_key(data_key_from, warn=True, should_exist=True)
        self._check_key(data_key_new, warn=True, should_exist=False)
        self._data[data_key_new] = copy.deepcopy(self._data[data_key_from])

    def delete_data(self, data_key:str):
        """ Removes a dataset from the collection via its key """
        self._check_key(data_key, warn=True, should_exist=True)
        del self._data[data_key]

    def clear_figure(self):
        """
        Clear the matplotlib state machine and generate a new figure
        and primary axes.
        """
        plt.clf()
        self._fig, self._ax_host = plt.subplots(figsize=self._figsize)
        self._host_occupied = False

    @property
    def data_keys(self):
        return tuple(self._data.keys())

    def info(self, data_key:str):
        """
        Print useful information about a loaded dataset
        """
        self._check_key(data_key, warn=True, should_exist=True)
        datum = self._data[data_key]
        print(f"Info on {data_key}:")
        print(f"\tsize: {datum.shape[0]}")
        print(f"\tx range: ({datum[0,0]}, {datum[-1,0]})")
        print(f"\ty range: ({min(datum[:,1])}, {max(datum[:,1])})")
        print(f"\ty mean: {sum(datum[:,1])/len(datum)}")
        print(f"\tfirst dx: {datum[1,0]-datum[0,0]}")

    def subset(self, data_key:str, xrange:tuple):
        """
        Reduce the data's domain to the provided range (inclusively).
        This modifies the data associated with the data_key. Data not included
        in the given xrange is deleted.
        """
        self._check_key(data_key, warn=True, should_exist=True)
        init = self._find_xindex(data_key, xrange[0],
                                 mode="more", strict=True)
        final = self._find_xindex(data_key, xrange[1],
                                  mode="less", strict=True)
        self._data[data_key] = self._data[data_key][init:final,:]

    def plot_range(self, data_key:str, xrange:list=None, label:str="",
                   color:str="#3399ff", alpha:float=1.0, plot_type:str="line",
                   linewidth:float=1):
        """ """
        plot_types = ("fill", "scatter", "line")
        self._check_key(data_key, warn=True, should_exist=True)
        if not xrange:
            xrange = (self._data[data_key][0,0], self._data[data_key][-1,0])
        if not len(xrange)==2:
            raise ValueError(f"xrange must be a 2-tuple (not {xrange})")
        if not xrange[0]<xrange[1] \
                and type(xrange[0])==int \
                and type(xrange[1])==int:
            raise ValueError(
                    "xrange values must be ints such that " + \
                    f"xrange[0]<xrange[1] (provided {xrange})")
        if not plot_type in plot_types:
            print(f"plot_type must be one of {plot_types}")

        # Find the index of x-values closest to the requested range
        init = self._find_xindex(data_key, xrange[0])
        final = self._find_xindex(data_key, xrange[1])
        """
        print(f"Found index range:\n\tinit (index {init}): " + \
                f"({self._data[data_key][init, 0]}, {self._data[data_key][init, 1]})" + \
                f"\n\tfinal (index {final}): " + \
                f"({self._data[data_key][final, 0]}, {self._data[data_key][final, 1]})")
        """

        if not self._host_occupied:
            self._xrange = xrange
            new_ax = self._ax_host
            self._host_occupied = True

        else:
            new_ax = self._ax_host.twinx()
            new_ax.set_frame_on(False)
            new_ax.patch.set_visible(False)
            for sp in new_ax.spines.values():
                sp.set_visible(False)

        #  Make sure the y range encapsulates all the data
        ymax = max(self._data[data_key][:,1])
        ymin = min(self._data[data_key][:,1])
        if not self._yrange:
            self._yrange = [ymin, ymax+ymax/12]
        else:
            if self._yrange[0]>ymin:
                self._yrange[0] = ymin
            if self._yrange[1]<ymax:
                self._yrange[1] = ymax+ymax/12

        #  Make sure the x range encapsulates all the data
        if not self._xrange:
            self._xrange = xrange
        else:
            if xrange[0]<self._xrange[0]:
                self._xrange[0] = xrange[0]
            if xrange[1]>self._xrange[1]:
                self._xrange[1] = xrange[1]


        #  Plot the data on the axes according to the selected option.
        if plot_type == "scatter":
            line = new_ax.scatter(self._data[data_key][init:final,0],
                           self._data[data_key][init:final,1],
                           label=label,
                           color=color,
                           alpha=alpha)
        elif plot_type == "line":
            line = new_ax.plot(self._data[data_key][init:final,0],
                        self._data[data_key][init:final,1],
                        label=label,
                        color=color,
                        alpha=alpha,
                        linewidth=linewidth,)
        elif plot_type == "fill":
            line = new_ax.fill(self._data[data_key][init:final,0],
                        self._data[data_key][init:final,1],
                        label=label,
                        color=color,
                        alpha=alpha,
                        linewidth=linewidth,)

        # Index zero here since plotting functions can return multiple lines.
        self._lines.update({data_key:line[0]})


    def save_fig(self, figure_path:Path, title:str="", xticks:int=10,
                 yticks:int=10, xlabel:str="", ylabel:str="",
                 font_size:int=20, padding:float=0,
                 legend_dims:tuple=(0.5, 0.5, 0.0, 0.0)):
        """
        :param legend_dims: Tuple of float values (x,y,dx,dy) where...
                - x: x position of top right of legend when dx=0
                - y: y position of top right of legend when dy=0
                - dx: rightward offset of legend wrt x position
                - dy: topward offset of legend wrt y position
        """
        labels = []
        for ax in self._fig.axes:
            ax.set_xlim(self._xrange[0], self._xrange[1])
            ax.set_ylim(self._yrange[0], self._yrange[1])
            ax.tick_params(labelsize=font_size*.8)

        plt.locator_params(axis="x", nbins=xticks)
        plt.locator_params(axis="y", nbins=yticks)
        plt.tight_layout(pad=padding)
        self._ax_host.set_ylabel(ylabel, fontsize=font_size)
        self._ax_host.set_xlabel(xlabel, fontsize=font_size)
        self._fig.legend(
                 [self._lines[l].get_label() for l in self._lines.keys()],
                 fontsize=font_size*1,
                 bbox_to_anchor=legend_dims,
                 )
        self._fig.suptitle(title, fontsize=font_size*1.5)
        self._fig.savefig(figure_path.as_posix())

    def _find_xindex(self, data_key:str, xvalue:tuple, axis=0,
                     mode:str="closest", strict:bool=False):
        """
        Searches the 0-column of the ndarray (which is supposed to be
        monotonic) for the nearest value to the provided xvalue.

        :param mode: Determines how the index is selected
            - "less" returns index of element with greatest value that's less
              than the provided value
            - "more" returns index of element with least value that's greater
              than the provided value
            - "closest" return the index of the element numerically closest
              to the provided value
        """
        self._check_key(data_key, warn=True, should_exist=True)
        if strict and (xvalue<self._data[data_key][0,0] \
                or xvalue>self._data[data_key][-1,0]):
            raise ValueError(f"{xvalue} isn't in range of {data_key}")
        idx = np.searchsorted(self._data[data_key][:,0], xvalue, side="left")
        if mode=="closest":
            if idx > 0 and (idx == len(self._data[data_key][:,0]) or \
                    m.fabs(xvalue-self._data[data_key][:,0][idx-1]) \
                        < m.fabs(xvalue-self._data[data_key][:,0][idx])):
                idx -= 1
        elif mode=="less":
            idx -= 1
        return idx


    def _check_key(self, data_key:str, warn:bool=False,
                   should_exist:bool=True):
        """
        Raises a warning if a data key is provided that already exists in
        the _data attribute. If a key should exist and doesn't, raises an
        error if "warn" is True. Similarly if a key shouldn't exist and does,
        prints a warning. In any case, returns bool indicating whether the
        key is present.
        """
        included = data_key in self._data.keys()
        if included and not should_exist:
            print(f'WARNING: Data with key "{data_key}" has already been ' + \
                    "loaded! The previous data will be overwritten.")
        if not included and should_exist:
            raise ValueError(
                f"{data_key} is not a valid data key in {self._data.keys()}")
        return included

    def read_datfile(self, data_path:Path, data_key:str):
        """
        Reads the and validates the csv data in the datfile format provided by
        the Gemini telescope organization.
        """
        self._check_key(data_key, warn=True, should_exist=False)
        data_path = self._validate_path(data_path)
        with open(data_path, "r") as datfp:
            lines = datfp.readlines()
        newlines = []
        for l in lines:
            l = l.strip()
            l = re.sub("\s{2,}", ",", l)
            l = l.replace("\n", "")
            l = l.split(",")
            l = list(map(float, l))
            newlines.append(l)

        self._data[data_key] = self._validate_data(np.asarray(newlines))

    def save_to_pkl(self, data_key:str, pkl_path:Path):
        """
        Save the ndarray referenced by data_key to a new pickle file at
        pkl_path.
        """
        self._check_key(data_key, warn=True, should_exist=True)
        pkl_path = self._validate_path(pkl_path, exists=False)
        with open(pkl_path.as_posix(), "wb") as pklfp:
            pkl.dump(self._data[data_key], pklfp)

    def load_pkl(self, pkl_path:Path, data_key:str):
        """
        Load a pickle assumed to have an ndarray with the standard dimensions
        into the provided data_key
        """
        pkl_path = self._validate_path(pkl_path)
        self._check_key(data_key, warn=True, should_exist=False)
        with open(pkl_path.as_posix(), "rb") as pklfp:
            self._data[data_key] = self._validate_data(pkl.load(pklfp))

    def load_ndarray(self, data:np.ndarray, data_key:str):
        """
        Load an ndarray with standard dimensions to the provided data_key
        """
        self._check_key(data_key, warn=True, should_exist=False)
        self._data[data_key] = self._validate_data(pkl.load(pklfp))


    @staticmethod
    def _validate_path(path:Path, exists:bool=True):
        """
        Verify that a file path exists and path is not a directory
        :param path: Path object of data file
        :param exists: if True, raises an error if the file DOESN'T exist.
        """
        if exists and not path.exists():
            raise ValueError(f"Path must exist: {path.as_posix()}")

        if not exists and path.exists():
            raise ValueError(f"Path must NOT exist: {path.as_posix()}")

        if path.is_dir():
            raise ValueError(f"Path cannot be a directory: {path.as_posix()}")

        return path

    def _validate_data(self, data:np.ndarray):
        """
        Verifies that a dataset is shaped (N,2) for N>1 and that N is
        ordered monotonically (generally assumed to be float wavelength in
        microns).
        """
        if data.dtype not in (np.float64,):
            raise ValueError(f"Invalid dtype ({data.dtype})")

        if data.shape[1] != 2:
            raise ValueError("array data ")

        if data.shape[0] < 2:
            raise ValueError("There must be at least 2 data values.")

        prev = data[0,0]
        for i in range(1, data.shape[0]-1):
            current = data[i,0]
            if current <= prev:
                raise ValueError("Spectrum data must increase monotonically.")
            prev = current

        init_diff = data[1, 0]-data[0,0]

        if self._thresh:
            for i in range(1, data.shape[0]-1):
                new_diff = data[i+1, 0] - data[i, 0]
                if int(self._thresh*(new_diff-init_diff)):
                    raise ValueError(
                            "Spectrum data does not have uniform intervals.")
        return data

    def interpolate(self, data_key_from:str, data_key_to:str, data_key_new:str,
                    function:None):
        """
        Interpolates data from one dataset onto the x-axis of a different
        dataset. The data being interpolated must necessarily span a larger
        x range than the data with the x-axis being interpolated onto.

        :param data_key_from: Dataset with a y values that will be linearally
                interpolated onto the x-axis of data_key_to. The range of x
                values must be a superset of the x-values in data_key_to.
        :param data_key_to: Dataset with an x-axis that data_key_from will be
                interpolated onto.
        :param data_key_new: New data key of the derived dataset with an
                x-axis identical to data_key_to, and y values that are
                interpolated from data_key_from or interpolated then used as
                parameters of the operation before being stored.
        :param function: function to be iteratively applied across the two
                source datasets. The first argument of the function is a
                interpolated y-value derived from data_key_from, and the second
                argument is the corresponding y-value from data_key_from at
                the same x. The output is stored wrt the x value in the
                generated array.
        """
        self._check_key(data_key_to, warn=True, should_exist=True)
        self._check_key(data_key_from, warn=True, should_exist=True)
        self._check_key(data_key_new, warn=True, should_exist=False)
        to = self._data[data_key_to]
        frm = self._data[data_key_from]
        if to[0,0]<frm[0,0] or to[-1,0]>frm[-1,0]:
            raise ValueError("The dataset being interpolated " + \
                    f"({data_key_from}) must have an x-axis that spans " + \
                    f"the entire range of {data_key_to}")
        interp = interp1d(frm[:,0], frm[:,1])
        newdata = []
        for i in range(to.shape[0]):
            if function:
                newdata.append(function(interp(to[i,0]), to[i,1]))
            else:
                newdata.append(interp(to[i,0]))
            #newdata.append(interp[i,0] if not function \
            #        else function(interp(to[i,0]), to[0,1]))
        self._data[data_key_new] = np.dstack(
                (to[:,0],np.asarray(newdata)))[0,:,:]

if __name__=="__main__":
    cerro_pkl = Path("./data/cerro-pachon_transmittance-moist_0900-5600.pkl")
    mauna_pkl = Path("./data/mauna-kea_transmittance-moist_0900-5600.pkl")
    cerro_csv = Path("./data/cerro-pachon_transmittance-moist_0900-5600.csv")
    mauna_csv = Path("./data/mauna-kea_transmittance-moist_0900-5600.csv")
    band4_response_pkl = Path("./data/goes_srf/goes-r_abi-fm2_srf_band-4.pkl")
    band5_response_pkl = Path("./data/goes_srf/goes-r_abi-fm2_srf_band-5.pkl")
    band6_response_pkl = Path("./data/goes_srf/goes-r_abi-fm2_srf_band-6.pkl")
    band7_response_pkl = Path("./data/goes_srf/goes-r_abi-fm2_srf_band-7.pkl")
    s = Spectrum(figsize=(70, 30))
    #s.read_datfile(mauna_csv, "mauna")
    #s.save_to_pkl("mauna", mauna_pkl)
    #s.load_pkl(cerro_pkl, "cerro")
    #s.load_pkl(band4_response_pkl, "band4")
    #s.load_pkl(band5_response_pkl, "band5")
    #s.load_pkl(band6_response_pkl, "band6")
    #s.load_pkl(band7_response_pkl, "band7")
