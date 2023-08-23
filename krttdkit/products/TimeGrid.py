from pathlib import Path
from datetime import datetime
import numpy as np
import pickle as pkl
import multiprocessing as mp

def _extract_pixels(args):
    """ """
    pixels, path = args
    X = np.load(path.as_posix())
    return np.array([ X[p] for p in pixels])

class TimeGrid:
    """
    TimeGrid is a class that abstracts a list of uniform-shaped numpy arrays
    stored serial files, which constitute a time series. The time series
    members should be (M,N,F)-shaped for M vertical coordinates, N horizontal
    coordinates, and F labeled features. The class also supports (M,N)-shaped
    static arrays that assign a scalar to each point in the member grids.

    Ultimately, the goal of this class is to provide a generalized way of
    extracting subsets of the time series given an ordered list of desired
    features, a time range, and a list of indeces.

    Currently the storage medium is simple ".npy" files for each time step
    since ".npz" files increase access time in exchange for disc space
    (since decompression takes time). Same principle with structured arrays.
    I'm assuming disc space isn't a limiting factor, and that file size only
    really matters when transferring data, in which case archive compression
    algorithms like tar/gzip
    """
    def __init__(self, time_file_tuples, labels):
        """
        Initialize a TimeGrid with a list of files, acquisition times, and
        feature labels corresponding to the 3rd axis of each array.

        This method does not load the data, but ensures that the provided files
        constitute a continuous and uniform-interval time series per the
        datetime object provided with each file.

        This method also doesn't validate shape constraints of the arrays
        contained in each file, namely that each array must have the same shape
        and that the length of the third axis must match the number of provided
        feature labels. For a (computationally expensive) sanity check, see the
        validate() object method.

        :@param time_file_tuples: list of (file_datetime, file_path) 2-tuples
            for each file in a unique dataset's time series. file_datetime
            must be a datetime object, and file_path must be a Path object to a
            valid file. Furthermore, each file must be a ".npy" serial binary,
            each with a (M,N,F) shape for M vertical coordinates, N horizontal
            coordinates, and F features (see labels)
        :labels: Ordered list of unique string label corresponding to each of
            the features specified in the third array axis of each ".npy" grid.
        """
        tfts = time_file_tuples
        # All file list members must be 2-tuples
        assert all(type(t)==tuple and len(t)==2 for t in tfts)
        # Sort files by their datetime component
        tfts.sort(key=lambda t:t[0])
        times, paths = zip(*tfts)
        # All file paths must be existing ".npy" file Path objects
        for p in paths:
            if not type(p)==Path and p.exists() and p.suffix=="npy":
                raise ValueError(f"Invalid Path: {p}")
        # All timesteps must be equal-interval
        dt = times[1]-times[0]
        for t0,t1 in zip(times[:-1],times[1:]):
            if not t1-t0==dt:
                raise ValueError(
                        f"Default time step ({dt}) not abided by ({t0},{t1})")
        self._paths = paths
        self._times = times
        self._labels = labels

    @property
    def size(self):
        return len(times)
    @property
    def times(self):
        return self._times
    @property
    def labels(self):
        return self._labels
    @property
    def paths(self):
        return self._paths

    def get_grid(self, time_idx:int, features:list=None):
        """
        Returns the (M,N,F) grid associated with a timestep index and the
        requested features, in the order of provided labels. If no features
        are provided, all features will be returned in the order of the
        original labels list.

        :@param time_idx: Time step index of the desired grid.
        :@param features: List of unique sring labels that all match one of
            the originally-provided labels.
        """
        assert type(idx) is int and 0<=idx<self.size
        features = features if features else self._labels
        assert all(f in self._labels for f in features)
        feat_idx = [self._labels.index(f) for f in features]
        return np.load(self._paths[time_idx])[:,:,feat_idx]

    def validate(self):
        """
        Validates that the arrays contained in all ".npy" files of a dataset
        are uniformly-shaped and have the same number of features (third axis)
        as feature labels. This is a computationally costly operation.
        """
        shape = np.load(self._paths[0], mmap_mode="r").shape
        # All arrays in the dataset must have the same shape
        if not all([np.load(p, mmap_mode="r").shape==shape
                    for p in self._paths]):
            raise ValueError(
                    f"Not all arrays in datset {dataset} have the same shape!")
        # Arrays must have the same number of features as feature labels
        nlabels = len(self._labels)
        if not shape[2]==nlabels:
            raise ValueError(
                    f"Dataset {dataset} must have the same number of "
                    f"features as feature labels ({shape[2]} != {nlabels})")

    def subset(self, t0:datetime=None, tf:datetime=None):
        """
        Returns a new TimeGrid in the provided time range, and with the
        provided features in the order provided.

        :@param t0: Inclusive initial datetime of the new returned TimeGrid.
            If t0 is less than the first time, the returned TimeGrid will start
            with the first supported index, and no error will be raised.
        :@param tf: Exclusive final datetime of the new returned TimeGrid. If
            the final time in this TimeGrid is less than the provided final
            time, the returned TimeGrid will end with the last supported time.
        """
        if t0 and tf:
            assert t0<tf
        if not t0 and not tf:
            return self
        idx0 = self._times.index(next(t for t in self._times if t>=t0)
                                 ) if t0 else 0
        idxf = self._times.index(next(t for t in self._times[::-1] if t<tf)
                                 )+1 if tf else len(self._times)
        return TimeGrid(
                list(zip(self._times[idx0:idxf], self._paths[idx0:idxf])),
                self._labels)

    def extract_timeseries(self, pixels:list, features:list=None,
                           nworkers:int=1):
        """
        Returns a 2-tuple like (times, arrays), where 'times' is a list of T
        datetimes, and 'arrays' is a list of (T,F) shaped arrays for T times
        and F features. The index of each member of the array list corresponds
        to order of the provided pixel list, and the index of each feature
        corresponds to the order of the provided features list.

        :@param pixels: List of 2-tuple pixel indeces on the arrays handled by
            this TimeGrid. Indeces correspond to each pixel extracted as a
            1-D time series in the returned arrays.
        :@param features: List of valid string feature labels supported by
            the registered arrays. The order of the features list determines
            the order of the 2nd axis of each returned pixel array.
        """
        assert all(type(p)==tuple and len(p)==2 for p in pixels)
        features = features if features else self._labels
        assert all(f in self._labels for f in features)
        # Parse all pixels from each file in the time series
        with mp.Pool(nworkers) as pool:
            args = [(pixels, p) for p in self._paths]
            results = np.dstack(pool.map(_extract_pixels, args))
        # Reshape the array to (times, features, pixels)
        results = np.transpose(results, [2,1,0])
        # Use fancy indexing to only keep the requested features, in the same
        # order as the provided list.
        fidx = [self._labels.index(f) for f in features]
        results = results[:,fidx,:]
        return [results[...,i] for i in range(results.shape[-1])]

if __name__=="__main__":
    tg_dir = Path("data/subgrids/")
    nldas_paths = [p for p in tg_dir.iterdir()
                   if p.stem.split("_")[0]=="FORA0125"]
    nldas = [(datetime.strptime(p.stem.split("_")[1], "%Y%m%d-%H"),p)
             for p in nldas_paths ]
    TG = TimeGrid()
