from pathlib import Path
import numpy as np
import pickle as pkl
from datetime import datetime
import multiprocessing as mp


def window_slide(X:np.ndarray, window_size:int, times:list=None):
    """
    Implementation of the moving window transform to generate a rolling series
    of samples from each axis after axis 0 in the provided array. If the
    provided array's first axis isn't divisible by the window size, the
    remainder of time steps will be ignored. The "new" axis with the provided
    window size will be the second dimension in the returned array, and the
    2nd dimension onward will be pushed back by a dimension and unaffected.

    The new first dimension will have size T - T%W - W for T timesteps and a
    window size of W.

    If a list of times are provided, returns a 2-tuple assigning a timestep
    to each sample in the first dimension. This is just a convenient way of
    keeping track of the time interval being trained on.

    :@param X: Array with the first axis representing timesteps that are
        subject to the window transform.
    :@param window_size: Number of timesteps to include in each sample.
    :@param times: Optional list of timesteps returned alongside the
        array, which helps identify the timestep each sample corresponds to,
        or in other words the last time -not- included in the corresponding
        element of the first axis in the look-back array.
    """
    if times:
        assert len(times)==X.shape[0]
    ntimes = X.shape[0]
    wdw_array = []
    for i in range(ntimes-(ntimes%window_size)-window_size):
        wdw_array.append(X[i:i+window_size])
    if times:
        times = times[window_size:ntimes-(ntimes%window_size)]
        return np.stack(wdw_array), times
    return np.stack(wdw_array)

def double_window_slide(X:np.ndarray, look_back:int, look_forward:int,
                        Y:np.ndarray=None, times:int=None):
    """
    Uses the sliding window transform to generate arrays for input and output
    sequences per the relevant multi-input multi-output standard. This is the
    de-facto way of generating sequence -> sequence datasets for MTS prediction
    problems in my current pipeline.

    In the simplest case...

     - X = [1,2,3,4,5,6,7,8,9,10]
     - look_back = 5
     - look_forward = 3

    The look-back array will have shape (3,5) and the look-forward array will
    have shape (3,3) such that, next to each other, back | forward looks like:

    1 2 3 4 5 | 6 7 8
    2 3 4 5 6 | 7 8 9
    3 4 5 6 7 | 8 9 10

    :@param X: Array with a first dimension that corresponds to the axis that
        should be "wrapped" into samples by the sliding window transform. If a
        Y array is provided, this array only contributes to the look-back array
        Otherwise, it is wrapped into both the look-back and look-forward.
    :@param look_back: Number of timesteps to include in each sample in the
        look-back array; equal to the size of the second dimension in the
        'X' look-back array.
    :@param look_forward: Number of timesteps to include in each sample in the
        look-forward array; equal to the size of the second dimension in the
        'Y' look-forward array.
    :@param Y: Optional separate array from which to derive output values for
        the look-forward array, which typically serves as a stand-in for the
        predicted domain of a seq->seq problem. This is useful for saving
        memory when the codomain is a subset or separate set of values than
        the input domain 'X'. The 'Y' array MUST have the same-size first axis
        as the 'X' array such that each indexed value represents the same time
        step in both arrays.
    :@param times: Optional list of timesteps corresponding to each index in
        the input array(s) Useful for keeping track of the time represented by
        the generated samples in the forward and backwards arrays.

    :@return: 2-tuple like (back_array, forward_array) where back_array and
        forward_array are (T,W,F) shaped arrays for T samples (same between
        both arrays), W window size (look_back, and look_forward respectively),
        and F is the number of features in 'X' or 'Y'. There may be additional
        trailing dimensions if, for example, X was originally 3d or 4d.

        NOTE: If a times array was provided, the returned value will be a
        3-tuple like (back_array, forward_array, sample_times) such that
        sample_times is a subset of the provided times array with each element
        corresponding to the first timestep AFTER the lookback window, which
        is also the EARLIEST value in the look-forward window for each time
        step.
    """
    Y = Y if not Y is None else np.copy(X)
    assert X.shape[0] == Y.shape[0]
    if times:
        assert len(times) == X.shape[0]
    X = window_slide(X[:-look_forward], look_back)
    Y = window_slide(Y[look_back:], look_forward)

    # Different look-back and look-forward window sizes can result in different
    # sample counts, but the first N samples between the two arrays will
    # correspond to the right timestep pairings. It's easier to just take the
    # largest number of valid samples shared between the two arrays than to
    # implement the modular arithmetic :)
    if X.shape[0]>Y.shape[0]:
        X = X[:Y.shape[0]]
    elif X.shape[0]<Y.shape[0]:
        Y = Y[:X.shape[0]]
    if times:
        times = times[look_back:look_back+X.shape[0]]
        return (X, Y, times)
    return (X, Y)

def gauss_norm(X:np.ndarray, axis=-1, mask:np.ndarray=None):
    """
    Independently normalize each feature (sub-array) along the provided axis,
    returning the scaled array along with 2 tuples. This method is general
    enough to handle any data shape.

    :@param X: Numpy array to normalize
    :@param axis: Axis indexing features that should be independently normed.
    :@param mask: Optional boolean mask with the same shape as X. Values masked
        'True' will be ignored in mean and standard devia calculations, but
        will be scaled along with the returned un-masked array.
    :@return: 3-tuple like (norm_array, means, stdevs), where norm_array is
        the gauss-normalized data, and means/stdevs are 1d numpy arrays with
        elements corresponding to each element in the specified axis.
    """
    # Convert any negative axes labels to positive
    axis = (axis+len(X.shape))%len(X.shape)
    mask = mask if not mask is None else np.full_like(X, False)
    M = np.ma.array(X, mask=mask)
    # Make broadcastable vectors for mean and standard deviation
    data_axes = tuple([i for i in range(len(X.shape)) if i!=axis])
    means = np.expand_dims(
            np.ma.mean(X, axis=data_axes),
            axis=data_axes)
    stdevs = np.expand_dims(
            np.ma.std(X, axis=data_axes),
            axis=data_axes)
    M = (M-means)/stdevs
    return M.data, np.squeeze(means).data, np.squeeze(stdevs).data

if __name__=="__main__":
    pass
