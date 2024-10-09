import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from scipy.fft import fft,ifft
#from numba import jit

# Kernels recognized by kernel_convolve() method
kernels = {
        "horizontal":[[-1,-1,-1], # Horizontal edge
                      [ 0, 0, 0],
                      [ 1, 1, 1]],
        "vertical":[[-1, 0, 1], # Vertical edge
                    [-1, 0, 1],
                    [-1, 0, 1]],
        "diagonal_bck":[[ 0, 1, 1], # Back diagonal
                        [-1, 0, 1],
                        [-1,-1, 0]],
        "diagonal_fwd":[[ 1, 1, 0], # Forward diagonal
                        [ 1, 0,-1],
                        [ 0,-1,-1]],
        "sobel_1":[[ 1, 2, 1],
                   [ 0, 0, 0],
                   [-1,-2,-1]],
        "sobel_2":[[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]],
        }

#def visualize_fourier(X:np.ndarray):
def log_fourier(X:np.ndarray):
    """ Return the provided array in natural log-scaled phase space """
    return np.log(1+np.abs(fft2(X)))

def border_mask(X:np.ndarray, cutoff:int, high_freq:bool=True, fill=None):
    """
    Applies a radius (high-pass/low-pass) to X centered on and returns a
    reconstructed version of the array.

    :@param X: 2d array of brightnesses to apply filter to.
    :@param radius: filter radius in pixels from center=(y,x).
    :@param center: integer 2-tuple like (y,x) of the center pixel in frequency
            space of the filter.
    :@param true_inside: If True (by default), applies low-pass filter (ie)
            frequencies inside the radius are kept. Otherwise applies a
            high-pass filter.
    """
    if high_freq:
        X[-cutoff:,-cutoff:] = fill
    else:
        X[:cutoff,:cutoff] = fill
    return X

def radius_mask(X:np.ndarray, radius:float=None, center:tuple=None,
                  true_inside:bool=True, fill=None):
    """
    Applies a radius (high-pass/low-pass) to X centered on and returns a
    reconstructed version of the array.

    :@param X: 2d array of brightnesses to apply filter to.
    :@param radius: filter radius in pixels from center=(y,x).
    :@param center: integer 2-tuple like (y,x) of the center pixel in frequency
            space of the filter.
    :@param true_inside: If True (by default), applies low-pass filter (ie)
            frequencies inside the radius are kept. Otherwise applies a
            high-pass filter.
    """
    cy, cx = center if center else map(lambda v: int(v/2), X.shape)
    radius = min(cx,cy,X.shape[0]-cy,X.shape[1]+cx) \
            if radius is None else radius
    # cool static constructor
    y_coords, x_coords = np.ogrid[:X.shape[0], :X.shape[1]]
    R = np.sqrt( (x_coords-cx)**2 + (y_coords-cy)**2 )
    mask = R > radius if true_inside else R < radius
    if fill is None:
        return mask
    X[mask] = fill
    return X

def dft2D(X:np.ndarray, inverse:bool=False, use_scipy:bool=True):
    if use_scipy:
        row_pass = np.stack([
            (fft,ifft)[inverse](X[i,:])
            for i in range(X.shape[0])
            ], axis=0)
        col_pass = np.stack([
            (fft,ifft)[inverse](row_pass[:,j])
            for j in range(row_pass.shape[1])
            ], axis=1)
        return col_pass

    row_pass = np.stack([discrete_fourier(X[i,:], inverse=inverse)
                          for i in range(X.shape[0])], axis=0)
    col_pass = np.stack([discrete_fourier(row_pass[:,j], inverse=inverse)
                          for j in range(row_pass.shape[1])], axis=1)
    return col_pass

def discrete_fourier(X:np.ndarray, inverse:bool=False, use_scipy:bool=False):
    """
    """
    # radix-2
    X = np.array(X, dtype=float)
    N = X.shape[0]
    assert len(X.shape)==1
    #if N==1:
    if N<=2:
        X = np.asarray(X, dtype=float)
        n = np.arange(X.shape[0])
        k = n.reshape((X.shape[0],1))
        return np.dot(np.exp([-2j,2j][inverse]*np.pi*k*n/X.shape[0]), X)
    odd_phase = discrete_fourier(X[1::2])
    even_phase = discrete_fourier(X[::2])
    phasor = np.exp([-2j, 2j][inverse]*np.pi*np.arange(N)/N)
    #get_phase = lambda k: np.exp(phasor, k)
    #Z = get_phase(odd_phase) # ?
    #Y = np.concatenate([ even_phase + Z, even_phase-Z ])
    #return Y/N if inverse else Y
    Y = np.concatenate([ even_phase + phasor[:int(N/2)] * odd_phase,
                        even_phase + phasor[int(N/2):] * odd_phase ])
    return Y

def naive_dft(X):
    n,m = X.shape
    w = math.e**(-2j*math.pi)
    Y = np.zeros_like(X)
    for k in range(m):
        for l in range(n):
            for j in range(n):
                row = X[j,:]
                row_const = w**(k*j/n)
                for i in range(m):
                    Y[k,l] = X[j,i] * w**(l*i/m) * row_const
    return Y/(m*n)

def kernel_convolve(X, kernel_name:str, scipy_mode:str="valid"):
    """
    Simple wrapper function on scipy.signal.convolve2d that selects
    from preconfigured kernels.
    """
    if kernel_name not in kernels.keys():
        raise ValueError(f"Provided kernel name {kernel_name} not one of",
                         list(kernels.keys()))
    return convolve2d(X, kernels[kernel_name], mode=scipy_mode)

def multi_edge(X:np.ndarray, sequence:bool=False):
    """
    Applies 4 edge detection kernels and returns an array of the total spatial
    gradient along each axis. Optionally apply the kernels as a sequence.
    """
    assert len(X.shape)==2 and all([ length>2 for length in X.shape ])
    kernel_keys = ("horizontal", "vertical",
                   "diagonal_bck", "diagonal_fwd")
    if sequence:
        for key in kernel_keys:
            X = convolve2d(X, kernels[key], mode="valid")
        grad = X
    else:
        convolutions = []
        for k in [ kernels[key] for key in kernel_keys ]:
            convolutions.append(convolve2d(X, k, mode="valid"))
        grad = np.sqrt(np.sum(np.dstack(
            [ convolve2d(X, k, mode="valid")**2 ]), axis=2))
    return grad

def sobel_edge(X:np.ndarray, weight:float=1):
    """
    Applies the Sobel algorithm edge detection kernels to the provided
    array and returns the corresponding diagonal gradient array as
    a 2d array with 2 fewer elements on each axis (no fill values).
    """
    assert len(X.shape)==2 and all([ length>2 for length in X.shape ])
    grad = np.zeros_like(X[:-2,:-2])
    kernel1 = np.asarray(kernels["sobel_1"])*weight
    kernel2 = np.asarray(kernels["sobel_2"])*weight
    for i in range(X.shape[0]-2):
        for j in range(X.shape[1]-2):
            del1 = np.sum(X[i:i+3,j:j+3]*kernel1)
            del2 = np.sum(X[i:i+3,j:j+3]*kernel2)
            grad[i,j] = math.sqrt(del1**2 + del2**2)
    return grad

def roberts_edge(X:np.ndarray):
    """
    Applies roberts operator to a 2d array, which calculates the "diagonal"
    gradients of pixels and returns a 2d array with one less element on
    each axis due to forward differencing.
    """
    grad = np.zeros_like(X[:-1,:-1])
    assert len(X.shape)==2 and all([ length>1 for length in X.shape ])
    for i in range(X.shape[0]-1):
        for j in range(X.shape[1]-1):
            del1 = X[i,j]-X[i+1, j+1] # Back diagonal
            del2 = X[i+1,j]-X[i, j+1] # Forward diagonal
            grad[i,j] = math.sqrt(del1**2 + del2**2)
    return grad

def color_average(pixels):
    """
    Given a list of RGB pixels, calculates the average along each color axis.
    """
    return [ sum([p[i] for p in pixels])/len(pixels) for i in range(3) ]

def linear_gamma_stretch(X:np.ndarray, lower:float=None, upper:float=None,
                         gamma:float=1, report_min_and_range:bool=False):
    """
    Linear-normalize pixel values between lower and upper bound, then apply
    gamma stretching if an argument is provided.

    By default, normalizes the full data range to [0,1]

    :@param X: any numpy ndarray or masked array to normalize
    :@param lower: lower bound to normalize between; This determines the
            value in data coordinates that will be considered 'zero'
    :@param upper: upper bound to normalize between;
    :@param gamma: X^(1/gamma) exponent.
    :@param report_min_and_range: if True, returns a 3-tuple containing
            the minimum value and the range in original dataset units.

    :@return: An array of float values in range [0, 1] normalized in bounds.
    """
    Xrange = [np.amin(X), np.amax(X)]
    lower = lower if not lower is None else Xrange[0]
    upper = upper if not upper is None else Xrange[1]
    Xnew = ((X-lower)/(upper-lower))**(1/gamma)
    if not report_min_and_range:
        return Xnew
    return (Xnew, np.amin(X), np.amax(X)-np.amin(X))

def gamma(X:np.ndarray, gamma, a=1.):
    """
    Applies basic gamma transform to all data points in X according to
    Y = a*X**(1/gamma) without performing any normalization.
    """
    return a*X**(1/gamma)

def norm_to_uint(X:np.ndarray, resolution:int, cast_type:type=np.uint,
                 norm=True):
    """
    Linearally normalizes the provided float array to bins between 0 and
    resolution-1, and returns the new integer array as np.uint.

    :@param resolution: Final integer resolution in brightness bins
    :@param cast_type: Integer type to cast the array to.
    :@param norm: If True, normalizes the array between its minimum and
            maximum values. Otherwise, only normalizes to the maximum. If
            False, the array must have all positive values, or else it can't
            be casted to a uint type. cast_type can be a float or otherwise,
            despite the name of this method.
    """
    if not norm:
        return (np.round(X/np.amax(X))*(resolution-1)).astype(cast_type)
    return (np.round(linear_gamma_stretch(X)*(resolution-1))).astype(cast_type)

def vertical_nearest_neighbor(X:np.ma.MaskedArray, debug=False):
    """
    Linearly interpolates masked values of a 2d array along axis 0,
    independently for each column using nearest-neighbor interpolation.
    If you need horizontal interpolation, transpose the array.

    This method is intended for VIIRS bowtie correction as a cosmetic
    correction, but may be used for other purposes.
    """
    print(f"Masked values: {np.count_nonzero(X.mask)}")
    if len(X.shape) != 2:
        raise ValueError(f"Array must be 2d; provided array shape: {X.shape}")

    if debug:
        print("Vertical-NN interpolating " + \
                f"{X.size-np.count_nonzero(X.mask)} points.")
    for i in range(X.shape[1]):
        col = X[:,i]
        valid = np.where(np.logical_not(col.mask))[0]
        f = interp1d(valid, col[valid], fill_value="extrapolate")
        X[:,i] = np.array(f(range(X.shape[0])))
    return X

def array_stat(X:np.ndarray):
    """
    Returns a dict of useful info about an array.
    """
    return {
            "shape":X.shape,
            "size":X.size,
            "stddev":np.std(X),
            "mean":np.average(X),
            "min":np.amin(X),
            "max":np.amax(X),
            "range":np.ptp(X),
            "nanmin":np.nanmin(X),
            "nanmax":np.nanmax(X),
            "nancount":np.count_nonzero(np.isnan(X)),
            }

def get_nd_hist(arrays:list, bin_counts=256, ranges:list=None):
    """
    MASK values with np.nan in order to exclude them from the counting

    :@param arrays: List of arbitrary-dimensional numpy arrays with uniform
        size. The order of these arrays corresponds to the order of the
        dimensions in the returned array of counts.
    :@param bin_counts: Integer number of bins to use for all provided arrays
        (on a per-axis scale) or list of integer bin counts for each array.
    :@param ranges: If ranges is defined, it must be a list of 2-tuple float
        value ranges like (min, max). This sets the boundaries for
        discretization in coordinate units, and thus sets the min/max values
        of the returned array, along with the mins if provided.
        Defaults to data range.
    :@param mins: If mins is defined, it must be a list of numbers for the
        minimum recognized value in the discretization. This sets the
        boundaries for discretization in coordinate units, and thus determines
        the min/max values of the returned array, along with any ranges.
        Defaults to data minimm

    :@return: 2-tuple like (H, coords) such that H and coords are arrays.

        H is a N-dimensional integer array of counts for each of the N provided
        arrays. Each dimension represents a different input array's histogram,
        and indeces of a dimension mark brightness values for that array.

        You can sum along axes in order to derive subsequent histograms.

        The 'coords' array is a length N list of numpy arrays. These arrays
        associate the corresponding dimension in H with actual brightness
        values in data coordinates. They may have different sizes since
        different bin_counts can be specified for each dimension.
    """
    s0 = arrays[0].size
    assert all(a.size==s0 for a in arrays)
    if type(bin_counts) is int:
        bin_counts = np.asarray([256 for i in range(len(arrays))])
    else:
        assert all(type(c)==int for c in bin_counts)
        assert len(bin_counts) == len(arrays)
        bin_counts = np.asarray(bin_counts)
    # Get a (P,F) array of P unmasked pixel values with F features
    X = np.stack(tuple(map(np.ravel, arrays))).T
    valid = np.logical_not(np.any(np.ma.getmask(np.ma.masked_invalid(X)),
                                  axis=1))
    # Normalize the unmasked values
    Y = X[valid]
    '''
    if not mins is None:
        assert len(mins)==len(arrays)
        mins = np.asarray(mins)
    else:
        mins = np.amin(Y, axis=0)
    '''
    if not ranges is None:
        assert len(ranges)==len(arrays)
        assert all(type(r) == tuple and len(r) == 2 for r in ranges)
        mins, maxes = map(np.asarray, zip(*ranges))
        ranges = maxes-mins
    else:
        mins = np.min(Y, axis=0)
        ranges = np.amax(Y, axis=0)-mins

    # Y is normalized to [0,1] independently in each dimension
    Y = (Y-np.broadcast_to(mins, Y.shape))/np.broadcast_to(ranges, Y.shape)
    Y = np.clip(Y,0,1)
    # Scale the float to the desired number of bins
    Y *= np.broadcast_to(bin_counts, Y.shape)-1
    # Round the bins to the nearest integer to discretize
    Y = np.ceil(np.clip(Y,0,None)).astype(np.uint)
    # discretize Y to the appropriate number of bins
    H = np.zeros(bin_counts)
    for i in range(Y.shape[0]):
        ybounds = tuple(Y[i])
        H[ybounds[0],ybounds[1]] += 1
    # Coordinates are the minimum value in each bin
    coords = [np.array([ranges[i]*j/bin_counts[i]+mins[i]
                        for j in range(bin_counts[i])])
              for i in range(len(bin_counts))]
    return H, coords

# old version, pending removal
def get_heatmap(X:np.ndarray, nbins, debug=False):
    """
    Note that the returned heat map is indexed from the 'top left' by
    the imaginary standard, so the y axis must be flipped before plotting.
    (Except for imshow, which does this for you)

    :@param X: (M,N,2) array with 2 independent variables
    :@param nbins: Number of bins to sample from X, or the side length
            of the returned square array.
    """
    X[:,:,0] = linear_gamma_stretch(X[:,:,0])
    X[:,:,1] = linear_gamma_stretch(X[:,:,1])
    if len(X.shape)==1:
        return get_pixel_counts(X, nbins)
    X = np.dstack((norm_to_uint(X[:,:,0], nbins),
                   norm_to_uint(X[:,:,1], nbins)))
    H = np.zeros((nbins, nbins))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            H[ X[i,j,0], X[i,j,1]] += 1
    print("DEPRECATION WARNING: Use updated and more general get_heatmap_nd()")
    return H


def get_pixel_counts(X:np.ndarray, nbins, debug=False):
    """
    Returns an integer array with length nbins which depicts the number of
    pixels with values in the index bin.

    Index zero corresponds to the bin from Xmin to Xmin + bin_size

    :@param X: any numpy ndarray with values that can be uniformly binned
    :@param nbins: Number data values ('resolution') to bin X into.
    :@return: tuple like (counts, bin_size, Xmin) containing an array with
            size nbins with the pixel count in each bin in X, a float value
            bin_size indicating the bin size in data coordinates, and a float
            value Xmin indicating the floor of the fist bin in data coords.
    """
    #X = X.compressed() if isinstance(X, np.ma.MaskedArray) else X.flatten()
    # Get all unmasked values
    if isinstance(X, np.ma.MaskedArray):
        if debug: print(f"Handling masked array")
        Xdata = X.compressed()
    else:
        Xdata = X.ravel()
    Xmin = np.amin(Xdata)
    Xmax = np.amax(Xdata)
    bin_size = (Xmax-Xmin)/nbins
    if debug:
        print(f"Binning {X.size} unmasked data points")
        print(f"Original array range: ({Xmin}, {Xmax})")
    counts = np.zeros(nbins)
    Xnorm = norm_to_uint(Xdata, nbins, cast_type=np.ulonglong)
    for px in Xnorm:
        try:
            counts[px] += 1
        except Exception as e:
            continue
    return counts, bin_size, Xmin

def get_cumulative_hist(X:np.ndarray, nbins:int, debug=False):
    """
    Get a cumulative array of binned pixel values for equalization

    :@param X: any numpy ndarray with values that can be uniformly binned
    :@param nbins: Number data values ('resolution') to bin X into.
    :@return: tuple like (counts, bin_size, Xmin) containing an array with
            size nbins with the cumulative pixel count up to that brightness
            level for each bin in X, a float value bin_size indicating the bin
            size in data coordinates, and a float value Xmin indicating the
            floor of the fist bin in data coords.
    """
    total = 0
    counts, bin_size, Xmin = get_pixel_counts(X, nbins)
    if debug: print("Accumulating histogram counts")
    for i in range(counts.size):
        total += counts[i]
        counts[i] = total
    return counts, bin_size, Xmin

def histogram_match(X:np.ndarray, Y:np.ndarray, nbins:int):
    """
    Do histogram matching between (M,N) or (M,N,3) arrays X and Y with nbins
    brightness levels, and return the resulting array.

    If X is a masked array, only histogram-matches unmasked values, setting
    masked values to the minimum of the array.
    """
    is_rgb = lambda A:len(A.shape)==3 and A.shape[2]==3
    if is_rgb(X) and is_rgb(Y):
        return np.dstack([
            histogram_match(np.copy(X)[:,:,i],np.copy(Y)[:,:,i], nbins)
            for i in range(3)])
    yc_hist, dy, ymin = get_cumulative_hist(Y, nbins)
    xc_hist, _, _ = get_cumulative_hist(X, nbins)
    if type(X) == np.ma.MaskedArray:
        mask = X.mask
        normX = X.data
        normX[np.where(mask)] = np.amin(normX)
        normX = norm_to_uint(normX, nbins)
    else:
        normX = norm_to_uint(X, nbins)
    matched = np.zeros_like(normX)
    yc_hist = (yc_hist * (nbins-1)/np.amax(yc_hist)).astype(int)
    xc_hist = (xc_hist * (nbins-1)/np.amax(xc_hist)).astype(int)
    for i in range(matched.shape[0]):
        for j in range(matched.shape[1]):
            matched[i,j] = np.argmin(np.abs(xc_hist[normX[i,j]]-yc_hist))
    return matched


def histogram_equalize(X:np.ndarray, nbins:int=512,
                       cumulative_histogram:np.array=None, debug=False):
    """
    Get a histogram-equalized version of X

    Y = (N-1)/S * C(X)
    Where X is the corrected array with N brightness bins and S pixels, C(X)
    is the cumulative number of pixels up to the brightness bin of pixel X.

    :@param X: any numpy ndarray with values that can be uniformly binned
    :@param nbins: Number data values ('resolution') to bin X into.
    :@param cumulative_histogram: 1-d array describing a custom cumulative
            histogram curve for correcting X. Array size must be nbins. If a
            histogram is provided, bin_size and Xmin are unknowable and will be
            returned as None.
    :@return: tuple like (counts, bin_size, Xmin) containing an array with
            size nbins with the cumulative pixel count up to that brightness
            level for each bin in X, a float value bin_size indicating the bin
            size in data coordinates, and a float value Xmin indicating the
            floor of the fist bin in data coords.
    """
    if cumulative_histogram is None:
        c_hist, bin_size, Xmin = get_cumulative_hist(X, nbins)
    else:
        if cumulative_histogram.size != nbins:
            raise ValueError(
                    f"Provided histogram must have {nbins} members, not " +
                    str(cumulative_histogram.size))
        c_hist = cumulative_histogram
        bin_size = None
        Xmin = None

    hist_constant = (nbins-1)/X.size
    normed = norm_to_uint(X, nbins)
    hist_bins = norm_to_uint(hist_constant*c_hist, nbins)
    Y = np.zeros_like(normed)
    for i in range(normed.shape[0]):
        for j in range(normed.shape[1]):
            Y[i,j] = hist_bins[normed[i,j]]

    #Y = [[  for j in X.shape[1]] for i in X.shape[0]]
    #Y = np.vectorize(lambda px: hist_scale*c_hist[px])(normed)
    #Y = np.vectorize(lambda px: hist_scale[px])(normed)
    return Y, bin_size, Xmin

def linear_contrast(X:np.ndarray, a:float=1, b:float=0, debug=False):
    """
    Perform linear contrast stretching on the provided array, and return the
    result as an integer. Simple linear equation y = ax+b with no
    normalization.
    """
    xmin = np.amin(X)
    xmax = np.amax(X)
    return np.clip(X*a+b, xmin, xmax)

def saturated_linear_contrast(X:np.ndarray, nbins:int=None,
                              lower_sat_pct:float=0, upper_sat_pct:float=1):
    """
    Perform saturated contrast stretching on an image, mapping the full range
    of brightness values to the
    """
    if not lower_sat_pct < upper_sat_pct <= 1:
        raise ValueError(f"The lower saturation percentile must be less " + \
                "than the upper percentile, and both must be less than 1.")
    X = linear_gamma_stretch(X)
    X[np.where(X <= lower_sat_pct)] = lower_sat_pct
    X[np.where(X >= upper_sat_pct)] = upper_sat_pct
    return norm_to_uint(X, nbins) if nbins else X

def do_histogram_analysis(X:np.ndarray, nbins:int, equalize:bool=False,
                          debug=False):
    """
    High-level helper method aggregating the frequency and cumulative
    frequency histograms of a provided array using n normalized value bins
    with a histogram-equalized array

    :@param X: any numpy ndarray with values that can be uniformly binned
    :@param nbins: Number data values ('resolution') to bin X into.
    :@param equalize: if True, the histogram equalization algorithm is
            applied and returned dictionary
    """
    freq, bin_size, Xmin = get_pixel_counts(X, nbins, debug=debug)
    cumulative_freq, _, _ = get_cumulative_hist(X, nbins, debug=debug)
    hist_dict= {
            "px_count":X.size,
            "hist":freq/X.size,
            "c_hist":cumulative_freq/X.size,
            "domain":np.linspace(Xmin, Xmin+nbins*bin_size, nbins),
            "bin_size":bin_size,
            "Xmin":Xmin,
            "stddev":np.std(X),
            "mean":np.average(X),
            "equalized":None,
            }

    if equalize:
        hist_dict["equalized"], _, _ = histogram_equalize(
                X, nbins, cumulative_freq, debug=debug)

    return hist_dict

