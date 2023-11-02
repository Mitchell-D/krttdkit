"""
General collection of recipes from any platform, and transforms
(which are 1:1 functions that may apply to any FeatureGrid)
"""
import numpy as np
from krttdkit.operate.Recipe import Recipe
from krttdkit.operate.abi_recipes import abi_recipes
import krttdkit.operate.enhance as enh
import krttdkit.visualize.guitools as gt

def colorize(X:np.ndarray):
    assert len(X.shape)==2
    return gt.scal_to_rgb(enh.linear_gamma_stretch(X), hue_range=(0,.5))

def selectgamma(X:np.ndarray, offset=-2.5, scale=5):
    print(f"Gamma range: [{offset}, {offset+scale}]")
    def _get_gamma(A,v):
        """ Given a value from 0 to 255, scale the array by gamma """
        v = ((1+v)/256)*scale+offset
        return enh.linear_gamma_stretch(X,v)

    if len(X.shape)==3:
        assert X.shape[2]==3
        return np.dstack([selectgamma(X[:,:,i],offset,scale)
                          for i in range(3)])

    response = gt.trackbar_select(X=X, func=_get_gamma, label="Gamma exponent")
    return enh.linear_gamma_stretch(X, ((1+response)/256)*scale+offset)

def gaussnorm(X:np.ndarray, radius=3):
    """
    Scale data values to the corresponding mean and standard deviation
    gaussian distribution, saturating values outside of the radius
    of standard deviations on either side of the mean.
    """
    print(X.shape)
    X1 = X-np.average(X)
    Xnorm = X1/np.std(X)
    return np.clip(Xnorm, -abs(radius), abs(radius))

def histgauss(X):
    if len(X.shape)==3:
        return np.stack([histgauss(X[...,i]) for i in range(X.shape[-1])],
                        axis=-1)
    N = gt.get_normal_array(*X.shape[:2])
    return enh.histogram_match(X, N, nbins=1024)

def fft2d(X):
    return enh.dft2D(X, inverse=False, use_scipy=True)
def ifft2d(X):
    return enh.dft2D(X, inverse=True, use_scipy=True)
def logfft2d(X):
    return np.log(1+np.abs(enh.dft2D(X, inverse=False, use_scipy=True)))
def ilogfft2d(X):
    return np.exp(enh.dft2D(X, inverse=True, use_scipy=True))-1

def selectlowpass(X):
    res = 512
    yroll, xroll = X.shape[0]//2, X.shape[1]//2
    dy, dx = np.meshgrid(map(np.arange, X.shape))
    D = enh.linear_gamma_stretch(((dy-yroll)**2+(dx-xroll)**2)**(1/2))
    def lowpass(X,v):
        enh.dft2D(X, inverse=True)

    return trackbarselect(resolution=res)

# todo: find way to pass runtime parameters to transforms, probably
# by abstracting them into a class like Recipe
transforms = {
        "norm1": lambda X: enh.linear_gamma_stretch(X),
        "histeq": lambda X: enh.histogram_equalize(
            X, nbins=256)[0].astype(np.uint8),
        "histgauss":histgauss,
        "norm256": lambda X: enh.norm_to_uint(X, 256, np.uint8),
        "selectgamma":selectgamma,
        "selectlowpass":selectlowpass,
        "colorize":colorize,
        "gaussnorm":gaussnorm,
        "fft2d":fft2d,
        "ifft2d":ifft2d,
        "logfft2d":logfft2d,
        "ilogfft2d":ilogfft2d,
        "fft2d":fft2d,
        }
