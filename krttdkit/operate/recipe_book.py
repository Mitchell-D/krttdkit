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
    return gt.scal_to_rgb(enh.linear_gamma_stretch(X))

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

# todo: find way to pass runtime parameters to transforms, probably
# by abstracting them into a class like Recipe
transforms = {
        "lingamma": lambda X: enh.linear_gamma_stretch(X),
        "histeq": lambda X: enh.histogram_equalize(
            X, nbins=256)[0].astype(np.uint8),
        "norm256": lambda X: enh.norm_to_uint(X, 256, np.uint8),
        "selectgamma":selectgamma,
        "colorize":colorize,
        }
