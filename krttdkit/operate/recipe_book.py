"""
General collection of recipes from any platform, and transforms
(which are 1:1 functions that may apply to any FeatureGrid)
"""
import numpy as np
from krttdkit.operate.Recipe import Recipe
from krttdkit.operate.abi_recipes import abi_recipes
import krttdkit.operate.enhance as enh
import krttdkit.visualize.guitools as gt

def selectgamma():
    return

# todo: find way to pass runtime parameters to transforms, probably
# by abstracting them into a class like Recipe
transforms = {
        "lingamma": lambda X: enh.linear_gamma_stretch(X),
        "histeq": lambda X: enh.histogram_equalize(
            X, nbins=256)[0].astype(np.uint8),
        "norm256": lambda X: enh.norm_to_uint(X, 256, np.uint8),
        "selectgamma":None,
        }
