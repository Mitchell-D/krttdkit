"""
The operate module contains functions for operating on data. The sub-modules
are designed to be as data-agnostic and dependency-free as possible, without
side-effects from graphics libraries or network utilities.

classify provides methods for classifying 2d feature grids using supervised
and unsupervised techniques.

enhance contains methods for applying a variety functions to scalar arrays.
This includes methods for edge detection, gamma/contrast, histogram
equalization/matching, intensity heatmaps, discretization, and fourier stuff.

geo_helpers has methods that assist with common geometric and geographic
tasks like snapping a point to a coordinate grid, calculating panoramic
distortion, and (eventually) doing interpolation of data onto a new coordinate
grid.
"""
