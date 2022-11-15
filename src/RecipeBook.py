"""
Helper methods for generating RGBs and scaler products using 2d or 3d
(time series) numpy ndarrays for the appropriate bands.
"""
import numpy as np
from .ABIManager import ABIManager

def cimss_truecolor(band1:np.ndarray, band2:np.ndarray, band3:np.ndarray):
    """
    Performs RGB color correction on the provided red, green, and blue
    ndarrays, which may be 2d single-frame values from ABI bands 1, 2, and 3,
    or 3d time series. In either case, the arrays must have the same shape.

    This function uses correction values that are tailored towards the
    relative spectral response of ABI band 3 reflectance to green-colored
    surface materials.

    :return: (R, G, B) tuple of ndarrays that have been color-corrected
        according to the CIMSS Natural True Color RGB recipe.

    Recipe reference:
    https://www.star.nesdis.noaa.gov/GOES/documents/ABIQuickGuide_CIMSSRGB_v2.pdf
    """

    # Normalize the reflectance to exclusive (0,1).
    # If the L1b radiances were actually converted, all the grid values
    # are probably already well within this range.
    B = np.clip(band1, 0, 1)
    R = np.clip(band2, 0, 1)
    G = np.clip(band3, 0, 1)

    # Gamma correction using gamma=2.2 for standard digital displays
    # (ref: https://en.wikipedia.org/wiki/Gamma_correction)
    gamma = 2.2
    B = np.power(B, 1/gamma)
    R = np.power(R, 1/gamma)
    G = np.power(G, 1/gamma)

    #print(type(B), type(R), type(G))
    #print(B.shape, R.shape, G.shape)

    # Get "True color" green according to CIMSS recipe, and collect the RGB
    # (ref: CIMSS Natural True Color Quick Guide)
    G_TC = np.clip(.45*R+.1*G+.45*B, 0, 1)

    return (R, G_TC, B)

def airmass(f6p2um:np.ndarray, f7p3um:np.ndarray, f9p6um:np.ndarray,
            f10p3um:np.ndarray):
    """
    Generate an airmass RGB according to the Air Mass RGB Quick Guide.
    Should generally be used for 2d and 3d (time series) ndarrays.
    """
    # Temperatures are in differential degrees celsius
    R = f6p2um-f7p3um
    G = f9p6um-f10p3um
    B = f6p2um-273.15
    # Normalize to recipe value ranges.
    R = np.clip((R--26.2)/(0.6--26.2), 0, 1)
    G = np.clip((G--42.2)/(6.7--42.2), 0, 1)
    B = np.clip((B--64.65)/(-29.25--64.65), 0, 1)
    # Invert Blue channel
    B = 1-B
    return (R, G, B)

def ndsii1(red:np.ndarray, swir:np.ndarray):
    """
    Use the ndsii1 recipe to transform the provided
    dimensionally equivalent ndarrays of scalar values.
    """
    return (red-swir)/(red+swir)

def watervapor(f6p2um:np.ndarray, f7p3um:np.ndarray, f10p3um:np.ndarray):
    """
    Use the ABI Water Vapor RGB Quick Guide recipe to generate a 2d or
    3d (time series)
    """
    # Temperatures are in differential degrees celsius
    R = f10p3um-273.15
    G = f6p2um-273.15
    B = f7p3um-273.15

    # Normalize to values provided in the Simple Water Vapor RGB Quick Guide.
    R = np.clip(1-(R--70.86)/(5.81--70.86), 0, 1)
    G = np.clip(1-(G--58.49)/(-30.48--58.49), 0, 1)
    B = np.clip(1-(B--28.03)/(-12.12--28.03), 0, 1)

    # Invert Blue channel
    B = 1-B
    return (R, G, B)
