"""
ABI L1b recipes from specific sources
"""
import numpy as np
from krttdkit.operate import Recipe

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

    # Get "True color" green according to CIMSS recipe, and collect the RGB
    # (ref: CIMSS Natural True Color Quick Guide)
    G_TC = np.clip(.45*R+.1*G+.45*B, 0, 1)

    return np.dstack((R, G_TC, B))

def airmass(f6p2um:np.ndarray, f7p3um:np.ndarray, f9p6um:np.ndarray,
            f10p3um:np.ndarray):
    """
    Generate an airmass RGB according to the Air Mass RGB Quick Guide.
    Should generally be used for 2d and 3d (time series) ndarrays.
    """
    if f6p2um.shape==3:
        return np.dstack(tuple(
            airmass(f6p2um[:,:,i], f7p3um[:,:,i],
                    f9p6um[:,:,i], f10p3um[:,:,i]) \
                            for i in range(len(f6p2.shape[2]))))
    # Temperatures are in differential degrees celsius
    R = f6p2um-f7p3um
    G = f9p6um-f10p3um
    B = f6p2um-273.15
    # Normalize to recipe value ranges.
    #R = np.clip((R--26.2)/(0.6--26.2), 0, 1)
    #G = np.clip((G--42.2)/(6.7--42.2), 0, 1)
    #B = np.clip((B--64.65)/(-29.25--64.65), 0, 1)
    R = (R--26.2)/(0.6--26.2)
    G = (G--42.2)/(6.7--42.2)
    B = (B--64.65)/(-29.25--64.65)
    # Invert Blue channel
    B = 1-B
    return np.dstack((R, G, B))

def ndsii1(red:np.ndarray, swir:np.ndarray):
    """
    Use the ndsii1 recipe to transform the provided
    dimensionally equivalent ndarrays of scalar values.
    """
    return (red-swir)/(red+swir)

def _gamma_norm(A:np.ndarray, floor=None, cap=None, gamma=1):
    """
    Color correction
    (ref: https://en.wikipedia.org/wiki/Gamma_correction)
    """
    if floor is None:
        floor = np.amin(A)
    if cap is None:
        cap = np.max(A)
    A[A<floor] = floor
    A[A>cap] = cap
    A = ((A-floor) / (cap-floor)) ** (1/gamma)
    return A

def day_cloud_phase(band2:np.ndarray, band4:np.ndarray, band5:np.ndarray):
    """
    https://cimss.ssec.wisc.edu/training/QuickGuides/QuickGuide_GOESR_Day_Cloud_Type.pdf
    """
    #R = np.power(band4, 1/1) # Originally .66
    #G = np.power(band2, 1/2) # Originally 1
    #B = np.power(band5, 1/2) # Originally 1
    R = _gamma_norm(band4, gamma=.66)
    G = _gamma_norm(band2, gamma=1)
    B = _gamma_norm(band5, gamma=1)
    return np.dstack((R, G, B))


def diffwv(f6p2um:np.ndarray, f7p3um:np.ndarray):
    """
    Use the differential water vapor RGB recipe to generate a 2d or
    3d (time series) RGB product.

    Expects 6.2um and 7.3um band ndarrays in Kelvin brightness temps
    """
    G = f7p3um-273.15 # Low level water vapor
    B = f6p2um-273.15 # Upper level water vapor
    R = G-B           # Vertical water vapor difference

    # Set data floors and ceilings
    #R = 1-np.clip(_gamma_norm(R, floor=-3, cap=30,  gamma=.2587), 0, 1)
    #G = 1-np.clip(_gamma_norm(G, floor=-60, cap=5,  gamma=.4), 0, 1)
    #B = 1-np.clip(_gamma_norm(B, floor=-64.65, cap=-29.25,  gamma=.4), 0, 1)
    R = 1-_gamma_norm(R, floor=-3, cap=30,  gamma=.2587)
    G = 1-_gamma_norm(G, floor=-60, cap=5,  gamma=.4)
    B = 1-_gamma_norm(B, floor=-64.65, cap=-29.25,  gamma=.4)

    # Gamma correction
    return np.dstack((R, G, B))

def ntmicro(f3p9um:np.ndarray, f10p4um:np.ndarray, f12p4um:np.ndarray):
    """
    https://weather.msfc.nasa.gov/sport/training/quickGuides/rgb/QuickGuide_NtMicro_GOESR_NASA_SPoRT_20191206.pdf

    Expects 2d or 3d (time series) arrays with equal sizes and data in
    Kelvin brightness temperatures.
    """
    R = f12p4um-f10p4um # Cloud optical depth
    G = f10p4um-f3p9um  # Partical size/phase
    B = f10p4um-273.15  # Surface temperature

    R = _gamma_norm(R, floor=-6.7, cap=2.6)
    G = _gamma_norm(G, floor=-3.1, cap=5.2)
    B = _gamma_norm(B, floor=-29.6, cap=19.5)

    return np.dstack((R, G, B))

def viirs_cloud_type(m9:np.ndarray, m5:np.ndarray, m10:np.ndarray):
    """
    Recipe: https://www.eumetsat.int/media/42198
    """
    R = (m9/.1)**(1/1.5)
    G = (m5/.8)**(1/0.75)
    B = (m10/.8)

    return (R, G, B)

def viirs_cloud_phase(m10:np.ndarray, m11:np.ndarray, m3:np.ndarray):
    """
    Recipe: https://www.eumetsat.int/media/42197
    """
    R = m10/.5
    G = m11/.5
    return (R, G, m3)

def watervapor(f6p2um:np.ndarray, f7p3um:np.ndarray, f10p3um:np.ndarray):
    """
    Use the ABI Water Vapor RGB Quick Guide recipe to generate a 2d or
    3d (time series)
    https://rammb.cira.colostate.edu/training/visit/quick_guides/QuickGuide_GOESR_DifferentialWaterVaporRGB_final.pdf
    """
    # Temperatures are in differential degrees celsius
    R = f10p3um-273.15
    G = f6p2um-273.15
    B = f7p3um-273.15

    # Normalize to values provided in the Simple Water Vapor RGB Quick Guide.
    #R = np.clip(1-(R--70.86)/(5.81--70.86), 0, 1)
    #G = np.clip(1-(G--58.49)/(-30.48--58.49), 0, 1)
    #B = np.clip(1-(B--28.03)/(-12.12--28.03), 0, 1)
    R = np.clip(1-_gamma_norm(R, cap=5.81, floor=-70.86), 0, 1)
    G = np.clip(1-_gamma_norm(G, cap=-30.48, floor=-58.49), 0, 1)
    B = np.clip(1-_gamma_norm(B, cap=-12.12, floor=-28.03), 0, 1)

    # Invert Blue channel
    #B = 1-B
    #G = 1-G
    #R = 1-R
    return np.dstack((R, G, B))

def sport_dust(b11, b13, b14, b15):
    ## Optical depth
    R = np.clip(_gamma_norm(b15-b13, -6.7, 2.6, 1), 0, 1)
    ## Particle phase; small -> ice and dust
    G = np.clip(_gamma_norm(b14-b11, -0.5, 20, 2.5), 0, 1)
    ## surface temp
    B = np.clip(_gamma_norm(b13, -11.95, 15.55, 1), 0, 1)
    return np.dstack((R, G, B))

abi_recipes = {
        "dust":Recipe(
            args=("11-tb", "13-tb", "14-tb", "15-tb"),
            func=sport_dust,
            name="SPoRT Dust RGB",
            ref="https://rammb.cira.colostate.edu/training/visit/"
                "quick_guides/Dust_RGB_Quick_Guide.pdf",
            ),
        "truecolor":Recipe(
            args=("1-ref","2-ref","3-ref"),
            func=cimss_truecolor,
            name="CIMSS Truecolor",
            ref="https://www.star.nesdis.noaa.gov/GOES/documents/" + \
                    "ABIQuickGuide_CIMSSRGB_v2.pdf"
            ),
        "airmass":Recipe(
            args=("8-tb","10-tb","12-tb","13-tb"),
            func=airmass,
            name="SPoRT Airmass RGB"
            ),
        "dcp":Recipe(
            args=("2-ref","4-ref","5-ref"),
            func=day_cloud_phase,
            name="CIMSS Day cloud phase RGB"
            ),
        "diffwv":Recipe(
            args=("8-tb","10-tb"),
            func=diffwv,
            name="Differential water vapor RGB"
            ),
        "ntmicro":Recipe(
            args=("7-tb","13-tb","15-tb"),
            func=ntmicro,
            name="SPoRT night-time microphysics"
            ),
        "wv":Recipe(
            args=("8-tb", "10-tb", "13-tb"),
            func=watervapor,
            name="CIRA water vapor"
            )
        }
