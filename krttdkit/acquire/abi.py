"""
Collection of functions for downloading and parsing NOAA GOES ABI netCDF files
into a common format
"""
import netCDF4 as nc
import numpy as np
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from krttdkit.acquire import get_goes as gg
from krttdkit.acquire.get_goes import parse_goes_path
from krttdkit.products import GeosGeom

"""
Dictionary mapping ABI band numbers to critical information about them
"""
bands={
        1:{"ctr_wl":0.47,
           "name":"Visible Blue",
           "default_res":1,
            "kappa0":0.0015839,
           },
        2:{"ctr_wl":0.64,
           "name":"Visible Red 0",
           "default_res":.5,
            "kappa0":0.0019586,
           },
        3:{"ctr_wl":0.86,
           "name":"Near-Infrared Veggie",
           "default_res":1,
            "kappa0":0.0033384,
           },
        4:{"ctr_wl":1.37,
           "name":"Near-Infrared Cirrus",
           "default_res":2,
            "kappa0":0.008853,
           },
        5:{"ctr_wl":1.6,
           "name":"Near-Infrared Snow/Ice",
           "default_res":1,
            "kappa0":0.0131734,
           },
        6:{"ctr_wl":2.2,
           "name":"Near-Infrared Cloud particle size",
           "default_res":2,
            "kappa0":0.0415484,
           },
        7:{"ctr_wl":3.9,
           "name":"Infrared Shortwave window",
           "default_res":2,
           },
        8:{"ctr_wl":6.2,
           "name":"Infrared Upper-level water vapor",
           "default_res":2,
           },
        9:{"ctr_wl":6.9,
           "name":"Infrared Midlevel water vapor",
           "default_res":2,
           },
        10:{"ctr_wl":7.3,
            "name":"Infrared Lower-level water vapor",
            "default_res":2,
            },
        11:{"ctr_wl":8.4,
            "name":"Infrared Cloud-top phase",
            "default_res":2,
            },
        12:{"ctr_wl":9.6,
            "name":"Infrared Ozone",
            "default_res":2,
            },
        13:{"ctr_wl":10.3,
            "name":"Infrared 'Clean' longwave window",
            "default_res":2,
            },
        14:{"ctr_wl":11.2,
            "name":"Infrared Longwave window",
            "default_res":2,
            },
        15:{"ctr_wl":12.3,
            "name":"Infrared 'Dirty' longwave window",
            "default_res":2,
            },
        16:{"ctr_wl":13.3,
            "name":"Infrared CO2 longwave",
            "default_res":2,
            },
        }

def download_l2_abi(data_dir:Path, product:str, target_time:datetime,
                    time_window:timedelta=None, satellite="16", replace=False):
    """
    Downloads L2 data specified by the product string to data_dir.
    Only downloads closest file to target_time if timedelta is None,
    or downloads all files in range described by target_time+time_window
    if a time window is provided.
    """
    goesapi = gg.GetGOES()
    l2_prod = gg.GOES_Product(satellite, "ABI", "L2", product.upper())
    if time_window is None:
        queue = goesapi.get_closest_to_time(
                product=l2_prod, target_time=target_time)
    else:
        queue = goesapi.search_hour(
                product=l2_prod, init_time=target_time,
                final_time=target_time+time_window)
    return [goesapi.download(f, data_dir, replace=False) for f in queue]

def get_l2(data_file:Path, include_latlon=True, include_angles=False):
    """
    Parse a supported GOES ABI L2 product file type

    :@param data_file: ABI L2 ABI netCDF file type
    :@param include_latlon: If True, includes 'lat' and 'lon' features using
        grids calculated from the geostationary geometry of the satellite.
    :@param include_angles: If True, includes the north/south and east/west
        scan angles (from NADIR) in radians for VZA calculations.
    """
    file = gg.parse_goes_path(data_file)
    assert file.product.sensor == "ABI"
    assert file.product.level == "L2"
    ds = nc.Dataset(data_file)

    data, labels = [], []
    # Cloud height
    if "ACHA" in file.product.scan:
        data += [ds["HT"][:].data, ds["DQF"][:].data]
        labels += ["height", "q_acha"]
    # Clear sky mask
    elif "ACM" in file.product.scan:
        data += [ds["BCM"][:], ds["ACM"][:], ds["Cloud_Probabilities"][:].data,
                 ds["DQF"][:].data]
        labels += ["cloud_mask", "cloud_class", "cloud_prob", "q_acm"]
    #Cloud top phase
    elif "ACTP" in file.product.scan:
        data += [ds["Phase"][:].data, ds["DQF"][:].data]
        labels += ["cloud_phase", "q_actp"]
    elif "BRF" in file.product.scan:
        labels += ["BRF1", "BRF2", "BRF3", "BRF5", "BRF6"]
        data += [ds[l][:].data for l in labels[-5:]]
        labels.append("q_brf")
        data.append(ds["DQF"][:].data)
        #print(ds["retrieval_solar_zenith_angle"][:])
    # Cloud optical depth
    elif "COD" in file.product.scan:
        data.append(ds["COD"][:])
        data.append(ds["DQF"][:])
        labels += ["cod", "q_cod"]
    # Cloud particle size
    elif "CPS" in file.product.scan:
        data.append(ds["PSD"][:])
        data.append(ds["DQF"][:])
        labels += ["cre", "q_cre"]
    else:
        print("Unrecognized:",file.product.scan)
        exit(0)

    if include_latlon:
        labels += ["lat", "lon"]
        data += list(get_abi_latlon(data_file))
    if include_angles:
        labels += ["sa_ew", "sa_ns"]
        data += list(np.meshgrid(ds["x"][:].data, ds["y"][:].data))

    meta = {
            "path":file.path,
            "time":file.stime.strftime("%s"),
            "desc":gg.describe(file.product),
            }
    return (data, labels, meta)

def default_band_resolution(band):
    """
    Provides the default NADIR resolution of an ABI band.
    """
    return next(res for res,bands in resolutions.items()
                if valid_band(band) in bands)

def valid_band(band):
    """
    Band validator that checks if the parameter is an integer between 1 and
    16 inclusively. Returns the integer band number if valid, or raises an
    error otherwise.

    :@param band: String or integer that must be validated as a band.
    :@return: Integer valid band if it meets the criteria above.
    """
    if not str(band).isnumeric():
        raise ValueError(
                f"Band must be an integer from 1 to 16 (not {band})")
    band = int(band)
    if not 0<band<17:
        raise ValueError(f"Band must be an integer in [0,16]; not {band}")
    return band


def download_l1b(data_dir:Path, satellite:str, scan:str, bands:list,
                 start_time:datetime, end_time:datetime=None, replace=False):
    """
    Download one or more ABI L1b netCDF files for each of a list of bands.
    This method implicitly assumes that all requested bands will have the
    same start time closest to the target time, which is true by default for
    GOES data as far as I know.

    Note that if end_time isn't provided, the inner list will only have one
    member corresponding to the closest file to start_time for that band.

    :@param data_dir: Directory to store downloaded netCDF files
    :@param satellite: GOES satellite number ('16', '17', or '18')
    :@param scan: Satellite scan type: 'C' for CONUS, 'F' for Full Disc,
        and 'M1' or 'M2' for Mesoscale 1 or 2.
    :@param bands: List of ABI bands, [1-16]. The order of this list
        corresponds to the order of the returned netCDF files for each band.
    :@param start_time: Inclusive initial time of the time range to search if
        end_time is defined. Otherwise if end_time is None, start_time stands
        for the target time of returned files.
    :@param end_time: If end_time isn't None, it defines the exclusive upper
        bound of a time range in which to search for files.

    :@return: Nested lists of downloaded GOES netCDF file locations. Each
        first-list entry corresponds to a unique timestep, which contains
        one or more bands' worth of files.
    """
    assert str(satellite) in tuple(map(str, range(16,19)))
    # Accept "RadM" style notation for consistency with product listing
    assert scan in ("C", "F", "M1", "M2")
    assert all(int(b) in range(1,17) for b in bands)
    search_str = f"Rad{scan[0]}"
    goesapi = gg.GetGOES()
    prod = gg.GOES_Product(satellite, "ABI", "L1b", search_str)
    if not end_time is None:
        files = goesapi.search_range(prod, start_time, end_time)
    else:
        files = goesapi.get_closest_to_time(prod, start_time)

    # If mesoscale scan requested, narrow down results to the requested view.
    if scan[0]=="M":
        files = [f for f in files if f.label.split("-")[2][-2:]==scan]
    files.sort(key=lambda f: f.stime)
    utimes = sorted(list(set(f.stime for f in files)))
    paths_by_timestep = []
    for t in utimes:
        band_files = [
                goesapi.download(f, data_dir, replace)
                for f in files
                if any(f"C{int(b):02}" in f.label.split("-")[-1]
                       for b in bands)
                and f.stime == t
                ]
        paths_by_timestep.append(band_files)
    return paths_by_timestep

'''
def get_abi_l1b_latlon(nc_file:Path):
    """
    Uses geometry information
    """
    ds = nc.Dataset(nc_file.as_posix())
    proj = ds["goes_imager_projection"]
    geom = GeosGeom(
        # Nadir longitude
        lon_proj_origin=proj.longitude_of_projection_origin,
        e_w_scan_angles=sa_ew, # Horizontal FGC (m)
        n_s_scan_angles=sa_ns, # Vertical FGC (m)
        satellite_alt=proj.perspective_point_height, # radius (m)
        # Earth spheroid equitorial radius (m)
        r_eq=proj.semi_major_axis,
        # Earth spheroid polar radius (m)
        r_pol=proj.semi_minor_axis,
        sweep=proj.sweep_angle_axis,
        )
    return (geom.lats, geom.lons)
'''

def get_abi_scanangle(nc_file:Path, _ds=None):
    """
    Extracts north/south and east/west scan angles from the provided
    NOAA-style ABI L1b netCDF file, returning them as uniform 2d grids
    with scan angle units in fixed-grid radians

    :@param nc_file: Valid L1b netCDF file with 'x' and 'y' dimensions.

    :@return: 2-tuple like (north-south angles, east-west angles) where each
        member is an (M,N) shaped scalar grid of sensor scan angles in radians.
    """
    ds = _ds if _ds else nc.Dataset(nc_file.as_posix())
    proj = ds["goes_imager_projection"]
    sa_ew,sa_ns = np.meshgrid(ds["x"][:], ds["y"][:])
    return (sa_ns, sa_ew)

def get_abi_latlon(nc_file:Path):
    """
    Uses geometric projection information stored in the provided NOAA-style
    ABI L1b netCDF file to calculate

    :@param nc_file: NOAA-style ABI L1b netCDF file
    :@return: 2-tuple like (lats, lons) where each member is a (M,N) shaped
        array of latitude and longitude floats in degrees for M vertical coords
        and N horizontal coords
    """
    ds = nc.Dataset(nc_file.as_posix())
    sa_ns, sa_ew = get_abi_scanangle(nc_file, _ds=ds)
    proj = ds["goes_imager_projection"]
    geom = GeosGeom(
        # Nadir longitude
        lon_proj_origin=proj.longitude_of_projection_origin,
        e_w_scan_angles=sa_ew, # Horizontal FGC (m)
        n_s_scan_angles=sa_ns, # Vertical FGC (m)
        satellite_alt=proj.perspective_point_height, # radius (m)
        # Earth spheroid equitorial radius (m)
        r_eq=proj.semi_major_axis,
        # Earth spheroid polar radius (m)
        r_pol=proj.semi_minor_axis,
        sweep=proj.sweep_angle_axis,
        )
    return (geom.lats, geom.lons)

def get_abi_l1b_radiance(nc_file:Path, get_mask:bool=False, _ds=None):
    """
    Extract the radiances from a L1b netCDf, optionally including a boolean
    mask for off-disc values.

    :@param nc_file: NOAA-style ABI L1b netCDF file
    :@param get_mask: if True, returns a (M,N) shaped boolean array along
        with the radiances.

    :@return: (M,N) shaped array of scalar radiances. If get_mask is True,
        returns 2-tuple like (radiances, mask) such that 'mask' is a (M,N)
        shaped boolean array which is True for off-limb values.
    """
    ds = _ds if _ds else nc.Dataset(nc_file.as_posix())
    if not get_mask:
        return ds["Rad"][:].data
    data = np.copy(ds["Rad"][:].data)
    mask = np.ma.getmask(ds["Rad"][:])
    # For some reason, nan values occasionally still pass the mask
    return data, mask|np.isnan(data)

def get_abi_l1b_ref(nc_file:Path, get_mask:bool=False):
    """
    Use the provided kappa0 coefficient to calculate lambertian reflectance
    reflectance  =  radiance * kappa0  =  radiance * [(pi * d^2) / E_sun]
    where E_sun is the solar irradiance for the band in W/(m^2 um)

    :@param nc_file: NOAA-style ABI L1b netCDF file
    :@param get_mask: if True, returns a (M,N) shaped boolean array along
        with the reflectances.

    :@return: (M,N) shaped array of scalar reflectance factors. If get_mask is
        True, returns 2-tuple like (radiances, mask) such that 'mask' is a
        (M,N) shaped boolean array which is True for off-limb values.
    """
    ds = nc.Dataset(nc_file.as_posix())
    kappa0 = ds["kappa0"][:].data
    if not kappa0:
        raise ValueError(
                f"Cannot confert to reflectance with {nc_file}; "
                "kappa0 coefficient missing. Is it a thermal band?")
    if not get_mask:
        return get_abi_l1b_radiance(nc_file, _ds=ds) * kappa0
    rad, mask = get_abi_l1b_radiance(nc_file, get_mask=True, _ds=ds)
    return rad*kappa0, mask

def get_abi_l1b_Tb(nc_file:Path, get_mask:bool=False):
    """
    Use the provided planck function coefficients to convert an ndarray of
    scalar radiance values to brightness temperatures.

    :@param nc_file: NOAA-style ABI L1b netCDF file
    :@param get_mask: if True, returns a (M,N) shaped boolean array along
        with the brightness temperatures.

    :@return: (M,N) shaped array of scalar brightness temperatuers. If get_mask
        is True, returns 2-tuple like (radiances, mask) such that 'mask' is a
        (M,N) shaped boolean array which is True for off-limb values.
    """
    ds = nc.Dataset(nc_file.as_posix())
    planck = (ds["planck_fk1"][:].data, ds["planck_fk2"][:].data,
              ds["planck_bc1"][:].data, ds["planck_bc2"][:].data)
    if not all(planck):
        raise ValueError(
                f"Cannot extract brightness temperature from {nc_file}; "
                "planck coefficients missing. Is it a reflectance band?")
    if not get_mask:
        return rad_to_Tb(get_abi_l1b_radiance(nc_file, _ds=ds), *planck)
    rad, mask = get_abi_l1b_radiance(nc_file, get_mask=True, _ds=ds)
    return rad_to_Tb(rad, *planck), mask

def rad_to_Tb(rads:np.array, fk1:np.array, fk2:np.array,
              bc1:np.array, bc2:np.array):
    """
    Use combined-constant coefficients of the planck function to convert
    radiance to brightness temperature.
    """
    return (fk2/np.log(fk1/rads+1) - bc1) / bc2

def is_reflective(band:int):
    """ Returns True if the provided ABI band is reflective """
    # Band 7 (3.9um) doesn't have a kappa0 coefficient, for some reason
    return int(band) < 7
def is_thermal(band:int):
    """ Returns True if the provided ABI band is a thermal band """
    return int(band) >= 7

if __name__=="__main__":
    pass
