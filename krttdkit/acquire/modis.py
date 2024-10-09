"""
"""

import requests
from pprint import pprint as ppt
import json
import netCDF4 as nc
import gc
import numpy as np
import math as m
from pathlib import Path
import pickle as pkl
from datetime import datetime as dt
from datetime import timedelta as td
from pyhdf.SD import SD, SDC
from PIL import Image

from . import laads
from . import modis_rsrs

l2_products = {
        'aod1':'1km Atmospheric Optical Depth Band 1',
        'aod3':'1km Atmospheric Optical Depth Band 3',
        'aod8':'1km Atmospheric Optical Depth Band 8',
        'aod_model':'1km Atmospheric Optical Depth Model',
        'water_vapor':'1km water_vapor',
        'aod_qa':'1km Atmospheric Optical Depth Band QA',
        'aod_cm':'1km Atmospheric Optical Depth Band CM',
        'R1_250':'250m Surface Reflectance Band 1',
        'R2_250':'250m Surface Reflectance Band 2',
        'R1_500':'500m Surface Reflectance Band 1',
        'R2_500':'500m Surface Reflectance Band 2',
        'R3_500':'500m Surface Reflectance Band 3',
        'R4_500':'500m Surface Reflectance Band 4',
        'R5_500':'500m Surface Reflectance Band 5',
        'R6_500':'500m Surface Reflectance Band 6',
        'R7_500':'500m Surface Reflectance Band 7',
        'R1_1000':'1km Surface Reflectance Band 1', # .620-.670um
        'R2_1000':'1km Surface Reflectance Band 2', # .841-.876um
        'R3_1000':'1km Surface Reflectance Band 3', # .459-.479um # Blue
        'R4_1000':'1km Surface Reflectance Band 4', # .535-.565um
        'R5_1000':'1km Surface Reflectance Band 5', # 1.230-1.250um
        'R6_1000':'1km Surface Reflectance Band 6', # 1.628-1.652um # Cloud phase; ice absorbs
        'R7_1000':'1km Surface Reflectance Band 7', # 2.105-2.155um # Particle size; larger less reflective.
        'R8_1000':'1km Surface Reflectance Band 8', # .405-.420um
        'R9_1000':'1km Surface Reflectance Band 9', # .438-.448um
        'R10_1000':'1km Surface Reflectance Band 10', # .483-.448um # Cyan
        'R11_1000':'1km Surface Reflectance Band 11', # .526-.536um
        'R12_1000':'1km Surface Reflectance Band 12', # .546-.556um # Green, less chlorophyll absorption
        'R13_1000':'1km Surface Reflectance Band 13', # .662-.672um # Short Red
        'R14_1000':'1km Surface Reflectance Band 14', # .674-.683um # Mid Red, chlorophyll absorption
        'R15_1000':'1km Surface Reflectance Band 15', # .743-.753um # Long Red
        'R16_1000':'1km Surface Reflectance Band 16', # .862-.877um # Veggie NIR
        'R26_1000':'1km Surface Reflectance Band 26', # 1.36-1.39um # Cirrus band
        'T20_1000':'BAND20', # 3.66-3.84um # SWIR magic band. Fires, LST, ref & emit
        'T31_1000':'BAND31', # 10.78-11.28um # Clean LWIR window
        'T32_1000':'BAND32', # 11.77-12.27um # Dirty-ish LWIR window
        'latitude':'Latitude',
        'longitude':'Longitude',
        'R20_albedo':'BAND20ALBEDO',
        'R_250_qflags':'250m Reflectance Band Quality',
        'R_500_qflags':'500m Reflectance Band Quality',
        'R_1000_qflags':'1km Reflectance Band Quality',
        'R_bands8-15_qflags':'1km b8-15 Reflectance Band Quality',
        'R_band16_qflags':'1km b16 Reflectance Band Quality',
        'R_1000_qa':'1km Reflectance Data State QA',
        'band3_1000_path_radiance':'1km Band 3 Path Radiance'
        }

def parse_modis_time(fpath:Path):
    """
    Use the VIIRS standard file naming scheme (for the MODAPS DAAC, at least)
    to parse the acquisition time of a viirs file.

    Typical files look like: VJ102IMG.A2022203.0000.002.2022203070756.nc
     - Field 1:     Satellite, data, and band type
     - Field 2,3:   Acquisition time like A%Y%j.%H%M
     - Field 4:     Collection (001 for Suomi NPP, 002 for JPSS-1)
     - Field 5:     File creation time like %Y%m%d%H%M
    """
    return dt.strptime("".join(fpath.name.split(".")[1:3]), "A%Y%j%H%M")

def sat_to_l2key(satellite):
    """ Convert a human-readable satellite argument to a L2 product key. """
    valid = {"terra", "aqua"}
    if satellite.lower() not in valid:
        raise ValueError(f"MODIS L2 satellite must be one of {valid}")
    return ["MYD09", "MOD09"][satellite == "terra"]

def query_modis_l1b(product_key:str, start_time:dt, end_time:dt,
                   latlon:tuple=None, archive=None, day_only:bool=False,
                   debug:bool=False):
    """
    Query the LAADS DAAC for MODIS L2 MOD021KM (Terra) and MYD021KM (Aqua)
    calibrated surface reflectance and select brightness temperatures.
    Instead of 1KM (for 1km resolution), products may use HKM or QKM
    substrings for half-kilometer and quarter-kilometer resolution bands,
    respectively.

    :@param product_key: One of the listed VIIRS l1b product keys
    :@param start_time: Inclusive start time of the desired range. Only files
            that were acquired at or after the provided time may be returned.
    :@param end_time: Inclusive end time of the desired range. Only files
            that were acquired at or before the provided time may be returned.
    :@param archive: Some products have multiple archives, which seem to be
            identical. If archive is provided and is a validarchive set,
            uses the provided value. Otherwise defualts to defaultArchiveSet
            as specified in the product API response.

    :@return: A list of dictionaries containing the aquisition time of the
            granule, the data granule download link, and optionally the
            download link of the geolocation file for the granule.
    """
    valid = {"MOD021KM", "MYD021KM", "MOD02QKM", "MYD02QKM",
             "MOD02HKM", "MYD02HKM", "MOD03", "MYD03"}
    if product_key not in valid:
        raise ValueError(f"Product key must be one of: {valid}")

    products = laads.query_product(product_key, start_time, end_time, latlon,
                                   archive=archive, debug=debug)
    for i in range(len(products)):
        products[i].update({"atime":parse_modis_time(Path(
                    products[i]['downloadsLink']))})
    products = [ p for p in products
            if p["illuminations"] == "D" or not day_only ]

    return list(sorted(products, key=lambda p:p["atime"]))

def query_modis_l2(product_key:str, start_time:dt, end_time:dt,
                   latlon:tuple=None, archive=None, day_only:bool=False,
                   debug:bool=False):
    """
    Query the LAADS DAAC for MODIS L2 MOD09 (Terra) and MYD09 (Aqua)
    calibrated surface reflectance and select brightness temperatures

    :@param product_key: One of the listed VIIRS l1b product keys
    :@param start_time: Inclusive start time of the desired range. Only files
            that were acquired at or after the provided time may be returned.
    :@param end_time: Inclusive end time of the desired range. Only files
            that were acquired at or before the provided time may be returned.
    :@param archive: Some products have multiple archives, which seem to be
            identical. If archive is provided and is a validarchive set,
            uses the provided value. Otherwise defualts to defaultArchiveSet
            as specified in the product API response.

    :@return: A list of dictionaries containing the aquisition time of the
            granule, the data granule download link, and optionally the
            download link of the geolocation file for the granule.
    """
    valid = {"MOD09", "MYD09"}
    if product_key not in valid:
        raise ValueError(f"Product key must be one of: {valid}")

    products = laads.query_product(product_key, start_time, end_time, latlon,
                                   archive=archive, debug=debug)
    for i in range(len(products)):
        products[i].update({"atime":parse_modis_time(Path(
                    products[i]['downloadsLink']))})
    products = [ p for p in products
            if p["illuminations"] == "D" or not day_only ]

    return list(sorted(products, key=lambda p:p["atime"]))

def validate_l2_bands(bands:tuple):
    """
    Raises an error if any string in provided tuple isn't a l2 key per the
    valid_bands dictionary.
    """
    invalid = [ b for b in bands if b not in l2_products.keys() ]
    if len(invalid):
        raise ValueError(f"Bands {invalid} are an invalid. Keys " + \
                "Must be one of {l2_products.keys()}")

def band_to_wl(band:int):
    """
    Returns the central wavelength of the integer band in um by finding
    the weighted mean wavelength of the spectral response.

    Spectral response functions provided by:
    https://nwp-saf.eumetsat.int/downloads/rtcoef_rttov13/ir_srf/rtcoef_eos_1_modis_srf/
    """
    rsr_dict = modis_rsrs.MODIS_RSRs[band]
    wl = rsr_dict["wavelength"]
    rsr = rsr_dict["rsr"]
    mid_idx = np.argmin(np.abs(np.cumsum(np.array(rsr))-round(sum(rsr)/2)))
    return wl[mid_idx]


def get_modis_data(datafile:Path, bands:tuple,
                   l1b_convert_reflectance:bool=True,
                   l1b_convert_tb:bool=True, debug=False):
    """
    Opens a Terra or Aqua L1b calibrated radiances or L2 atmospherically-
    corrected reflectance/emission granule file, and parses the
    requested bands based on keys defined in dictionaries above.

    The values returned by this method are contingent on the file type
    provided to datafile.

    For l1b files:
    sunsat and geolocation data is bilinear-interpolated from a 5x5 subsampled
    grid. By default reflectance bands are converted to BRDF and thermal
    bands are left as radiances.

    :@param datafile: Path to level 1b or level 2 hdf4 file, probably from the
            laads daac.
    :@param bands: Band key defined in l2_products dictionary if datafile is
            a l2 path, or a valid MODIS band number.
    :return: (data, info, geo) 3-tuple. data is a list of ndarrays
            corresponding to each requested band, info is a list of data
            attribute dictionaries for the respective bands, and geo is a
            2-tuple (latitude, longitude) of the 1km data grid.
    """
    #validate_l2_bands(bands)
    mod_sd = SD(datafile.as_posix(), SDC.READ)
    data = []
    info = []
    # Awful way of determining if this is a l2 file, but this method is bound
    # to the LAADS DAAC key naming scheme regardless.
    is_l2 = any([ "Surface Reflectance" in k
                 for k in mod_sd.datasets().keys()])
    if is_l2:
        """ L2 file parsing using l2_products band naming """
        for b in bands:
            name = l2_products[b]
            tmp_sds = mod_sd.select(l2_products[b])
            tmp_data = tmp_sds.get()
            tmp_info = tmp_sds.attributes()
            ndr = tmp_info["Nadir Data Resolution"]
            tmp_info = {
                    k:tmp_info[k] for k in("add_offset", "scale_factor",
                                           "units", "valid_range")
                    if k in tmp_info.keys()}
            tmp_info["name"] = name
            tmp_info["key"] = b
            tmp_info["nadir_resolution"] = ndr
            info.append(tmp_info)
            # Scale the data with the provided values.
            data.append((tmp_data+tmp_info["add_offset"])/tmp_info["scale_factor"])

        geo = (mod_sd.select("Latitude").get(), mod_sd.select("Longitude").get())
        return data, info, geo

    """ 1km L1b file parsing """
    # Get subsampled geolocation and sun/satellite geometry arrays
    geo = [mod_sd.select("Latitude").get(),
           mod_sd.select("Longitude").get(),
           mod_sd.select("Height").get() ]
    sunsat = [mod_sd.select("SolarZenith").get()*.01,
              mod_sd.select("SolarAzimuth").get()*.01,
              mod_sd.select("SensorZenith").get()*.01,
              mod_sd.select("SensorAzimuth").get()*.01]

    # Ridiculous 5x5 upscaling of geolocation data
    _, h, w = mod_sd.select("EV_1KM_RefSB").get().shape
    geo = [ np.array(Image.fromarray(X).resize(
                size=(w,h), resample=Image.BILINEAR))
            for X in geo]
    sunsat = [ np.array(Image.fromarray(X).resize(
                size=(w,h), resample=Image.BILINEAR))
            for X in sunsat]

    if debug and l1b_convert_reflectance:
        print("Converting to reflectance")
    if debug and l1b_convert_tb:
        print("Converting to brightness temperature")

    data = []
    info = []
    for b in bands:
        if b in mod_sd.select("Band_250M").get():
            idx = list(mod_sd.select("Band_250M")).index(b)
            tmp_sd = mod_sd.select("EV_250_Aggr1km_RefSB")
            tmp_attrs = tmp_sd.attributes()
            data.append(tmp_sd.get()[idx,:,:])
        elif b in mod_sd.select("Band_500M").get():
            idx = list(mod_sd.select("Band_500M")).index(b)
            tmp_sd = mod_sd.select("EV_500_Aggr1km_RefSB")
            tmp_attrs = tmp_sd.attributes()
            data.append(mod_sd.select("EV_500_Aggr1km_RefSB").get()[idx,:,:])
        elif b in mod_sd.select("Band_1KM_Emissive").get():
            idx = list(mod_sd.select("Band_1KM_Emissive")).index(b)
            tmp_sd = mod_sd.select("EV_1KM_Emissive")
            tmp_attrs = tmp_sd.attributes()
            data.append(mod_sd.select("EV_1KM_Emissive").get()[idx,:,:])
        elif b in mod_sd.select("Band_1KM_RefSB").get():
            idx = list(mod_sd.select("Band_1KM_RefSB")).index(b)
            tmp_sd = mod_sd.select("EV_1KM_RefSB")
            tmp_attrs = tmp_sd.attributes()
            data.append(mod_sd.select("EV_1KM_RefSB").get()[idx,:,:])
        elif b == 26:
            idx = 0
            tmp_sd = mod_sd.select("EV_Band26")
            tmp_attrs = tmp_sd.attributes()
            data.append(mod_sd.select("EV_Band26").get()[:,:])
        else:
            raise ValueError(f"Band {b} not found.")

        # Construct a dictionary of attribute info for this band.
        tmp_info = {
                "band":b,
                "units":tmp_attrs["radiance_units"],
                "fill":tmp_attrs["_FillValue"],
                "long_name":tmp_attrs["long_name"],
                "valid_range": tmp_attrs["valid_range"],
                "rad_scale":tmp_attrs["radiance_scales"][idx],
                "rad_offset":tmp_attrs["radiance_offsets"][idx],
                "is_reflective": "reflectance_units" in tmp_attrs.keys(),
                "ctr_wl": band_to_wl(b),
                "product":datafile.stem.split(".")[0],
                }
        # Use reflectance scale if converting to reflectance
        if tmp_info["is_reflective"] and l1b_convert_reflectance:
            tmp_info["ref_scale"] = tmp_attrs["reflectance_scales"][idx]
            tmp_info["ref_offset"] = tmp_attrs["reflectance_offsets"][idx]
            #if b == 1:
            #    print(tmp_info)
            tmp_info["units"] = "Reflectance (BRDF)"
            # Bidirectional reflectance factor (Petty 5.15)
            # From l1b PUG p43, reflectance = rho*cos(sza)
            data[-1] = (data[-1]-tmp_info["ref_offset"]) \
                    * tmp_info["ref_scale"] / np.cos(np.deg2rad(sunsat[0]))
        # Otherwise convert to radiance
        else:
            data[-1] = (data[-1]-tmp_info["rad_offset"])*tmp_info["rad_scale"]

        # If emissive band and converting, use radiances and weighted rsr mean
        # to estimate brightness temperature. Could do a discrete weighted sum
        # along wavelength in the future to improve Tb estimate.
        if not tmp_info["is_reflective"] and l1b_convert_tb:
            # Rough brightness temp with inverse planck function
            ctr_wl = tmp_info["ctr_wl"]
            c1 = 1.191042e8 # W / (m^2 sr um^-4)
            c2 = 1.4387752e4 # K um
            # Get brightness temp with planck's function at the center wavelength
            data[-1] = c2/(ctr_wl * np.log(c1/(ctr_wl**5 * data[-1]) + 1))
            tmp_info["units"] = "Kelvin"

        info.append(tmp_info)
    return data, info, geo, sunsat

def download_modis_granule(
        product_key, dest_dir:Path, token_file:Path, bands:tuple,
        target_time:dt=None, latlon:tuple=None, keep_files:bool=True,
        day_only:bool=False, replace:bool=False, include_geo:bool=False,
        include_sunsat:bool=False, mask:bool=False, archive=None, debug=False):
    """
    Downloads the granule belonging to the requested product closest to the
    target_time and returns its full array for each band along with
    geolocation information.

    Note that the reflectance values for M01-M11 and I01-I03 are the true
    reflectance divided by SZA. If you need the actual observed reflectance,
    invert this using the sunsat sza array.

    M12-M16 and I05-I05 are stored as radiances, but have brightness temp
    lookup tables.

     - Required Parameters -
    :@param product_key: LAADS DAAC product key for a VIIRS L1b dataset;
            see query_viirs_l1b() documentation.
    :@param dest_dir: Destination directory to download netCDF files into.
    :@param token_file: ASCII text file containing only a LAADS DAAC token,
            which can be generated using the link above.
    :@param bands: Tuple of integer band numbers to limit the array returned
            by this method to, which is necessary since a contiguous array of
            multiple granules can be very, very big.

     - Optional Parameters -
    :@param target_time: Approximate acquisition time of file to download. If
            None, defaults to current UTC time.
    :@param latlon: Add a constraint that this latitude and longitude must be
            contained within the granule. Accepts a tuple of latitude and
            longitude like (34.2, -86.4).
    :@param keep_files: If False, the downloaded netCDF files are deleted,
            so only the observation and geolocation data arrays are maintained.
    :@param day_only: If True, adds a constraint that the temporally-nearest
            granule downloaded by this method must have daytime illumination.
    :@param replace: If True, overwrites identically-named files in dest_dir.
    :@param include_geo: If True, provides geopositioning info including
            latitude, longitude, and terrain height at every pixel in the
            returned tuple at index 1.
    :@param include_sunsat: If True, provides sun/satellite/pixel geometry info
            including solar and sensor azimuth and zenith angles at every pixel
            in the returned tuple at index 2.
    :@param mask: If True, applies a mask any time each band's quality flags
            are nonzero.

    :@return: 4-Tuple like (data,  info, geo, sunsat).
            geo and sunsat will be None unless their respective arguments
            include_geo or include_sunsat are set to True.
    """
    if product_key not in valid_products:
        raise ValueError(f"Product key must be one of: {valid_products}")
    # Search for a range of about 4 days to make sure any specified latlon
    # value shows up in a swath during the daytime.
    if target_time is None:
        target_time = dt.utcnow()
    granules = []
    breadth = 1
    while not len(granules):
        granules = query_viirs_l1b(
                product_key, target_time-td(days=breadth),
                target_time+td(days=breadth), latlon=latlon,
                archive=archive, debug=debug)
        try:
            granules = [
                    [ g[k] for k in
                        ("atime", "downloadsLink", "geoLink") ]
                    for g in granules
                    if g["illuminations"] == "D" or not day_only ]
            _, closest_granule = min(enumerate(granules),
                                     key=lambda g: abs(g[1][0]-target_time))
            _, data_link, geo_link = closest_granule
        except ValueError:
            if debug:
                print(f"No passes found within +/- {breadth} days; " + \
                        f"expanding search to +/- {breadth+1} days.")
            breadth += 1


    # Download the data and geometry files
    data_file = laads.download(data_link, dest_dir, token_file=token_file,
                               replace=replace, debug=debug)
    data, info = get_viirs_data(l1b_file=data_file, bands=bands, mask=mask,
                                debug=debug)
    sunsat = None
    geo = None
    if include_geo or include_sunsat:
        geo_file = laads.download(geo_link, dest_dir, token_file=token_file,
                                  replace=replace, debug=debug)
        if include_geo:
            geo = get_viirs_geoloc(geoloc_file=geo_file, debug=debug)
            info["lat_range"] = (np.amin(geo[0]), np.amax(geo[0]))
            info["lon_range"] = (np.amin(geo[1]), np.amax(geo[1]))
        if include_sunsat:
            sunsat = get_viirs_sunsat(geoloc_file=geo_file, debug=debug)
    if not keep_files:
        geo_file.unlink()
        data_file.unlink()
    return (data, info, geo, sunsat)

