
import requests
from pprint import pprint as ppt
import json
import netCDF4 as nc
import gc
import numpy as np
from pathlib import Path
import pickle as pkl
from datetime import datetime as dt
from datetime import timedelta as td

from . import laads

valid_products = frozenset(("VJ102MOD", "VJ102IMG", "VJ102DNB",
                           "VNP02MOD", "VNP02IMG", "VNP02DNB"))

def parse_viirs_time(fpath:Path):
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

def query_viirs_l1b(product_key:str, start_time:dt, end_time:dt, add_geo=True,
                    latlon:tuple=None, archive=None, debug:bool=False):
    """
    Query the LAADS DAAC for VIIRS l1b products for a specified product type,
    which must be one of the key strings in the table below.

              |  Moderate  Imagery   Day-night
    ____________________________________________________
    JPSS-1    |  VJ102MOD  VJ102IMG  VJ102DNB"
    Suomi NPP |  VNP02MOD  VNP02IMG  VNP02DNB"

    :@param product_key: One of the listed VIIRS l1b product keys
    :@param start_time: Inclusive start time of the desired range. Only files
            that were acquired at or after the provided time may be returned.
    :@param end_time: Inclusive end time of the desired range. Only files
            that were acquired at or before the provided time may be returned.
    :@param add_geo: If True, queries LAADS for the geolocation product and
            includes the download link of the result
    :@param archive: Some products have multiple archives, which seem to be
            identical. If archive is provided and is a validarchive set,
            uses the provided value. Otherwise defualts to defaultArchiveSet
            as specified in the product API response.

    :@return: A list of dictionaries containing the aquisition time of the
            granule, the data granule download link, and optionally the
            download link of the geolocation file for the granule.
    """
    if product_key not in valid_products:
        raise ValueError(f"Product key must be one of: {valid_products}")

    products = laads.query_product(product_key, start_time, end_time, latlon,
                                   archive=archive, debug=debug)
    geos = {
        parse_viirs_time(Path(g['downloadsLink'])):g['downloadsLink']
        for g in laads.query_product(
            product_key.replace("02", "03"), start_time, end_time, latlon,
            archive=archive, debug=debug)
        } if add_geo else {}

    for i in range(len(products)):
        products[i].update({"atime":parse_viirs_time(Path(
                    products[i]['downloadsLink']))})
        if add_geo:
            # Geolocation files should always match acquisition time with data.
            products[i].update({'geoLink':geos.get(products[i]['atime'])})
    return list(sorted(products, key=lambda p:p["atime"]))

def get_viirs_sunsat(geoloc_file:Path, debug=False):
    """
    Parse latitude, longitude, and terrain height variables from a viirs
    netCDF, which is expected to follow the LAADS DAAC geolocation format.

    :@param geoloc_file: LAADS-downloaded VIIRS L1bgeolocation file
    :@return: Tuple of float ndarrays for sun/pixel/satellite geometry like:
            (solar_zenith, solar_azimuth, sensor_zenith, sensor_azimuth)
    """
    if debug: print(f"Loading sun/sat/px: {geoloc_file.as_posix()}")
    glnc = nc.Dataset(geoloc_file, 'r').groups["geolocation_data"]
    solar_zen = glnc["solar_zenith"][::-1,::-1]
    solar_azi = glnc["solar_azimuth"][::-1,::-1]
    sensor_zen = glnc["sensor_zenith"][::-1,::-1]
    sensor_azi = glnc["sensor_azimuth"][::-1,::-1]
    return (solar_zen, solar_azi, sensor_zen, sensor_azi)

def get_viirs_geoloc(geoloc_file:Path, debug=False):
    """
    Parse latitude, longitude, and terrain height variables from a viirs
    netCDF, which is expected to follow the LAADS DAAC geolocation format.

    :@param geoloc_file: LAADS-downloaded VIIRS L1bgeolocation file
    :@return: Tuple of float ndarrays for surface geolocation like:
            (latitude, longitude, height)
    """
    if debug: print(f"Loading geolocation: {geoloc_file.as_posix()}")
    glnc = nc.Dataset(geoloc_file, 'r').groups["geolocation_data"]
    height = glnc["height"][::-1,::-1]
    latitude = glnc["latitude"][::-1,::-1]
    longitude = glnc["longitude"][::-1,::-1]
    return (latitude, longitude, height)

def get_viirs_data(l1b_file:Path, bands:tuple, debug=False, mask=False):
    """
    :@param l1b_file: Path to netCDF of VIIRS L1b observation data
    :@param bands: tuple of integers corresponding to bands to retrieve.
            Bands are numbered according to their type (ie IMG, DNB, MOD).
    :@param mask: If True, applies a mask any time each band's quality flags
            are nonzero.

    :@return: 2-tuple like (data, info) where data is a tuple containing a
            full-size numpyu array for each requested band, and info is
            a dictionary with observation time information.
    """
    # Hacky way of getting the 1-character VIIRS sensor codes
    sensor = l1b_file.name.split(".")[0][-3]
    keys = [ f"{sensor}{band:>02}" for band in bands ]
    if debug:
        print(f"Loading bands: {keys}")
        print(f"\tfrom: {l1b_file.as_posix()}")
    gran = nc.Dataset(l1b_file, 'r')
    observation = gran.groups["observation_data"]
    scan = gran.groups["scan_line_attributes"]
    info = {
            "obs_start":dt.strptime(gran.StartTime.split(".")[0],
                                    "%Y-%m-%d %X"),
            "obs_end":dt.strptime(gran.EndTime.split(".")[0],
                                  "%Y-%m-%d %X"),
            "file_time":parse_viirs_time(l1b_file),
            "day_night":str(gran.DayNightFlag),
            "bands":keys,
            "qflags":{ k:observation.variables[f"{k}_quality_flags"][::-1,::-1]
                      for k in keys }
            }

    # Watts/meter^2/steradian/micrometer
    all_bands = []
    for k in keys:
        data = observation.variables[k][::-1,::-1]
        if mask:
            data = np.ma.masked_where(info["qflags"][k]!=0, data)
        all_bands.append(data)

    return (all_bands, info)

def download_viirs_granule(
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

def download_viirs_swath(
        product_key:str, start_time:dt, end_time:dt, dest_dir:Path,
        token_file:Path, bands:tuple, keep_files:bool=True, replace:bool=False,
        archive=None, debug:bool=False):
    """
    Download a continuous swath of geolocated VIIRS data within a provided UTC
    time range, and append all data for each provided band along the
    along-track axis.

    Generate a DAAC token here:
    https://ladsweb.modaps.eosdis.nasa.gov/profiles/#generate-token-modal

    :@param product_key: LAADS DAAC product key for a VIIRS L1b dataset;
            see query_viirs_l1b() documentation.
    :@param start_time: Acquisition start time for the swath.
    :@param end_time: Acquisition start time for the swath.
    :@param dest_dir: Destination directory to download netCDF files into.
    :@param token_file: ASCII text file containing only a LAADS DAAC token,
            which can be generated using the link above.
    :@param bands: Tuple of bands to limit the array returned by this method
            to, which is necessary since a contiguous array of multiple
            granules can be very, very big.
    :@param keep_files: If False, the downloaded netCDF files are deleted,
            so only the observation and geolocation data arrays are maintained.
    :@param replace: If True, overwrites identically-named files in dest_dir.
    """
    if product_key not in valid_products:
        raise ValueError(f"Product key must be one of: {valid_products}")
    viirs_urls = query_viirs_l1b(product_key=product_key,
            start_time=start_time, end_time=end_time, debug=debug)
    arrays = []
    geos = []
    for v in viirs_urls:
        arrays.append(get_viirs_data(laads.download(
            v["downloadsLink"], dest_dir, token_file=token_file,
            replace=replace, debug=debug), bands=bands))
        geos.append(get_viirs_geoloc(laads.download(
            v["geoLink"], dest_dir, token_file=token_file,
            replace=replace, debug=debug)))

    data = np.concatenate(arrays, axis=0)
    geo = np.concatenate(geos, axis=0)

def generate_viirs_pkl(
        product_key:str, label:str, dest_dir:Path, token_file:Path,
        bands:tuple, target_time:dt=None, day_only=True, replace=True,
        keep_nc_files=False, latlon=None, include_sunsat=True,
        include_geo=True, mask=False, debug=True):
    """
    Wrapper method for searching, downloading, subsetting, and pickling a viirs
    L1b file. See laads.viirs.download_viirs_granule for more documentation.

    - Unique param -
    :@param label: Label field of the generated pkl, which should be unique and
            descriptive enough to specify the band composition and target
            location (if applicable) of the requested data. For example, a
            string like 'j01_truecolor_hsv'.

    :@return: Path of the successfully-generated pkl file.
    """
    # Get the granule closest to the desired parameters
    gran = download_viirs_granule(
            product_key=product_key, dest_dir=dest_dir, token_file=token_file,
            bands=bands, target_time=target_time, day_only=day_only,
            keep_files=keep_nc_files, replace=replace, latlon=latlon,
            include_sunsat=include_sunsat, include_geo=include_geo,
            debug=debug)
    data, info, geo, sunsat = gran
    aqtime = info["file_time"]
    pkl_path = dest_dir.joinpath(Path(f"{label}_" + \
            f"{aqtime.strftime('%Y%m%d_%H%M')}.pkl"))
    if pkl_path.exists():
        if debug and replace:
            print(f"Overwriting pickle at {pkl_path}")
    with pkl_path.open("wb") as pklfp:
        pkl.dump(gran, pklfp)
    # Clear the garbage collector since the arrays may be re-loaded
    # from the pkl shortly, which isn't an optimal practice, but more
    # generalizable when sequentially storing and loading array pkls.
    del gran
    gc.collect()
    return pkl_path

def get_viirs_args():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--hour", dest="hour", type=str,
                        help="UTC hour of observation. If no day is " + \
                                "provided, defaults to today.",
                        default=None)
    parser.add_argument("-M", "--minute", dest="minute", type=str,
                        help="Minute of observation. If no hour is "+ \
                                "provided, this value is ignored.",
                        default=None)
    parser.add_argument("-D", "--day", dest="day", type=str,
                        help="Day in YYYYMMDD format",
                        default=None)
    parser.add_argument("-r", "--recipe", dest="recipe", type=str,
                        help="Imagery recipe to use; defaults to truecolor",
                        default="truecolor")
    parser.add_argument("--center", dest="center", type=str,
                        help="lat/lon center, formatted '\d+.\d+,\d+.\d+",
                        default=None)
    parser.add_argument("--aspect", dest="aspect", type=str,
                        help="Grid aspect ratio, formatted '\d+.\d+,\d+.\d+",
                        default=None)
    parser.add_argument("--sat", dest="sat", type=str,
                        help="Satellite to query data from",
                        default="noaa-goes16")
    raw_args = parser.parse_args()

    try:
        assert all([ arg==None for arg in
                    (raw_args.center, raw_args.aspect, raw_args.recipe)])
    except:
        raise ValueError("Grid selection or location arguments " + \
                "aren't supported yet.")

    """ Parse any time arguments """
    if not raw_args.hour is None:
        hour = int(raw_args.hour)%24
        # Only regard the minutes argument if an hour is provided
        target_tod = td( hours=hour,
                minutes=0 if raw_args.minute is None \
                        else int(raw_args.minute)%60)
        # If no day is provided, default to the last occurance of the
        # provided time.
        if raw_args.day is None:
            target_time = (dt.utcnow()-target_tod).replace(
                    hour=0, minute=0, second=0, microsecond=0)+target_tod
        # If a day and
        else:
            try:
                target_day = dt.strptime(raw_args.day, "%Y%m%d")
                target_time = target_day+target_tod
            except:
                raise ValueError("Target day must be in YYYYmmdd format.")
    # Only accept a day argument if an hour is also provided
    # If no day argument or hour argument is provided, default to now.
    else:
        if raw_args.day is None:
            target_time = dt.utcnow()
        else:
            raise ValueError("You cannot specify a day without " + \
                    "also specifying an hour.")

    """ Parse any grid arguments """
    grid_center = None
    grid_aspect = None
    if raw_args.center and raw_args.aspect:
        grid_center = tuple(map(float, raw_args.center.split(",")))
        grid_aspect = tuple(map(float, raw_args.aspect.split(",")))
    elif raw_args.center or raw_args.aspect:
        raise ValueError("You must provide both a center and an aspect ratio")

    return raw_args.recipe, target_time, grid_center, grid_aspect

if __name__=="__main__":
    pass
