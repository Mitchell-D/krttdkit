from boto.s3.connection import S3Connection
from datetime import datetime as dt
from datetime import timedelta as td
from pprint import pprint as ppt
from pathlib import Path
import re
from .abi_spec import get_products, findall, ABIProduct

buckets = {"noaa-goes16"}

def get_bucket_paths(bucket, product:ABIProduct, start:dt, end:dt,
                     band:int=None):
    """
    Get a list of files of the provided product inclusively between the given
    start and end times.

    :param product: ABIProduct object depicting the desired product
    :param start: start time for files in the requested bucket
    :param end: end time for files in the requested bucket
    """
    # Construct an AWS bucket prefix
    prefix = lambda key, time: "/".join((key, time.strftime("%Y/%j/%H/")))

    # Calculate time variables
    increment = td(hours=1) # Bucket directories are in hour increments.
    start = start.replace(second=0, microsecond=0)
    end = end.replace(second=0, microsecond=0)
    hours = int((end.replace(minute=0)-start.replace(minute=0)).seconds/3600)+1

    # Get bucket prefixes that fall on the hour time scale
    prefixes = [ prefix(product.aws_string, start+i*increment)
                for i in range(hours) ]

    # Get a list of AWS object keys for the prefixed buckets
    keys = []
    connection = S3Connection(anon=True).get_bucket(bucket)
    for p in prefixes:
        keys+=list(connection.list(prefix=p))

    # Find keys with stimes that fall within the given range
    results = []
    for k in keys:
        nc_str = k.name.split("/")[-1]
        #print(re.search("(_s\d{14}_)", nc_str[:-2]).group(0))
        time = dt.strptime(
                re.search("(_s\d{13})", nc_str[:-2]).group(0),
                "_s%Y%j%H%M%S")
        if start <= time <= end:
            results.append((time, k))

    if not band is None:
        results = list(filter(lambda r:key_is_of_band(r,band), results))
    # Return list of (time, boto key) tuples
    return list(sorted(results, key=lambda r:r[0]))

def download_from_key(key, directory:Path):
    """
    Download a file at the provided boto key into the provided directory.
    """
    f = directory.joinpath(Path(key.name.split("/")[-1]))
    print(f"Downloading {key.name}\n\tto {f}")
    with open(f.as_posix(), "wb") as nc_fp:
        key.get_contents_to_file(nc_fp)
    return f

def key_is_of_band(key, band:int):
    """
    Returns True if the boto Key name matches the provided band.
    Assumes the band is the last 2 characters of the second underscore-
    separated field in the key's base name.
    """
    return int(key[1].name.split("/")[-1].split("_")[1][-2:])==band

if __name__=="__main__":
    abi_products = get_products()
    rgb_type = "truecolor"
    grid_center = (34.754, -86.768) # (lat, lon)
    grid_aspect = (5,8) # (lat, lon)
    bucket = "noaa-goes16"
    instrument = "ABI"
    domain = "CONUS"
    level = "L1b"

    lookback = td(minutes=45)
    end = dt.now()
    start = end-lookback

    products = findall(
        mapping={"level":level,"instrument":instrument,"domain":domain})
    print(products)
    '''
    times, keys = tuple(zip(*get_bucket_paths(
        bucket, products[0], start, end)))
    ppt(keys)
    '''
