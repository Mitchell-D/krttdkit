"""
Module containing methods for interfacing with the MODAPS LAADS DAAC API,
which includes many historical and NRT products at various processing levels.

Since file naming conventions vary, downloaders for specific instruments or
products are in separate modules to maintain generality.

EXAMLE USAGE:

To get a dictionary of all product information and a descriptive printout:
get_all_products(print_info=True)

To get more detailed info on example product VJ102IMG:
get_product_info("VJ102IMG", print_info=True)

To search for available granules of a given product type
get_product_info("VJ102IMG", print_info=True)

To search for available granules of a given product type within a provided
time range and including a geographic location:
query_product(
        product_key="VJ102MOD",
        start_time=dt(year=2022, month=7, day=22, hour=16, minute=30),
        end_time=dt(year=2022, month=7, day=22, hour=17, minute=15),
        latlon=(34, -86),
        debug=True
        )
"""
import itertools
import requests
import json
from pathlib import Path
from datetime import datetime as dt
from datetime import timedelta as td
from pprint import pprint as ppt
import shlex
from subprocess import Popen, PIPE

laads_root = "https://ladsweb.modaps.eosdis.nasa.gov"
api_root = laads_root+"/api/v2"

def get_all_products(print_info:bool=False, debug=False):
    """
    Query the LAADS DAAC API for information about every product.
    API: https://ladsweb.modaps.eosdis.nasa.gov/api/v2/measurements/products

    :@param print_info: if True, pretty-prints the key, title, and info page
        link for each product to stdout.
    """
    # API info to keep without modifying
    keep_keys = {"title", "temporalCoverage", "temporalResolution", "sensor",
                "processingLevel", "spatialResolution"}

    # Query the measurements API for JSON info on all products.
    if debug is True: print("Querying LAADS API...")
    url = api_root + "/measurements/products"
    products = json.loads(requests.get(url).text)

    # Prune the dictionary to useful information and format new fields
    new_products = {}
    for p in products.keys():
        pnew = { k:products[p][k] for k in keep_keys }
        pnew["infoLink"] = laads_root+products[p]["descriptionLink"]
        pnew["collectionLinks"] = [
                "{r}/content/details/allData/{c}/{n}".format(
                    r=api_root, c=c, n=products[p]["name"])
                for c in products[p]["archiveSet"].keys() ]
        new_products.update(pnew)
        if print_info:
            print(f"\033[1m{p}:\033[0m")
            print(f"    \033[92m{pnew['title']}\033[0m")
            print(f"    \033[96m{pnew['infoLink']}\033[0m")
    return products

def get_product_info(product_key:str, print_info:bool=False, debug=False):
    """
    Query the LAADS DAAC API for a specific product key, and return a dict
    of useful information about that product.

    https://ladsweb.modaps.eosdis.nasa.gov/api/v2/measurements/products

    :@param product_key: string key of the desired product. For valid options,
            see get_all_products(), or the LAADS DAAC link above.
    :@param print_info: Prints the dataset IDs and a link to the API download
            directory tree if True.
    """
    # Query the API with the provided product key
    if debug is True: print("Querying LAADS API...")
    resp = requests.get(f"{api_root}/measurements/products/{product_key}")
    if resp.status_code == 400:
        raise ValueError(f"Invalid product key: {product_key}\n" +
                         "See options with get_all_products(print_info=True).")

    # Return the product info as a dictionary, printing links to the data
    # download if requested.
    product_info = json.loads(resp.text)[product_key]
    product_info["archives"] = {
            c:f"{api_root}/content/details/allData/{c}/{product_key}"
            for c in list(product_info["archiveSet"].keys()) }
    product_info["descriptionLink"] = laads_root + \
            product_info["descriptionLink"]
    del product_info["archiveSet"]
    #del product_info["collection"]
    if print_info:
        print(f"\033[1m{product_key}\033[0m")
        for k in product_info["archives"].keys():
            print(f"\033[92m    {k} \033[96m\033[4m" + \
                    f"{product_info['archives'][k]}\033[0m")
    return product_info

def query_product_day(product_key:str, utc_day:dt=None, archive:str=None,
                      debug=False, _pinfo=None):
    """
    The LAADS DAAC API is hierarchically organized per UTC day; query the site
    for the requested product, and return a list of URLs corresponding to
    all data files available in the requested UTC day.

    This looks for the data grid files (ie VJ102MOD), but geolocation files
    (ie VJ103MOD) should correspond to the same observation times.

    https://ladsweb.modaps.eosdis.nasa.gov/api/v2/measurements/products

    :@param product_key: string key of the desired product. For valid options,
            see get_all_products(), or the LAADS DAAC link above.
    :@param utc_day: Observation day to query. defaults to today.
    :@param archive: Some products have multiple archives, which seem to be
            identical. If archive is provided and is a valid archive set,
            uses the provided value. Othewise defaults to the defaultArchiveSet
            from the product API response.
    :@param _pinfo: Hidden workaround for providing the product dictionary
            as an argument rather than pinging LAADS each time this method
            is called. Useful if many days are being queried at the same time.
    """
    # Query the product info dict
    pinfo = _pinfo if not _pinfo is None else \
            get_product_info(product_key, debug=debug)
    # Determine which archive to use
    if not archive is None and archive not in pinfo["archives"].keys():
        raise ValueError(f"Provided archive not a valid option for " + \
                "product {product_key};\nvalid: {pinfo['archives'].keys()}")
    archive = archive if not archive is None \
            else pinfo["defaultArchiveSet"]
    # Construct the data url and request the JSON of file paths
    #url = pinfo['archives'][archive] + utc_day.strftime("/%Y/%j")
    url = pinfo['archives'][str(archive)] + utc_day.strftime("/%Y/%j")
    return [ f["downloadsLink"] for f in
            json.loads(requests.get(url).text)["content"] ]

def query_product(product_key:str, start_time:dt, end_time:dt,
                  latlon:tuple=None, archive=None, debug=False, _pinfo=None):
    """
    Use the CGI component of the API to query data files for any product within
    an inclusive time range. Optionally specify a latitude/longitude geographic
    location that must be contained within the data.

    https://ladsweb.modaps.eosdis.nasa.gov/api/v2/measurements/products

    :@param product_key: string key of the desired product. For valid options,
            see get_all_products(), or the LAADS DAAC link above.
    :@param start_time: datetime of the first valid minute in the range
    :@param end_time: datetime of the last valid minute in the range
    :@param latlon: tuple (lat, lon) in degrees specifying a location that must
            be contained within the data swath.
    :@param archive: Sometimes apparently-identical products are stored at
            multiple endpoints. If no archive is provided, defaults to the
            value provided in the API response.
    :@param _pinfo: Hidden workaround to provide the output of get_product_info
            for this product instead of querying the API for it again.
    """
    # Query the product info dict
    pinfo = _pinfo if not _pinfo is None else \
            get_product_info(product_key, debug=debug)
    # Determine which archive to use
    if not archive is None and archive not in pinfo["archives"].keys():
        raise ValueError(f"Provided archive {archive} not a valid option " + \
            f"for product {product_key};\nvalid: {pinfo['archives'].keys()}")
    print(pinfo)
    archive = str(archive) if not archive is None \
            else str(list(pinfo["archives"].keys())[0])
    url = api_root + "/content/details?products=" + product_key
    url += "&temporalRanges=" + start_time.strftime('%Y-%jT%H:%M')
    url += ".." + end_time.strftime('%Y-%jT%H:%M')
    if not latlon is None:
        # The API is currently forgiving enough to allow identical N/S and E/W
        # coordinates in a boundary box, which lets us query at a point without
        # worrying about overlapping a pole or the dateline.
        lat, lon = latlon
        url += f"&regions=[BBOX]W{lon} N{lat} E{lon} S{lat}"

    def recursive_page_get(url):
        """
        Small internal recursive method to aggregate all pages. This dubiously
        trusts that the API won't loop nextPageLinks, but whatever.
        """
        if debug: print(f"\033[32;1mQuerying new page: \033[34;0m{url}\033[0m")
        result = requests.get(url)
        if result.status_code != 200:
            raise ValueError(f"Invalid query. See response:\n{result.text}")
        res_dict = json.loads(result.text)
        next_page = res_dict.get('nextPageLink')
        this_result = [ {k:c[k] for k in ("downloadsLink", "illuminations")}
                       for c in res_dict["content"]
                       if str(c["archiveSets"])==archive ]
        if next_page is None or url == next_page:
            return this_result
        return this_result + recursive_page_get(next_page)
    return recursive_page_get(url)

def download(target_url:str, dest_dir:Path, raw_token:str=None,
             token_file:Path=None, replace:bool=False, debug=False):
    """
    Download a file with a wget subprocess invoking an authorization token.

    Generate a token here:
    https://ladsweb.modaps.eosdis.nasa.gov/profiles/#generate-token-modal

    :@param target_url: File path, probably provided by query_product().
    :@param dest_dir: Directory to download the new file into.
    :@param token_file: ASCII text file containing only a LAADS DAAC API token,
            which can be generated using the link above.
    """
    if not raw_token and not token_file:
        raise ValueError(f"You must provide a raw_token string or token_file.")

    if token_file:
        token = token_file.open("r").read().strip()
    else:
        token = raw_token
    #result = requests.get(target_url, stream=True, headers={
    #    'Authorization': k'Bearer {token}'})
    dest_path = dest_dir.joinpath(Path(target_url).name)
    if dest_path.exists():
        if not replace:
            raise ValueError(f"File exists: {dest_path.as_posix()}")
        dest_path.unlink()
    command = f"wget -e robots=off -np - -nH --cut-dirs=4 {target_url}" + \
            f' --header "Authorization: Bearer {token}"' + \
            f" -P {dest_dir.as_posix()}"
    if debug:
        print(f"\033[33;7mDownloading\033[0m \033[34;1;4m{target_url}\033[0m")
    stdout, stderr = Popen(shlex.split(command),
                           stdout=PIPE, stderr=PIPE).communicate()
    return dest_path

if __name__=="__main__":
    get_all_products(print_info=True)
    #print(query_product_day("VNP03IMGLL", dt(year=2022, month=7, day=22))[0])
    '''
    ppt(query_product(
            product_key="VJ102MOD",
            start_time=dt(year=2022, month=7, day=22, hour=16, minute=30),
            end_time=dt(year=2022, month=7, day=22, hour=17, minute=15),
            #latlon=(34, -86),
            debug=True
            ))
    '''
