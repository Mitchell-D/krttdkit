
"""
Methods for generating valid URLS and downloading (via HTTP) from
Goddard's GES DISC archive. Presently configured for  NLDAS2 and Noah-LSM data
"""
import pygrib
import numpy as np
from pathlib import Path
import requests
import multiprocessing as mp
from datetime import datetime as dt
from datetime import timedelta as td
import subprocess
import shlex
from http.cookiejar import CookieJar
import urllib

gesdisc_url = "https://hydro1.gesdisc.eosdis.nasa.gov"
# URL for GES DISC NLDAS2 data on a 0.125deg resolution grid
nldas2_url = f"{gesdisc_url}/data/NLDAS/NLDAS_FORA0125_H.002"
nldas2_template = "NLDAS_FORA0125_H.A{YYYYmmdd}.{HH}00.002.grb"
# URL for GES DISC run of Noah-LSM on the NLDAS2 domain
noahlsm_url = f"{gesdisc_url}/data/NLDAS/NLDAS_NOAH0125_H.002"
noahlsm_template = "NLDAS_NOAH0125_H.A{YYYYmmdd}.{HH}00.002.grb"

def hourly_noahlsm_urls(t0:dt, tf:dt):
    """
    Returns a list of URLs to hourly Noah LSM files in the EOSDIS archive,
    corresponding to the provided inclusive initial and exclusive final times.

    This method only compiles URL strings based on Goddard's HTTP standard,
    so there is no garuntee that the returned URLs actually link to an existing
    data file.

    Sub-hour time bound components are rounded down to the nearest whole hour.

    :@param: inclusive initial time of data range (up to hour precision)
    :@param: exclusive final time of data range (up to hour precision)
    """
    assert tf>t0
    # Round down to the nearest whole hour.
    t0 = dt(year=t0.year, month=t0.month, day=t0.day, hour=t0.hour)
    tf = dt(year=tf.year, month=tf.month, day=tf.day, hour=tf.hour)
    # Iterate over hours in the time range
    fhours = int(((tf-t0).total_seconds()))//3600
    ftimes = [ t0+td(hours=h) for h in range(fhours) ]
    return [f"{noahlsm_url}/{t.year}/{t.strftime('%j')}/"+\
            noahlsm_template.format(YYYYmmdd=t.strftime("%Y%m%d"),
                                    HH=t.strftime("%H"))
            for t in ftimes]

def hourly_nldas2_urls(t0:dt, tf:dt):
    """
    Returns a list of URLs to hourly NLDAS-2 files in the EOSDIS archive,
    corresponding to the provided inclusive initial and exclusive final times.

    This method only compiles URL strings based on Goddard's HTTP standard,
    so there is no garuntee that the returned URLs actually link to an existing
    data file.

    Sub-hour time bound components are rounded down to the nearest whole hour.

    :@param: inclusive initial time of data range (up to hour precision)
    :@param: exclusive final time of data range (up to hour precision)
    """
    assert tf>t0
    # Round down to the nearest whole hour.
    t0 = dt(year=t0.year, month=t0.month, day=t0.day, hour=t0.hour)
    tf = dt(year=tf.year, month=tf.month, day=tf.day, hour=tf.hour)
    # Iterate over hours in the time range
    fhours = int(((tf-t0).total_seconds()))//3600
    ftimes = [ t0+td(hours=h) for h in range(fhours) ]
    return [f"{nldas2_url}/{t.year}/{t.strftime('%j')}/"+\
            nldas2_template.format(YYYYmmdd=t.strftime("%Y%m%d"),
                                   HH=t.strftime("%H"))
            for t in ftimes]

def gesdisc_auth(username,password):
    """
    Initialize a session using a CookieJar in order to maintain authorization.
    """
    # Create a password manager to deal with the 401 reponse login
    pass_man = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    pass_man.add_password(None, nldas2_url, username, password)
    # Initialize a session using a CookieJar and install handlers
    opener = urllib.request.build_opener(
        urllib.request.HTTPBasicAuthHandler(pass_man),
        urllib.request.HTTPCookieProcessor(CookieJar()),
        #urllib.request.HTTPHandler(debuglevel=1),
        #urllib.request.HTTPSHandler(debuglevel=1),
        )
    urllib.request.install_opener(opener)
    return opener

def gesdisc_curl(urls:list, dl_dir:Path, cookie_file="~/.urs_cookies",
        skip_if_exists:bool=True, use_wget=False, debug=False):
    """
    GES DISC authentication is messy with OAuth in Python. Just curl it
    using the invasive GES DISC cookie file requirements :(
    https://uat.gesdisc.eosdis.nasa.gov/information/howto/How%20to%20Generate%20Earthdata%20Prerequisite%20Files

    :@param urls: List of string URLs to downloadable data files
    :@param dl_dir: local directory to dump downloaded files
    :@param skip_if_exists: Don't download existing files.
    """
    curl_command = "curl -n {cookie_file} -b {cookie_file} -LJO --url" + \
            " {url} -o {dl_path}"
    wget_command = "wget -nv --load-cookies {cookie_file} " + \
            "--save-cookies {cookie_file} -O {dl_path} -L {url}"
    paths = []
    for u in urls:
        dl_path = dl_dir.joinpath(Path(u).name)
        paths.append(dl_path)
        if dl_path.exists() and skip_if_exists:
            if debug:
                print(f"Skipping {dl_path.name}; exists already")
            continue
        if use_wget:
            cmd = shlex.split(wget_command.format(
                url=u, dl_path=dl_path, cookie_file=cookie_file))
        else:
            cmd = shlex.split(curl_command.format(
                url=u, dl_path=dl_path, cookie_file=cookie_file))
        if debug:
            print("\n"+u)
        subprocess.call(cmd)
    return paths

def gesdisc_download(urls:list, dl_dir:Path, auth=None, letfail:bool=True,
                  skip_if_exists:bool=True, debug=False):
    """
    Make GET requests to the provided URLs and download the response to a file
    in the provided directory named by the leaf of each URL path, which is
    assumed to be a valid data file.

    This relies on the GES DISC cookie requirements outlined here.
    https://uat.gesdisc.eosdis.nasa.gov/information/howto/How%20to%20Generate%20Earthdata%20Prerequisite%20Files

    :@param urls: List of string URLs to downloadable data files
    :@param dl_dir: local directory to dump downloaded files
    :@param auth: (user,pass) string tuple for a site, if necessary.
    :@param letfail: If True, failed requests don't raise an error.
    :@param skip_if_exists: Don't download existing files.
    """
    #auth = None if not auth else requests.auth.HTTPBasicAuth(*auth)
    opener = gesdisc_auth(*auth)
    for u in [ urls[0] ]:
        #request = urllib.request.Request(urls[0])
        # Skip already-downloaded files by default
        dl_path = dl_dir.joinpath(Path(u).name)
        if dl_path.exists() and skip_if_exists:
            print(f"Skipping {dl_path.name}; exists already")
            continue
        try:
            #urllib.request.urlretrieve(u, dl_path)
            #request = urllib.request.Request(u)
            #response = urllib.request.urlopen(request)
            response = opener.open(u, timeout=15)
        except Exception as E:
            if letfail:
                print(E)
                continue
            raise E
        response.read()
        print(f"Download success: {dl_path}")
    return

def nldas2_to_time(nldas2_path):
    """
    Parse the second and third fields of file names as a datetime

    This method is identical to noahlsm_to_time, but separate since I'm not
    sure whether the entire EOSDISC archive or GES DISC DAAC follow standard.

    :@param: nldas2_path conforming to the GES DISC standard of having the
        2nd and 3rd '.' -separated file name fields correspond to the date,
        for example: "NLDAS_FORA0125_H.A20190901.0000.002.grb"
    """
    return dt.strptime("".join(nldas2_path.name.split(".")[1:3]),"A%Y%m%d%H%M")

def noahlsm_to_time(nldas2_path):
    """
    Parse the second and third fields of file names as a datetime.

    This method is identical to nldas2_to_time, but separate since I'm not
    sure whether the entire EOSDISC archive or GES DISC DAAC follow standard.

    :@param: noahlsm_path conforming to the GES DISC standard of having the
        2nd and 3rd '.' -separated file name fields correspond to the date,
        for example: "NLDAS_NOAH0125_H.A20190901.0000.002.grb"
    """
    return dt.strptime("".join(nldas2_path.name.split(".")[1:3]),"A%Y%m%d%H%M")

if __name__=="__main__":
    debug = True
    data_dir = Path("/Users/mtdodson/Desktop/testbed/data")
    nldas2_dir = data_dir.joinpath("nldas2_2018")
    noahlsm_dir = data_dir.joinpath("noahlsm_2018")
    init_time = dt(year=2018, month=1, day=1)
    final_time = dt(year=2019, month=1, day=1)

    """ Download all of the files within the provided time range """

    # Generate strings for and download each hourly NLDAS2 file in range
    nldas_urls = hourly_nldas2_urls(t0=init_time, tf=final_time)
    gesdisc_curl(nldas_urls, nldas2_dir, debug=debug)
    exit(0)

    # Generate strings for and download each hourly Noah-LSM file in range
    lsm_urls = hourly_noahlsm_urls(t0=init_time, tf=final_time)

    # Download the Noah LSM files
    gesdisc_curl(lsm_urls, noahlsm_dir, debug=debug)
