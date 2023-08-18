"""
Straightforward script for searching for valid GOES products and files,
and downloading the corresponding netCDF files given time constraints.
"""
import argparse
import re

from datetime import datetime
from datetime import timedelta
from pathlib import Path

#from krttdkit.operate import enhance as enh
from krttdkit.acquire.get_goes import GetGOES, GOES_Product, GOES_File
from krttdkit.acquire.get_goes import visual_search
from krttdkit.acquire import abi
from krttdkit.visualize import TextFormat as TF

if __name__=="__main__":
    """ ----------(Configure product and desired time ranges)---------- """
    # Directory where any netCDF files are downloaded
    data_dir = Path("/home/krttd/tools/pybin/krttdkit/tests/buffer/abi")

    # Highly electrified line moving through around sunset.
    target_time = datetime(2023, 8, 14, 23)
    search_window = timedelta(hours=2)
    #search_window = None

    # GOES product search constraints. Leave some fields None to list options.
    query = GOES_Product(
            satellite="18",
            #satellite=None,
            sensor="ABI",
            #sensor=None,
            level="L1b",
            #level=None,
            scan="RadC",
            #scan=None,
            )
    # Substring of all returned file labels. May be None
    # Labels are the second underscore-separated field of the netCDF files.
    #label_substr = "M2-M6C16"
    label_substr = None
    # If True, downloads all returned data files.
    download = False

    """ --------------------------------------------------------------- """

    """
    Search the S3 bucket for GOES products or files matching the provided
    constraints.
    """
    # GetGOES object retrieves and stores the AWS API at init
    goesapi = GetGOES(refetch_api=False)
    results = visual_search(
            query=query,
            time=target_time,
            search_window=search_window,
            label_substr=label_substr,
            goesapi=goesapi,
            )
    has_files = type(results[0]) is GOES_File
    """
    If the query resulted in a list of valid files and download=True,
    download all netCDF files in the list to data_dir
    """
    if download and has_files:
        for f in files:
            goesapi.download(f, data_dir, replace=False)

    #from krttdkit.acquire.get_goes import list_config
    #list_config(satellite="17", goesapi=goesapi)
