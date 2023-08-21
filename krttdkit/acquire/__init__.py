"""
The acquire module contains submodules for searching for and downloading data
from multiple host APIs, and for parsing specific data types into
straightforward and generalizable formats.

:module abi: Methods for searching for and downloading and parsing ABI
    instrument data, information, and geometry into a simple array format.
:module get_goes: Namedtuples, classes, and methods for describing GOES
    products and interfacing with the NOAA AWS S3 buckets. GetGOES is an
    abstraction on the S3 API that provides a framework to search for products
    and individual files based on flexible search constraints.
:module laads: Methods for interfacing with the MODAPS LAADS DAAC web API,
    which enables the user to search for and download many products based on
    time and geographic constraints.
:module modis: Query the LAADS DAAC for MODIS data, download the corresponding
    hdf4 files, parse data, data descriptions, and geometric information into
    simple array formats.
:module viirs: Query the LAADS DAAC for VIIRS data, download the corresponding
    netCDF files, parse data, data descriptions, and geometric information into
    simple array formats.
"""
