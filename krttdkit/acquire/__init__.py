"""
The acquire module contains submodules for searching for and downloading data
from multiple host APIs, and for parsing specific data types into
straightforward and generalizable formats.

 - abi
 - get_goes
 - laads: Methods for interfacing with the MODAPS LAADS DAAC web API, which
   enables the user to search for many products based on time and geographic
   constraints
 - modis: Query the LAADS DAAC for MODIS data, download the corresponding
   hdf4 files, parse data, data descriptions, and geometric information into
   simple array formats.
 - viirs: Query the LAADS DAAC for VIIRS data, download the corresponding
   netCDF files, parse data, data descriptions, and geometric information into
   simple array formats.
"""
