"""
Collection of functions for downloading and parsing NOAA GOES ABI netCDF files
into a common format
"""
import netCDF4 as nc
import numpy as np
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from krttdkit.acquire.get_goes import GetGOES, GOES_Product, GOES_File
from krttdkit.products import GeosGeom

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
    :@param end_time:

    :@return: Nested lists of GOES_File objects corresponding to netCDF files
        downloaded to data_dir. The outer list indexes the band in the same
        order as provided band labels, and the inner lists contain GOES_File
        objects for that band, in order of increasing stime.
    """
    assert str(satellite) in tuple(map(str, range(16,19)))
    # Accept "RadM" style notation for consistency with product listing
    assert scan in ("C", "F", "M1", "M2")
    assert all(int(b) in range(1,17) for b in bands)
    search_str = f"Rad{scan[0]}"
    goesapi = GetGOES()
    prod = GOES_Product(satellite, "ABI", "L1b", search_str)
    if not end_time is None:
        files = goesapi.list_range(prod, start_time, end_time)
    else:
        files = goesapi.get_closest_to_time(prod, start_time)

    # If mesoscale scan requested, narrow down results to the requested view.
    if scan[0]=="M":
        files = [f for f in files if f.label.split("-")[2][-2:]==scan]
    paths_by_band = []
    files.sort(key=lambda f: f.stime)
    for b in bands:
        band_str = f"C{int(b):02}"
        band_files = [
                goesapi.download(f, data_dir, replace)
                for f in files
                if band_str in f.label.split("-")[-1]
                ]
        paths_by_band.append(band_files)
    return paths_by_band

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

def get_abi_l1b_scanangle(nc_file:Path, _ds=None):
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

def get_abi_l1b_latlon(nc_file:Path):
    """
    Uses geometric projection information stored in the provided NOAA-style
    ABI L1b netCDF file to calculate

    :@param nc_file: NOAA-style ABI L1b netCDF file
    :@return: 2-tuple like (lats, lons) where each member is a (M,N) shaped
        array of latitude and longitude floats in degrees for M vertical coords
        and N horizontal coords
    """
    ds = nc.Dataset(nc_file.as_posix())
    sa_ns, sa_ew = get_abi_l1b_scanangle(nc_file, _ds=ds)
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
    return ds["Rad"][:].data, np.ma.getmask(ds["Rad"][:])

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

def parse_goes_path(nc_file:Path):
    """
    Parses a NOAA-standard ABI file path, formatted like:
    OR_ABI-L1b-RadM1-M6C02_G16_s20232271841252_e20232271841309_c20232271841341.nc

    This method is meant to be generalized for any GOES instrument, so label
    sub-fields like ABI-L1b-RadM1-M6C02 aren't parsed.

    :@param nc_file: Path to a NOAA ABI file following the above format.

    :@return: 2-tuple like (band_string, file) where band_string is an ABI band
        like '02', and file is a GOES_File with all the appropriate fields.
    """
    _,label,satellite,stime,_,_ = nc_file.name.split("_")
    sensor,level,scan = label.split("-")[:3]
    return GOES_File(
            product=GOES_Product(
                # "G16" -> "16"
                satellite=satellite[-2:],
                sensor=sensor,
                level=level,
                scan=scan
                ),
            # "s20232271841252" -> datetime
            stime=datetime.strptime(stime, "s%Y%j%H%M%S%f"),
            label=label,
            path=nc_file.as_posix()
            )

if __name__=="__main__":
    data_dir = Path("/home/krttd/tools/pybin/krttdkit/tests/buffer/abi")
    abi_paths = [p for p in data_dir.iterdir() if "ABI-L1b" in p.name]
    #for p in abi_paths:
    #get_abi_l1b(abi_paths[0])
    start = datetime(2023,8,15,0,30)
    end = datetime(2023,8,15,1,0)
    bands = [2,5,14]

    #bands = download_l1b(data_dir, 16, "M2", [2, 5, 14], start, end)
    band_paths = download_l1b(data_dir, 16, "C", bands, start, end)
    for i in range(len(bands)):
        is_reflective = bands[i]<=7
        is_thermal = bands[i]>=7
        for path in band_paths[i]:
            if is_reflective:
                ref, mask = get_abi_l1b_ref(path, get_mask=True)
                print(np.min(ref),np.max(ref))
            if is_thermal:
                tb, mask = get_abi_l1b_Tb(path, get_mask=True)
                print(np.min(tb),np.max(tb))
