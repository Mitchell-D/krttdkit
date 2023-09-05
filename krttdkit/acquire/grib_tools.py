"""
General methods for interfacing with the grib format, including methods to
extract a series of TimeGrid-style ".npy" arrays using the __main__ context.
"""

import pygrib
import numpy as np
from pathlib import Path
import multiprocessing as mp
import subprocess
import shlex
#from http.cookiejar import CookieJar
#import urllib

def _parse_pixels(args:tuple, debug=False):
    """
    :@param args: tuple of args like (file path, list of 2d index tuples)
    :@return: (N,B(,..)) shaped array for N pixels each with B records.
    """
    if debug:
        print(f"Opening {args[0]}")
    try:
        data,_,_ = get_grib1_data(args[0])
        return np.dstack(data)[tuple(zip(*args[1]))]
    except Exception as e:
        raise Exception(f"Issue with file {args[0]}:\n{e}")


def grib_parse_pixels(pixels:list, grib1_files:list, chunk_size:int=1,
                      workers:int=None,debug=False):
    """
    Opens a grib1 file and extracts the values of pixels at a list of
    indeces like [ (j1,i1), (j2,i2), (j3,i3), ], returning a list of
    tuples like [ (fname, [val1,val2,val3]), (fname, [val1,val2,val3]), ]

    This method uses a multiprocessing Pool by default in order to parallelize
    the pixel parsing without having many arrays open simultaneously.

    Assumes the provided grib1 file paths are valid, and that each of the
    records in the file are on a uniform grid indexed in the first 2 dims
    by the provided list of pixels.

    :@param pixels: list of 2-tuple indeces for the first 2 dimensions of all
        valid grib1 arrays, for each of the requested components.
    :@param grib1_files: list of string file paths to grib1 files containing
        records on equally-sized grids. No geographic coordinates are
        retrieved, so they are assumed to be consistent or otherwise available.
    :@param chunk_size: Number of values assigned to each thread at a time
    """
    # Using imap, extract values from each 1st and 2nd dim index location
    with mp.Pool(workers) as pool:
        args = [(g,pixels,debug) for g in grib1_files]
        results = list(pool.imap(_parse_pixels, args))
    return results

def wgrib_tuples(grb1:Path):
    """
    Calls wgrib on the provided file as a subprocess and returns the result
    as a list of tuples corresponding to each record, the tuples having string
    elements corresponding to the available fields in the grib1 file.
    """
    wgrib_command = f"wgrib {grb1.as_posix()}"
    out = subprocess.run(shlex.split(wgrib_command), capture_output=True)
    return [tuple(o.split(":")) for o in out.stdout.decode().split("\n")[:-1]]

def wgrib(grb1:Path):
    """
    Parses wgrib fields for a grib1 file into a dict of descriptive values.
    See: https://ftp.cpc.ncep.noaa.gov/wd51we/wgrib/readme
    """
    return [{"record":int(wg[0]),
             "name":wg[3],
             "lvl_str":wg[11], # depth level
             "mdl_type":wg[12], # Model type; anl or fcst
             "date":wg[2].split("=")[-1],
             "byte":int(wg[1]),
             "param_pds":int(wg[4].split("=")[-1]), # parameter/units
             "type_pds":int(wg[5].split("=")[-1]), # layer/level type
             "vert_pds":int(wg[6].split("=")[-1]), # Vertical coordinate
             "dt_pds":int(wg[7].split("=")[-1]),
             "t0_pds":int(wg[8].split("=")[-1]),
             "tf_pds":int(wg[9].split("=")[-1]),
             "fcst_pds":int(wg[10].split("=")[-1]), # Forecast id
             "navg":int(wg[13].split("=")[-1]), # Number of grid points in avg
             } for wg in wgrib_tuples(grb1)]

def get_grib1_data(grb1_path:Path):
    """
    Parses grib1 file into a series of scalar arrays of the variables,
    geographic coordinate reference grids, and information about the dataset.

    :@param grb1_path: Path of an existing grb1 file file with all scalar
        records on uniform latlon grids.
    :@return: (data, info, geo) such that:
        data -> list of uniform-shaped 2d scalar arrays for each record
        info -> list of dict wgrib results for each record, in order.
        geo  -> 2-tuple (lat,lon) of reference grid, assumed to be uniform
                for all 2d record arrays.
    """
    f = grb1_path
    assert f.exists()
    gf = pygrib.open(f.as_posix())
    geo = gf[1].latlons()
    gf.seek(0)
    # Only the first entry in data is valid for FORA0125 files, the other
    # two being the (uniform) lat/lon grid. Not sure how general this is.
    data = [ d.data()[0] for d in gf ]
    return (data, wgrib(f), geo)

""" Below was previously extract_nldas_subgrid.py """

def _extract_nldas_subgrid(args:tuple):
    """
    Extract a subgrid of the specified nldas grib1 file storing a series of
    records, and save it to a .npy binary file in the provided directory.

    args = [grib1_path, file_time, vert_slice, horiz_slice, output_dir,
            record_list, data_label]
    """
    try:
        ftype = args[6] #args[0].name.split("_")[1]
        time = args[1] # gesdisc.nldas2_to_time(args[0]).strftime("%Y%m%d-%H")
        new_path = Path(args[4].joinpath(f"{ftype}_{time}.npy"))
        if new_path.exists():
            print(f"Skipping {new_path.as_posix()}; exists!")
            return
        alldata,info,_ = get_grib1_data(args[0])
        data = []
        # append records in order
        for r in args[5]:
            data.append(next(
                alldata[i].data for i in range(len(alldata))
                if info[i]["record"]==r
                ))
        data = np.dstack(data)[args[2],args[3]]
        np.save(new_path, data)
    except Exception as e:
        #print(f"FAILED: {args[0]}")
        print(e)

def mp_extract_nldas_subgrid(grib_files:list, file_times:list, v_bounds:slice,
                             h_bounds:slice, out_dir:Path, records:list,
                             data_label:str, nworkers:int=1):
    """
    Multiprocessed method to extract a pixel subgrid of NLDAS2-grid grib1
    forcing files, which includes data from the NLDAS run of the Noah-LSM.
    Extracted array subgrids are stored as ".npy" serial files

    :@param grib_files: List of valid grib1 files to extract arrays from.
    :@param file_times: List of datetimes associated with each grib1 file.
    :@param v_bounds: Slice of vertical coordinates to subset the array.
    :@param h_bounds: Slice of horizontal coordinates to subset the arrays.
    :@param out_dir: Directory to deposit new ".npy" files in
    :@param records: List of record numbers from wgrib for data to keep in
        the serial files.
    :@param data_label: Identifying string for the dataset being extracted.
        This is the first underscore-separated field in the generated ".npy"
        arrays, followed by the date in YYYYmmdd-HH format.
    :@param nworkers: Number of subprocesses to spawn concurrently in order
        to extract files.
    """
    assert len(file_times)==len(grib_files)
    with mp.Pool(nworkers) as pool:
        args = [(Path(grib_files[i]), file_times[i], v_bounds, h_bounds,
                 out_dir, records, data_label)
                for i in range(len(grib_files))]
        results = pool.map(_extract_nldas_subgrid, args)

def merge_grids(label_a:str, label_b:str, new_label:str, data_dir:Path):
    """
    Simple and high-level method to merge collections of TimeGrid-style ".npy"
    arrays along the feature axis, saving them as new grids.

    This method assumes...
    - both array collections are in the same directory
    - both collections conform to underscore-separated naming like
      <data label>_<YYYYmmdd-HH time>.npy
    - both collections have identical timesteps within the directory
    - both collections have identically-shaped first and second axes

    If label_a refers to (M,N,A) shaped data with 'A' features and label_b to
    (M,N,B) data with 'B' features, the final array will be (M,N,A+B) shaped,
    such that the 'A' dataset's features are indexed first, extended by 'B'.

    :@param label_a: Unique string equivalent to the first underscore-separated
        field of every ".npy" file in the 'A' dataset.
    :@param label_b: Unique string equivalent to the first underscore-separated
        field of every ".npy" file in the 'B' dataset.
    :@param new_label: New unique string to use as the first field of ".npy"
        files generated as a combination of datasets 'A' and 'B'
    :@param data_dir: directory both where array files are retrieved and where
        new combined data arrays are serialized and deposited.
    """
    paths_a = [p.name for p in data_dir.iterdir() if label_a in p.name]
    paths = [(data_dir.joinpath(p),
              data_dir.joinpath(p.replace(label_a,label_b)),
              datetime.strptime(p.split("_")[1],"%Y%m%d-%H.npy"))
             for p in paths_a]
    assert all(p[0].exists() and p[1].exists() for p in paths)
    for a,b,t in paths:
        X = np.dstack((np.load(a),np.load(b)))
        new_path = data_dir.joinpath(f"{new_label}_{t.strftime('%Y%m%d-%H')}")
        print(f"Saving {new_path.as_posix()}")
        #np.save(new_path,X)

if __name__=="__main__":
    data_dir = Path("data")
    # Directory containing only grib files to be extracted
    grib_dir = data_dir.joinpath("noahlsm_2020")
    # Directory where subgrid files fill be deposited
    out_dir = data_dir.joinpath("buffer/tg_tmp")
    # Vertical and horizontal pixel range of subgrid to extract
    v_bounds = slice(64,192)
    h_bounds = slice(200,328)
    # Record numbers (per wgrib) of fields to extract.
    # records = list(range(1,12)) # nldas2 (all)
    records = list(range(25,34)) # noahlsm (SOILM + LSOIL fields)

    #'''
    import gesdisc
    grib_files, file_times = zip(*[(f,gesdisc.nldas2_to_time(f))
        for f in grib_dir.iterdir() if "NOAH0125" in f.name])
    mp_extract_nldas_subgrid(
            grib_files=grib_files,
            file_times=file_times,
            v_bounds=v_bounds,
            h_bounds=h_bounds,
            out_dir=out_dir,
            data_label="noahlsm",
            records=records, # noahlsm (SOILM + LSOIL)
            nworkers=4,
            )
    #'''
    #merge_grids("FORA0125","NOAH0125","newlabel",Path("data/old_tg"))
