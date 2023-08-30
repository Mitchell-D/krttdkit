"""
This module contains functions for executing and parsing results from the
SBDART model https://github.com/paulricchiazzi/SBDART

the __main__ context (at the bottom) serves as a unit test, and evaluates
the provided input for all outputs, performing data integrity checks for
each of the runs.

======= Basic Usage =======
The most minimal use of the sbdart module is the following:
```
from pathlib import Path
from krttdkit.acquire.sbdart import dispatch_sbdart, parse_iout
iout_id = 7
output = dispatch_sbdart({"iout":iout_id}, Path("/tmp"))
data = parse_iout(iout_id=iout_id, output)
```
After running this code, the data variable will contain a dictionary
of data for output style 7 and all-default settings.
"""
from pathlib import Path
from krttdkit.acquire.sbdart_info import default_params
import shlex
from subprocess import Popen, PIPE
import itertools as it
import shutil
import numpy as np

out_labels = {
        "rad":"Radiance (in W/m^2/sr) for each phi/uzen combo",
        "srad":"Spectral radiance (in W/m^2/um/sr)",
        "flux":"Flux (in W/m^2)",
        "sflux":"Spectral flux (in W/m^2/um)",
        "flux_labels":"String labels for flux values",
        "sflux_labels":"String labels for spectral flux values",
        "srad_labels":"String labels for spectral radiance values",
        "phi":"Azimuth angle in degrees",
        }
{
        "ffew":"filter function equivalent width (um)",
        "topdn":"total downward flux at ZOUT(2) km (w/m^2) ",
        "topup": "total upward flux at ZOUT(2) km (w/m^2)",
        "topdir": "direct downward flux at ZOUT(2) km (w/m^2)",
        "botdn": "total downward flux at ZOUT(1) km (w/m^2)",
        "botup": "total upward flux at  ZOUT(1) km (w/m^2)",
        "botdir": "direct downward flux at ZOUT(1) km (w/m^2)",
        }

def dispatch_sbdart(params:dict, tmp_dir:Path, sbdart_bin:Path=Path("sbdart")):
    """
    Run SBDART with the provided parameters, generating an output file
    conforming to the SBDART iout parameter's standards
    """
    dkeys, dvals, ddescs = zip(*default_params)
    assert all(k in dkeys for k in params.keys())
    if tmp_dir.exists():
        raise ValueError(
                f"Cannot use temporary directory; {tmp_dir} exists!")
    if not tmp_dir.parent.exists():
        raise ValueError(
                f"Parent directory of tmp_dir {tmpdir} doesn't exist!")

    tmp_dir.mkdir()
    args = " ".join(["=".join(map(str,p)) for p in params.items()])
    tmp_dir.joinpath("INPUT").open("w").write(f"&INPUT {args}/\n")
    stdout, stderr = Popen(sbdart_bin, cwd=tmp_dir.as_posix(),
                           stdout=PIPE,stderr=PIPE).communicate()
    shutil.rmtree(tmp_dir.as_posix())
    if stderr:
        raise ValueError(stderr)
    return stdout

def split_record(record:bytes, sep:bytes=b" "):
    """
    Strips and decodes a bytes record into a tuple, converting to float if
    possible and removing any blank entries.

    :@param record: String of bytes composed of string or float entries.
    :@param sep: Separator for splitting entries
    """
    def _tentative_float(arg):
        """
        Converts an argument to a float, if possible, returning either the
        original value or a float version of it.
        """
        try:
            arg = float(arg)
        except ValueError:
            pass
        return arg
    record = (v.strip().decode() for v in tuple(record.split(sep)) if v)
    return tuple(map(_tentative_float, record))

def get_records(sb_out:list, record_width:int=None):
    """
    Given a list of byte strings corresponding to features separated by
    multiple spaces, extracts all features from each row into a tuple;

    If a record_width parameter is provided, only tuples with the specified
    number of numeric values will be included in the returned records. This
    can help filter out other unneeded data

    :@param sb_out: List of byte strings lik dispatch_sbdart() output
    :@param record_width: If provided, only records (rows) with the specified
        number of values will be returned.
    """
    # split each record (row) into float or string values
    records = [r for r in map(split_record,sb_out) if len(r)]
    if record_width is None:
        return records
    # otherwise filter out records that don't match the provided length
    return list(filter(lambda r: len(r)==record_width, records))

def _parse_iout1(sb_out:bytes):
    """
    Parses DISORT output for the IOUT=5

    This option provides upward, downward, and direct flux totals at TOA and
    BOA for each wavelength independently.

    Feature labels (in same order as returned record arrays)
    ("wl","ffv","topdn","topup","topdir","botdn","botup","botdir")

    :@param sb_out: List of byte strings corresponding to each line of stdout
        conforming to the iout1 column format, as shown above.
    :@return: dict containing data for keys:
        sflux: 8-tuple of strings labels for spectral flux of features shown
            above; 8-tuple of 1d arrays containing float data
    """
    sb_out = sb_out.split(b"\n")[3:]
    labels = ("wl","ffv","topdn","topup","topdir","botdn","botup","botdir")
    lines = tuple(map(np.array, zip(*get_records(sb_out,record_width=8))))
    return {"sflux":lines, "sflux_labels":labels}

def _parse_iout5(sb_out:list):
    """
    Parses DISORT output for the IOUT=5

    This option provides flux totals at each wavelength similar to iout=1,
    but also includes a 3d array of TOA spectral radiances in W/m^2/um/sr. The
    three axes correspond to a (U,P,W) for U user zenith angles, P azimuth
    (phi) angles, and W wavelengths.

    :@param sb_out: list of bytes strings, like dispatch_sbdart output.
    :return: dict of results values with the following keys:
        sflux:       tuple of 1d arrays corresponding to flux labels.
        sflux_labels: labels corresponding to the order of the flux arrays.
        uzen:        nzen float values for user zenith angles in degrees
        phi:         nphi float values for azimuth angles in degrees
        wl:          num_wl float values wavelengths in um
        srad:        3d (nzen,nphi,num_wl) shaped array of float
                     spectral radiances.
    """
    #sb_out = tuple(map(lambda r:r.replace(b"\n",b""), sb_out.split(b"\n\n")))
    sb_out = sb_out.split(b"\n")
    records = get_records(sb_out, record_width=None)
    num_wl = int(records[1][0])
    records = records[2:]
    assert not len(records)%num_wl # num_wl should divide record count
    # Subdivide the output into equivalent-format records by wavelength.
    # Unfortunately this involves considerable tuple manipulation since
    # byte records are spread across multiple lines.
    wl_rec_size = len(records)//num_wl
    wl_recs = [records[wl_rec_size*i:wl_rec_size*(i+1)] for i in range(num_wl)]
    sflux = tuple(map(np.array, zip(*[r[0] for r in wl_recs])))
    nphi, nzen = map(int,wl_recs[0][1])
    phi_and_uzen = tuple(it.chain(*wl_recs[0]))[10:10+nzen+nphi]
    uzen = phi_and_uzen[nphi:]
    phi = phi_and_uzen[:nphi]
    rad_per_wl = []
    for i in range(num_wl):
        rad = np.array(tuple(it.chain(*wl_recs[i]))[10+nzen+nphi:])
        rad = np.reshape(rad, (nzen,nphi))
        rad_per_wl.append(rad)
    srad = np.dstack(rad_per_wl)

    # Get the spectral fluxes by parsing each wavelength record
    # Make a dictionary of data from the record
    return {"sflux":sflux,
            "sflux_labels":(
                "wl","ffv","topdn","topup","topdir", "botdn","botup","botdir"),
            "wl":sflux[0],
            "uzen":uzen,
            "phi":phi,
            "srad":srad,
            }

def _parse_iout7(sb_out:list):
    """
    Parses DISORT output for iout=7

    This option provides spectral radiative flux at each wavelength and
    atmospheric layer

    :@param sb_out:Raw bytes exactly like output from dispatch_sbdart when
        the iout paramater is set to 7.
    :@return: dict of results values including keys for:
        labels: tuple of string feature labels for the first axis
        wl: array of float values for wavelengths labeling the second axis.
        sflux: Length F tuple of (W,Z) shaped numpy arrays for F features,
            W wavelengths, and Z vertical bins in the atmospheric profile.
    """
    # Split records by double-newline just to get nz
    records = get_records(sb_out.split(b"\n\n"))
    nz = int(records[0][1])
    # ravel all data by splitting on spaces
    all_data = tuple(it.chain(*records[1:]))
    # Weak but apparently consistent assumption that each wavelength is
    # separated by a double new-line
    nw = sum(v=='' for v in all_data)
    assert len(all_data)%nw == 0
    # record width including 5*nz data records and nw labels
    rec_width = len(all_data) // nw
    # Split into data records per wavelength
    wl_recs = [all_data[rec_width*i:(i+1)*rec_width][1:] for i in range(nw)]
    wavelengths = [r[0] for r in wl_recs]
    # Parse out the 5 nz-length rows from each wavelength record
    sflux = [[r[j*nz+1:(j+1)*nz+1] for j in range(5)] for r in wl_recs]
    # If the following is False, it suggests that the row parsing is asymmetric
    assert all([f[0]==sflux[0][0] for f in sflux[1:]])
    assert len(sflux)==nw
    sflux = np.asarray(sflux).transpose((1,0,2))
    z = sflux[0,0]
    return {
            "wl":wavelengths,
            "z":sflux[0,0],
            "sflux_labels":("fdird", "fdifd", "flxdn", "flxup"),
            "sflux":tuple(sflux[i] for i in range(1,sflux.shape[0])),
            }

def _parse_iout10(sb_out):
    """
    Parses DISORT output for iout=10

    This option provides a single record with total wavelength-integrated
    upward, downward, and direct radiative flux at zout[0] and zout[1].

    :@param sb_out:Raw bytes exactly like output from dispatch_sbdart when
        the iout paramater is set to 10.
    :@return: dict of results values including keys for:
        labels: tuple of string feature labels for each value
        totals:9-tuple total wavelength-integrated radiative fluxes at the
            vertical locations specified by zout.
    """
    labels = ("wlinf", "wlsup", "ffew", "topdn", "topup", "topdir",
              "botdn", "botup", "botdir")
    totals = tuple(float(v) for v in sb_out.split(b" ") if v)
    assert len(labels)==len(totals)
    return {"flux":totals, "flux_labels":labels}

def _parse_iout11(sb_out:list):
    """
    Parses DISORT output for iout=11

    This option provides wavelength-integrated upward and downward fluxes
    and heating rates for each atmospheric model layer specified by ngrid.

    :@param sb_out: Raw bytes exactly like output from dispatch_sbdart when
        the iout paramater is set to 11.
    :@return: dict of results values including keys for:
        flux_labels: tuple of string feature labels for the flux arrays
        flux: F length tuple of 1d numpy arrays values corresponding to the
            F flux labels. Each has G entries; one for each grid layer.
    """
    labels = ("z", "p", "fxdn", "fxup", "fxdir", "dfdz", "heat")
    records = get_records(sb_out.split(b"\n"), record_width=7)
    flux = tuple(map(np.array,zip(*records)))
    return {"flux_labels":labels, "flux":flux}

def _parse_iout20(sb_out):
    """
    Parses DISORT output for iout=20

    This option provides top and bottom wavelength-integrated total fluxes as
    well as radiance (in W/m^2/sr) for each phi/uzen combination specified by
    parameters: (nzen, uzen, nphi, phi)

    :@param sb_out: Raw bytes exactly like output from dispatch_sbdart when
        the iout paramater is set to 20.
    :@return: dict of results values including keys for:
        flux: Flux totals at points specified by zout.
        flux_labels: string labels for each flux quantity
        phi: P azimuth angles in degrees
        uzen: U user zenith angles in degrees
        rad: (U,P) shaped numpy array for U uzer zenith and P azimuth (phi)
            angles corresponding to wavelength-integrated radiances in the
            provided direction.
    """
    labels = ("wlinf", "wlsup", "ffew", "topdn", "topup", "topdir",
              "botdn", "botup", "botdir")
    # Split byte records by line
    sb_split = get_records(sb_out.split(b"\n"))
    # First numerical row contain flux values with the labels above
    flux = sb_split[0]
    nphi,nzen = map(int,sb_split[1])
    # Combine all data into a single array
    alldata = np.array(tuple(it.chain(*sb_split[2:])))
    # Extract phi, uzen, and radiance entries by their position
    phi = alldata[:nphi]
    uzen = alldata[nphi:nphi+nzen]
    rad = np.reshape(alldata[nphi+nzen:], (nzen,nphi))
    return {"flux":flux,"flux_labels":labels,"phi":phi,"uzen":uzen,"rad":rad}

def _parse_iout22(sb_out:list):
    """
    Parses DISORT output for the IOUT=22

    This option provides wavelength-integrated radiance and flux at each
    atmospheric layer. This includes upward, downward, and direct-beam flux
    through each layer (in W/m^2) and radiance at every layer/uzen/phi
    combination (in W/m^2/sr).

    :@param sb_out: list of bytes strings, like dispatch_sbdart output.
    :return: dict of results values with the following keys:
        flux:      tuple of 1d arrays corresponding to flux labels.
        flux_labels: labels corresponding to the order of the flux arrays.
        uzen:        nzen float values for user zenith angles in degrees
        phi:         nphi float values for azimuth angles in degrees
        z:           nz float values for layer altitude in km
        rad:         3d (nzen,nphi,nz) shaped array of float radiances.
    """
    records = get_records(sb_out.split(b"\n"))
    nphi, nzen, nz, ffew = records[0]
    nphi, nzen, nz = map(int, (nphi, nzen, nz))
    # Get a single tuple containing all remaining values
    all_vals = tuple(it.chain(*records[1:]))
    # Parse phi, use zenith, and altitude arrays
    phi = np.array(all_vals[:nphi])
    uzen = np.array(all_vals[nphi:nphi+nzen])
    z = np.array(all_vals[nphi+nzen:nphi+nzen+nz])
    # Parse out each of the three flux columns
    flux = tuple([np.array(all_vals[nphi+nzen+nz+i*nz:nphi+nzen+nz+i*(nz+1)])
              for i in range(3)])
    flux_labels = ("fxdn", "fxup", "fxdir")
    # Parse out the (nz, nzen, nphi) shaped radiances array for each layer
    rad = np.array(all_vals[nphi+nzen+nz*4:])
    # Reshape the array to (nzen, nphi, nz) to mimic iout5 output
    rad = np.reshape(rad, (nz, nzen, nphi)).transpose((1,2,0))
    return {"flux":flux, "flux_labels":flux_labels,
            "uzen":uzen, "phi":phi, "z":z, "rad":rad}

def parse_iout(iout_id:int, sb_out:list, print_stdout:bool=True):
    """
    Parses the bytes values returned from stdout by disbatch_sbdart when
    the parameter iout==1.

    valid iout ids: {1, 5, 6, 7, 10, 11, 20, 21, 22, 23}

    :@param iout_id: Integer iout
    :@param sb_out: List of byte strings corresponding to each line of stdout,
        expected to be formatted identically to the output of disbatch_sbdart
    """
    parse_funcs = {1:_parse_iout1,
                   5:_parse_iout5,
                   6:_parse_iout5, # Output identical to iout=5
                   7:_parse_iout7,
                   10:_parse_iout10,
                   11:_parse_iout11,
                   20:_parse_iout20,
                   21:_parse_iout20, # Output identical to iout=20
                   22:_parse_iout22,
                   23:_parse_iout20, # Output identical to iout=20
                   }
    if print_stdout:
        for r in sb_out.split(b"\n"):
            print(r)
        print()
    assert iout_id in parse_funcs.keys()
    data = parse_funcs[iout_id](sb_out)
    return data

if __name__=="__main__":
    tmp_dir = Path("/tmp/sbdart")
    args = {
            "idatm":1,
            "isat":2,
            "wlinf":.5,
            "wlsup":13.0,
            "wlinc":.05,
            "iout":22,
            "nstr":30,
            "zcloud":"10,14",
            "tcloud":"8,13",
            #"zout":"0,100",
            "ngrid":42,
            "nphi":4,
            "phi":"0,90",
            "nzen":25,
            "uzen":"0,85",
            #"sza":60,
            }
    allkeys = set()
    # output types should be consistent within each of these dicts
    coords = {"z":[], "phi":[], "wl":[], "uzen":[]}
    arrays = {"rad":[], "srad":[], "flux":[], "sflux":[]}
    labels = {"flux_labels":[], "sflux_labels":[]}
    for iout in (1,5,6,7,10,11,20,21,22,23):
        args["iout"] = iout
        data = parse_iout(
                iout_id=iout,
                sb_out=dispatch_sbdart(args, tmp_dir),
                print_stdout=False,
                )
        # Keep track of the all data output types
        for unit_dict in (coords, arrays, labels):
            for k in sorted(data.keys()):
                if k in unit_dict.keys():
                    unit_dict[k].append(data[k])

    # Validate all coordinate axes
    for k,A in coords.items():
        # All coordinate arrays must be identical for each axis
        if not all(tuple(a)==tuple(A[0]) for a in A[1:]):
            print(f"Not all coordinate rows uniform for axis {k}")
    # Validate all arrays
    for k,V in arrays.items():
        assert all(type(v) in (np.ndarray,tuple) for v in V)
    # Check that there is a label for every 1d flux array in each run
    for k,L in labels.items():
        # All labels must be strings
        assert all(all(type(l) is str for l in labels) for labels in L)
    for l,f in zip(labels["flux_labels"],arrays["flux"]):
        # Each label array must have the same length as its flux array
        assert len(l)==len(f)
        # Each tuple element is either a ndarray or a float value
        assert all(type(v) in (np.ndarray,float) for v in f)
    for l,f in zip(labels["sflux_labels"],arrays["sflux"]):
        # Each label array must have the same length as its flux array
        assert len(l)==len(f)
        # Each tuple element is either a ndarray or a float value
        assert all(type(v) in (np.ndarray,float) for v in f)
