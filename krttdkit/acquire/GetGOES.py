import s3fs
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from dataclasses import dataclass
from collections import namedtuple

from krttdkit.acquire.abi_desc import abi_desc
from krttdkit.visualize import TextFormat as TF

GOES_Product = namedtuple("GOES_Product", "satellite sensor level scan")
GOES_File = namedtuple("GOES_File", "product  stime label path")

class GetGOES:
    def __init__(self):
        self._s3 = s3fs.S3FileSystem(anon=True)
        self._valid_satellites = {"16", "17", "18"}

        self._products = self._get_product_api()
        # Dictionary of valid options for product constraints
        self._valid = {
                "satellite":self._valid_satellites,
                "sensor":tuple(set(p.sensor for p in self._products)),
                "level":tuple(set(p.level for p in self._products)),
                "scan":tuple(set(p.scan for p in self._products)),
                }

    def download(self, file:GOES_File, data_dir:Path, replace=False):
        """
        Download the provided GOES_File from the AWS S3 bucket to the provided
        directory, keeping the file's default naming convention.

        :@param file: GOES_File with a path to a valid AWS bucket location.
        :@param data_dir: Directory to deposit downloaded netCDF file
        :@param replace: If True, existing files with the same name will be
            overwritten by a newly-downloaded version
        """
        new_path = data_dir.joinpath(Path(file.path).name)
        if new_path.exists() and not replace:
            print(TF.RED(f"{new_path.as_posix()} exists already; skipping."))
        else:
            print(TF.BLUE(f"Downloading to ", bright=True),
                  TF.WHITE(new_path.as_posix(), bright=True))
            self._s3.download(file.path, new_path.as_posix())
        return new_path

    def _get_product_api(self):
        """
        Queries the NOAA GOES S3 bucket and returns a list of all available
        products as instances of the GOES_Product dataclass. This makes it
        easy to search for available products by attribute.

        This method relies on the NOAA product naming convention described here
        https://github.com/awslabs/open-data-docs/blob/main/docs/noaa/noaa-goes16/

        in which products are named by dash-separated strings like:
        [instrument]-[processing_level]-[product]

        :@return: list of GOES_Product objects for available bucket subdirs
        """
        products = []
        for satellite in self._valid_satellites:
            for r in self._s3.ls(f"noaa-goes{satellite}", refresh=True):
                tmp = r.split("/")[-1].split("-")
                # index.html is listed along with products; skip anything that
                # doesn't conform to the 3-field standard
                if len(tmp) != 3:
                    continue
                sensor, level, scan = tmp
                products.append(GOES_Product(satellite, sensor, level, scan))
        return products

    def list_products(self,satellite=None,sensor=None,level=None,scan=None):
        """
        Return a list of available products given a series of constraints
        """
        cand = self._products
        if satellite:
            if satellite not in self._valid["satellite"]:
                raise ValueError(
                        f"Provided satellite {satellite} is not one"
                        f" of the valid options {self._valid['satellite']}")
            cand = [c for c in cand if c.satellite==satellite]

        if sensor:
            if sensor not in self._valid["sensor"]:
                raise ValueError(
                        f"Provided sensor {sensor} is not one"
                        f" of the valid options {self._valid['sensor']}")
            cand = [c for c in cand if c.sensor==sensor]

        if level:
            if level not in self._valid["level"]:
                raise ValueError(
                        f"Provided level {level} is not one"
                        f" of the valid options {self._valid['level']}")
            cand = [c for c in cand if c.level==level]

        if scan:
            if scan not in self._valid["scan"]:
                raise ValueError(
                        f"Provided scan {scan} is not one"
                        f" of the valid options {self._valid['scan']}")
            cand = [c for c in cand if c.scan==scan]
        return cand

    def list_hour(self, product:GOES_Product, target_time:datetime,
                  label_substr:str=None):
        """
        List all of the available files for the provided product within the
        provided UTC hour. Many products are sub-divided by a label which is
        the second underscore-separated field of the filenames. If the
        optional desired label is provided, the search will be further
        restricted to files that match the label.
        """
        if any(p is None for p in product):
            raise ValueError(f"Provided product has a None field: {product}")
        product_key = "-".join((product.sensor, product.level, product.scan))
        s3_path = "noaa-goes{sat}/{product}/{year}/{jday}/{hour}".format(
                sat=product.satellite,
                product=product_key,
                year=target_time.strftime("%Y"),
                jday=target_time.strftime("%j"),
                hour=target_time.strftime("%H"),
                )
        paths = []
        for path in self._s3.ls(s3_path):
            _,label,_,timestr,_,_ = Path(path).name.split("_")
            if label_substr and label_substr not in label:
                continue
            paths.append(GOES_File(
                product=product,
                stime=datetime.strptime(timestr, "s%Y%j%H%M%S%f"),
                label=label,
                path=path,
                ))
        return paths

    def list_range(self, product:GOES_Product, init_time:datetime,
                   final_time:datetime, label_substr:str=None):
        """
        List all files with start times for the specified product falling
        within the provided range, with an inclusive init_time and exclusive
        final_time.

        :@param product: valid GOES_Product namedtuple.
        :@param init_time: Inclusive initial start time of files to return.
            Start time is the 4th field of the NOAA naming standard.
        :@param final_time: Exclusive final start time of files to return.
            Start time is the 4th field of the NOAA naming standard.
        :@param label_substr: Optional string label to restrict search results.
            The label is the second field of the NOAA naming standard, and
            typically denotes channels or scan modes.
        """
        assert init_time < final_time
        dhours = (final_time-init_time).total_seconds()//(60*60)
        tmp_time = datetime(init_time.year, init_time.month,
                            init_time.day, init_time.hour)
        files = []
        while tmp_time<final_time:
            files += self.list_hour(product, tmp_time, label_substr)
            tmp_time += timedelta(hours=1)

        files = [f for f in sorted(files,key=lambda f:f.stime)
                 if init_time<=f.stime and f.stime<final_time]
        return files

    def get_closest_to_time(self, product:GOES_Product, target_time:datetime,
                            label_substr:str=None, time_window_hrs=4):
        """
        Returns a list of products with the closest start time to the requested
        time.

        :@param product: Valid GOES_Product object with all fields filled out.
        :@param target_time: Desired start time of returned GOES_File object.
        :@param label_substr: bands and derived products are identified by a
            label in the 2nd underscore-separated field of the netCDF file
            name. If label_substr is a substring of one of these, only the
            files with the corresponding label are returned.
        :@param time_window_hrs: breadth of time window in which to search for
            files. May be useful for more transient products.
        """
        # All fields in the GOES_Product must be filled out
        assert all(product)
        start = target_time-timedelta(hours=time_window_hrs/2)
        end = target_time+timedelta(hours=time_window_hrs/2)
        files = self.list_range(product, start, end, label_substr)
        # If a label is provided, constrain files by substring
        if label_substr:
            files = [f for f in files if label_substr in f.label]
        # Get a list of time differences and return the minimum
        diffs = [abs((f.stime-target_time).total_seconds()) for f in files]
        closest = [files[i] for i in range(len(files)) if diffs[i]==min(diffs)]
        return closest

    def describe(self, product:GOES_Product):
        """
        Pretty-print product descriptions if available in abi_desc.py,
        returning None in every case. If you need the string version of
        descriptions, import the abi_desc module directly instead of using
        an instance of the GetGOES class.

        :@param product: valid product to describe
        """
        prod_str = "-".join(product[1:])
        if prod_str not in abi_desc.keys():
            print(TF.RED(f"Missing description for {prod_str}", bright=True))
        else:
            print(TF.BLUE(prod_str, bright=True) + " " + \
                    TF.WHITE(abi_desc[prod_str], bright=True))

def check_config():
    """
    """
    prods = GetGOES().list_products()
    cfg_prods = list(abi_desc.keys())
    cfg_prods = [GOES_Product("16", *p.split("-")) for p in cfg_prods]
    print(TF.RED("RED products are available but not described", bright=True))
    print(TF.RED("GREEN products are available and described", bright=True))
    for p in prods:
        if p not in  cfg_prods:
            print(TF.RED(p))
        else:
            print(TF.GREEN(p))


if __name__=="__main__":
    check_config()
    target_time = datetime(2022, 7, 14, 10, 36, 17)
    end_time = datetime(2022, 7, 15, 1, 26, 17)
    gg = GetGOES()
    for p in gg.list_products("17", "ABI", "L1b"):
        print(p)
    products = gg.list_range(
            product=GOES_Product("17", "ABI", "L1b", "RadC"),
            init_time=target_time,
            final_time=end_time,
            )
