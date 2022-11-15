import os
from pathlib import Path
from boto.s3.connection import S3Connection
from datetime import datetime as dt
from datetime import timedelta as td


def get_goes_hour(bucket, product, band, datadir, time, mode="3", goes_version:int=16, skip:int=1):
    connection = S3Connection(anon=True)
    bucket = connection.get_bucket(bucket)

    prefix = product + '/' + str(time.year) + '/' + time.strftime('%j').zfill(3) \
                + '/' + str(time.hour).zfill(2) + '/'
    filename = "OR_"+product+"-M"+mode+"C"+band.zfill(2)+f"_G{goes_version}_"

    print(prefix+filename)

    blist = list(bucket.list(prefix=prefix+filename))
    #blist = list(bucket.list(prefix=prefix))

    times = [
        dt.strptime(
            os.path.basename(f.name).split("_")[-1][:-4],
            "c%Y%j%H%M%S") + td(seconds=float(
                os.path.basename(f.name).split("_")[-1][-4])/10.)
        for f in blist
        ]

    for f in blist[::skip]:
        basename = Path(f.key).name
        data_dir = Path(datadir).joinpath(basename)
        with open(data_dir.as_posix(), "w") as ncfile:
            print(f"downloading {basename} to {data_dir.as_posix()}")
            f.get_contents_to_file(ncfile)

if __name__=="__main__":
    goes_version = 16
    bucket = f"noaa-goes{goes_version}"
    product = "ABI-L1b-RadC" # L1b CONUS domain
    #product = "ABI-L1b-RadF" # L1b Full-disk
    #product = "ABI-L2-MCMIPC" # L2 Cloud/Moisture imagery CONUS

    # Only get the file every "skip" indeces
    skip=3
    band = str(5)
    # In special cases, may need to check scan mode for date/time range
    mode = str(6)
    #datadir = Path("/home/krttd/uah/22.f/aes572/hw3/data/great_lakes/band05")
    #datadir = Path("/mnt/warehouse/data/abi/aes572_hw3/laura_conus/band03")

    datadir = Path("/mnt/warehouse/data/abi/aes572_hw2/ian_data/band05")

    init_time = dt(year=2022, month=9, day=27, hour=13)
    final_time = dt(year=2022, month=9, day=28, hour=2)

    #init_time = dt(year=2020, month=8, day=27, hour=12)
    #final_time = dt(year=2020, month=8, day=27, hour=17)
    #init_time = dt(year=2018, month=2, day=14, hour=13)
    #final_time = dt(year=2018, month=2, day=14, hour=23)

    init_time = init_time.replace(minute=0, second=0, microsecond=0)
    final_time = final_time.replace(minute=0, second=0, microsecond=0)
    d_hours = int((final_time.timestamp()-init_time.timestamp())/3600)
    for i in range(d_hours+1):
        get_goes_hour(
            bucket=bucket,
            product=product,
            band=band,
            datadir=datadir.as_posix(),
            time=init_time+td(hours=i),
            mode=mode,
            goes_version=goes_version,
            skip=skip,
            )
