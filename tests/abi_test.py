from datetime import datetime
from datetime import timedelta
from pathlib import Path

from krttdkit.operate import enhance as enh
from krttdkit.acquire import GetGOES, GOES_Product
from krttdkit.acquire import abi
from krttdkit.visualize import TextFormat as TF

data_dir = Path("/home/krttd/tools/pybin/krttdkit/tests/buffer/abi")

""" Configure product and desired time ranges """
start_time = datetime(2023, 8, 14, 23)
end_time = datetime(2023, 8, 15, 2)
prod = GOES_Product(
        satellite="16",
        sensor="ABI",
        level="L2",
        scan=None,
        )
#
download = False
download_slice_or_idx = slice(0,3)


"""
If the product has None fields, search for and describe valid products.
Otherwise, search for the product within the provided time bounds
"""
gg = GetGOES()
if not all(prod):
    TF.YELLOW(f"Printing product descriptions for {prod}", bold=True)
    available = gg.list_products(**prod._asdict())
    for p in available:
        gg.describe(p)
    exit(0)
else:
    files = gg.list_range(
            product=prod, init_time=start_time, final_time=end_time)
    all_times = sorted(list(set([ f.stime for f in files])))
    print(TF.BLUE("Found data in time range ") +
          TF.WHITE(f"[{all_times[0]}, {all_times[-1]}]"))
    print(TF.BLUE("With unique labels: "))
    print(sorted(list(set([ f.label for f in files]))))

    if download:
        for p in list(files[download_slice_or_idx]):
            gg.download(p, data_dir)

exit(0)
#paths = list(data_dir.iterdir())

ds = abi.get_dataset(paths[0])
for k in ds.variables.keys():
    print(k, ds.variables[k].shape)
print(ds.dimensions.keys())
