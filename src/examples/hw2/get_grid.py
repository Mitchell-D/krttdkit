import krttdkit as kk
from pathlib import Path
import datetime as dt
from krttdkit import RecipeBook as RB


# Define desired grid domain
grid_center = (26.5, -81) # (lat/lon)
grid_aspect = (2, 2) # (lat/lon)
#grid_center = (27, -82) # (lat/lon)
#grid_aspect = (4, 6) # (lat/lon)
#grid_center = (30, -82) # (lat/lon)
#grid_aspect = (24, 36) # (lat/lon)

# time range determines which files to ingest from the data directory
#time_range = (dt.datetime(year=2021, month=6, day=29, hour=23, minute=0),
#        dt.datetime(year=2021, month=6, day=30, hour=3, minute=30))

#ncdir = Path("/mnt/warehouse/data/abi/aes572_hw2/pyroCb_data/goes17/band13")
#pklpath = Path("data/pkls/pyroCb_tb_b13_2km-goes17.pkl")
#ncdir = Path("/mnt/warehouse/data/abi/aes572_hw2/pyroCb_data/goes17/band02")
#ncdir = Path("/mnt/warehouse/data/abi/aes572_hw2/ian_data/band02")
#ncdir = Path("/mnt/warehouse/data/abi/aes572_hw2/ian_data/band02")
ncdir = Path("/mnt/warehouse/data/abi/aes572_hw2/ian_data/band05")
#pklpath = Path("data/pkls/ian_ref_b02_p5km-goes16.pkl")
#pklpath = Path("data/pkls/ian-look3_ref_b02_p5km-goes16.pkl")
pklpath = Path("data/pkls/ian-look3_ref_b05_1km-goes16.pkl")

buffdir = Path("data/buffer")
label = "ref"
stride=1
convert_Tb = False # Convert to brightness temp
convert_Ref = True # Convert to reflectance

gm = kk.GridManager(buffer_dir=buffdir)
gm.get_abi_grid(
        data_dir=ncdir,
        pkl_path=pklpath,
        #data_dir=ncdir.joinpath(Path("band02")),
        #pkl_path=outdir.joinpath(Path("pyroCb_rad_b02_2km.pkl")),
        #data_dir=ncdir.joinpath(Path("band02")),
        #pkl_path=outdir.joinpath(Path("pyroCb_ref_b02_2km.pkl")),
        grid_center=grid_center,
        grid_aspect=grid_aspect,
        label=label,
        stride=stride,
        convert_Tb=convert_Tb, # Convert to brightness temperature
        convert_Ref=convert_Ref, # Convert to reflectance
        field="Rad",
        #ti=time_range[0],
        #tf=time_range[1],
        _debug=True
        )

gm.clear() # clear memory, run garbage collector.
