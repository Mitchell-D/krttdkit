import krttdkit as kk
from pathlib import Path
import datetime as dt
from krttdkit import RecipeBook as RB
import numpy as np

pkldir = Path("/home/krttd/uah/22.f/aes572/hw2take2/data/pkls")
figdir = Path("/home/krttd/uah/22.f/aes572/hw2take2/data/figures/pyroCb_Tb")
sounding_path=Path("/home/krttd/uah/22.f/aes572/hw2take2/data/sounding_20210629_12z.txt")

gm = kk.GridManager()
#sat_key = "pyroCb_tb_b13_2km-goes16"
sat_key = "pyroCb-large_tb_b13_2km-goes17"
gm.load_pkl(pkldir.joinpath(sat_key+".pkl"))
label=gm.labels[0]

plot_spec = {
    "title":"Temperatures (K)",
    "title_size":8,
    "gridline_color":"gray",
    "fig_size":None,
    "borders":True,
    "border_width":0.5,
    "border_color":"black",
    "cb_orient":"vertical",
    "cb_label":"BT (K)",
    "cb_tick_count":15,
    "cb_levels":80,
    "cb_size":.6,
    "cb_pad":.05,
    "cb_label_format":"{x:.1f}",
    #"cb_cmap":"CMRmap",
    "cb_cmap":"jet",
    "xtick_count":6,
    "xtick_rotation":45,
    "xtick_size":8,
    "ytick_count":12,
    "ytick_size":8,
    }

""" Plot"""

infra_da = gm.subgrids["infra"]["am"].data["infra"]

mins = [ np.amin(infra_da.isel(time=i)).item()
        for i in range(len(infra_da["time"]))]
plot_spec["title"] = "Minimum Tb over entire domain"
plot_spec["ylabel"] = "Brightness Temp (K)"
kk.geo_plot.basic_plot(
        infra_da["time"].data,
        mins,
        image_path=figdir.joinpath(Path(sat_key+"-Tb-min-any.png")),
        plot_spec=plot_spec
        )

am = gm.subgrids["infra"]["am"]
idx = am.get_closest_latlon(*(51.5, -121))
print(am.data["lat"].data[idx], am.data["lon"].data[idx])
print("mean BC sfc T (51.5, -121):",np.average(am.data["infra"].data[idx]))
idx = am.get_closest_latlon(*(50.7, -127.4))
print("mean YZT sfc T (50.7, -127.4):",np.average(am.data["infra"].data[idx]))

exit(0)

plot_spec["title"] = "ABI Band 13 Tb (K)"
plot_spec["ylabel"] = "Brightness Temps (K)"
kk.geo_plot.geo_scalar_plot(
    data=infra_da.data,
    lat=infra_da["lat"],
    lon=infra_da["lon"],
    fig_path=figdir.joinpath(Path(sat_key+".gif")),
    plot_spec=plot_spec,
    animate=True
    )


# Get a full-res version of the array normalized on [0,1]
darr = kk.GridManager.norm_to_unit(
        np.stack((infra_da.data,infra_da.data, infra_da.data), axis=3))
# Generate a "raw" unprojected image at full resolution
kk.geo_plot.generate_raw_image(
        RGB=darr,
        image_path=figdir.joinpath(Path(sat_key+"-raw.gif")),
        gif=True,
        fps=15
        )

"""
Use linear regression to build a monotonic mapping from temperture to altitude
based on a sounding from nearby the pyroCb
"""

# Read the altitude and temperature profile data from the sounding
with open(sounding_path, "r") as sfp:
    lines = sfp.readlines()
height = []
temps = []
for l in lines:
    if l[0]=="#":
        continue
    tmpd = tuple(l.split(" "))
    height.append(float(tmpd[1]))
    temps.append(float(tmpd[2]))

# Only use the sounding data up to the 100mb level; the stratospheric
# temperature inversion was previously throwing off the interpolation
height = np.asarray(height[:60]) # m
temps = np.asarray(temps[:60])+273.15 # K

# Get linear regression of temps wrt height, and use it to convert
# brightness temperatures to altitudes. Since this is linear and the
# infrared surface has warmer values than the sounding surface, floor
# the data at zero interpolated altitude.
a, b = tuple(np.polyfit(x=height, y=temps, deg=1))
print("Coeffs: ",a,b)
height = ((infra_da.data-b)/a)
height[height<0] = 0

plot_spec["title"] = "Estimated height (m)"
plot_spec["cb_label"] = "Height (m)"
kk.geo_plot.geo_scalar_plot(
    data=height,
    lat=infra_da["lat"],
    lon=infra_da["lon"],
    fig_path=figdir.joinpath(Path(sat_key+"-height.gif")),
    plot_spec=plot_spec,
    animate=True
    )
"""
Minimum brightness temperature
"""
min_tb = np.amin(infra_da.data)
idx_x, idx_y, idx_t = tuple(np.where(infra_da.data==min_tb))
#infra_da = gm.subgrids[label]["am"].data[label].isel(time=idx_t)
temps = infra_da.isel(x=idx_x, y=idx_y).data.squeeze()
times = infra_da.coords["time"].data

plot_spec["title"] = "Temperature at minimum Tb point"
plot_spec["ylabel"] = "Brightness Temp (K)"
kk.geo_plot.basic_plot(times, temps,
        image_path=figdir.joinpath(Path(sat_key+"-Tb-min-temps.png")),
        plot_spec=plot_spec
        )

min_time = kk.ABIManager.dt64_to_datetime(times[idx_t][0])
min_lat = infra_da.coords['lat'].isel(x=idx_x, y=idx_y)[0][0]
min_lon = infra_da.coords['lon'].isel(x=idx_x, y=idx_y)[0][0]
min_str = f"({min_lat:.2f}, {min_lon:.2f}) at {min_time.strftime('%H:%M')}"

print(f"minumum found: {min_str}")

plot_spec["title"] = f"Minumum Tb: {min_str}"
plot_spec["cb_label"] = "Brightness Temp (K)"
kk.geo_plot.geo_scalar_plot(
    data=infra_da.isel(time=idx_t).data[:,:,0],
    lat=infra_da["lat"].data,
    lon=infra_da["lon"].data,
    fig_path=figdir.joinpath(Path(sat_key+"-min-Tb.png")),
    plot_spec=plot_spec,
    animate=False)

"""
Set an upper-bound threshold, restricting the data to temperatures <= 227 K
"""
upper_bound = 227
gm.subgrids[label]["am"].restrict_data(
                bounds=[None, upper_bound],
                replace_val=[None, upper_bound])
infra_da = gm.subgrids[label]["am"].data[label]

# Plot a scalar projection with matplotlib
plot_spec["title"] = f"Brightness Temperatures > {upper_bound} K"
kk.geo_plot.geo_scalar_plot(
        data=infra_da.data,
        lat=infra_da["lat"],
        lon=infra_da["lon"],
        fig_path=figdir.joinpath(Path(sat_key+f"-ub{upper_bound}K.gif")),
        plot_spec=plot_spec,
        animate=True
        )

"""
Count the number of pixels that exceed the threshold
"""

# Get a count wrt time of pixels exceeding the anvil threshold
counts = np.asarray([ 0 for i in range(infra_da.shape[2]) ])
for i in range(len(counts)):
    counts[i] = np.count_nonzero(infra_da[:,:,i] < upper_bound)

times = gm.coords["time"].values
plot_spec["title"] = f"Pixels exceeding {upper_bound}K Brightness Temp"
kk.geo_plot.basic_plot(
        times, counts,
        image_path=figdir.joinpath(Path(sat_key+"-anvil.png")),
        plot_spec=plot_spec)

"""
Calculate and plot the anvil expansion rate, using the configured threshold
as the upper-bound temperature for pixels considered part of the anvil.
"""
t = []
a = []
for i in range(len(counts)-1):
    # Area in m^2
    da = (counts[i+1]-counts[i])*4*1000 # km^2
    ti = kk.ABIManager.dt64_to_datetime(times[i])
    tf = kk.ABIManager.dt64_to_datetime(times[i+1])
    dt = ((tf-ti)/2)
    t.append(dt.total_seconds()*i)
    a.append(da/dt.total_seconds())

plot_spec["title"] = f"{upper_bound}K-Threshold Expansion Rate (m^2/s)"
plot_spec["xlabel"] = "Time (seconds)"
plot_spec["ylabel"] = "Anvil Expansion (m^2/s)"
kk.geo_plot.basic_plot( t, a,
        image_path=figdir.joinpath(Path(sat_key+"-expansion.png")),
        plot_spec=plot_spec
        )
