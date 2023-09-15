"""
matplotlib-based methods for visualizing a variety of data types.
"""

import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.gridliner import LongitudeLocator, LatitudeLocator
import datetime as dt
import numpy as np
import math as m
import pickle as pkl
import imageio

from pathlib import Path
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolo
import matplotlib.animation as animation
from matplotlib.ticker import LinearLocator, StrMethodFormatter, NullLocator
from matplotlib.transforms import Affine2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    })


plot_spec_default = {
    "title":"",
    "title_size":14,
    "label_size":12,
    "gridline_color":"gray",
    "data_colors":None,
    "fig_size":(16,9),
    "dpi":800,
    "borders":True,
    "border_width":0.5,
    "border_color":"black",
    "cmap":"nipy_spectral",
    "grid":False,
    "legend_font_size":8,
    "legend_ncols":1,
    "x_ticks":None,
    "y_ticks":None,
    "marker":"o",
    "line_style":":",
    # imshow_extent: 4-tuple (left, right, bottom, top) data coordinate values
    "imshow_extent":None,
    # imshow_norm: string or Normalize object for data scaling
    "imshow_norm":None,
    # Float aspect ratio for imshow axes
    "imshow_aspect":None,
    "alpha":.5,
    "xrange":None,
    "yrange":None,
    "xlabel":"",
    "ylabel":"",
    "yrange":None,
    "cb_orient":"vertical",
    "cb_label":"",
    "cb_tick_count":15,
    "cb_tick_size":None,
    "cb_levels":80,
    "cb_size":.6,
    "cb_pad":.05,
    "cb_label_format":"{x:.2f}",
    "line_width":2,
    #"cb_cmap":"CMRmap",
    "cb_cmap":"jet",
    #"xtick_count":12,
    "xtick_size":8,
    #"ytick_count":12,
    }
def plot_classes(class_array:np.ndarray, class_labels:list, colors:list=None,
                 fig_path:Path=None, show:bool=False, plot_spec:dict={}):
    """
    Plots an integer array mapping pixels to a list of class labels

    :@param class_array: 2d integer array such that integer values are the
        indeces of the corresponding class label and color.
    :@param fig_path: Path to generated figure
    :@param class_labels: string labels indexed by class array values.
    :@param colors: List of 3-element [0,1] float arrays for RGB values.
    """
    old_ps = plot_spec_default
    old_ps.update(plot_spec)
    plot_spec = old_ps
    if colors is None:
        colors = [[i/len(class_labels),
                   .5+(len(class_labels)-i)/(2*len(class_labels)),
                   .5+i/(2*len(class_labels))] for i in range(len(class_labels))]
        colors = [ mcolo.hsv_to_rgb(c) for c in colors ]
    assert len(colors)==len(class_labels)
    cmap, norm = matplotlib.colors.from_levels_and_colors(
            list(range(len(colors)+1)), colors)
    im = plt.imshow(class_array, cmap=cmap, norm=norm, interpolation="none")
    handles = [ Patch(label=class_labels[i], color=colors[i])
               for i in range(len(class_labels)) ]
    plt.legend(handles=handles, fontsize=plot_spec.get("fontsize"))
    plt.tick_params(axis="both", which="both", labelbottom=False,
                    labelleft=False, bottom=False, left=False)
    fig = plt.gcf()
    fig.set_size_inches(*plot_spec.get("fig_size"))
    plt.title(plot_spec.get("title"))
    if fig_path:
        print(f"saving figure as {fig_path.as_posix()}")
        plt.savefig(fig_path, bbox_inches="tight", dpi=plot_spec.get("dpi"))
    if show:
        plt.show()

def stats_1d(data_dict:dict, band_labels:list, fig_path:Path=None,
             show:bool=False, class_space:float=.2, bar_sigma:float=1,
             shade_sigma:float=1/3, yscale="linear", plot_spec:dict={}):
    """
    Plot the mean and standard deviation of multiple classes on the same
    X axis. Means and standard deviations for each band in each class must be
    provided as a dictionary with class labels as keys mapping to a dictionary
    with "means" and "stdevs" keys mapping to lists each with N members for
    N bands. Labels for each of the N bands must be provided separately.

    data_dict = {
        "Class 1":{"means":[9,8,7], "stdevs":[1,2,3]}
        "Class 2":{"means":[9,8,7], "stdevs":[1,2,3]}
        "Class 3":{"means":[9,8,7], "stdevs":[1,2,3]}
        }

    :@param yscale: linear by default, but logit may be good for reflectance.
    :@param class_space: directs spacing between class data points/error bars
    :@param bar_sigma: determines the size of error bars in terms of a
        constant multiple on the class' standard deviation.
    :@param shade_sigma: the shaded region of the error bar is typically
        smaller than the bar sigma.
    """
    cat_labels = list(data_dict.keys())
    band_count = len(band_labels)
    assert band_count > 1
    for cat in cat_labels:
        assert len(data_dict[cat]["means"]) == band_count
        assert len(data_dict[cat]["stdevs"]) == band_count

    # Merge provided plot_spec with un-provided default values
    old_ps = plot_spec_default
    old_ps.update(plot_spec)
    plot_spec = old_ps

    fig, ax = plt.subplots()
    transforms = [Affine2D().translate(n, 0.)+ax.transData
                 for n in np.linspace(-.5*class_space, .5*class_space,
                                      num=len(cat_labels))]
    ax.set_yscale(yscale)
    ax.set_ylim(plot_spec.get("yrange"))
    for i in range(len(cat_labels)):
        cat = cat_labels[i]
        ax.errorbar(
                band_labels,
                data_dict[cat]["means"],
                yerr=data_dict[cat]["stdevs"],
                marker=plot_spec.get("marker"),
                label=cat_labels[i],
                linestyle=":",
                transform=transforms[i],
                linewidth=plot_spec.get("line_width"),
                )
        ax.fill_between(
                **{"x":band_labels,
                   "y1":[ m-s*shade_sigma for m,s in zip(
                       data_dict[cat]["means"], data_dict[cat]["stdevs"]) ],
                   "y2":[ m+s*shade_sigma for m,s in zip(
                       data_dict[cat]["means"], data_dict[cat]["stdevs"]) ]},
                alpha=plot_spec.get("alpha"), transform=transforms[i])
        ax.grid(visible=plot_spec.get("grid"))
        ax.set_xlabel(plot_spec.get("xlabel"),
                      fontsize=plot_spec.get("label_size"))
        ax.set_ylabel(plot_spec.get("ylabel"),
                      fontsize=plot_spec.get("label_size"))
        ax.set_title(plot_spec.get("title"), fontsize=plt.get("title_size"))
        ax.legend(fontsize=plot_spec.get("legend_font_size"))

    fig.tight_layout()
    fig.set_size_inches(*plot_spec.get("fig_size"))
    if show:
        plt.show()
    if fig_path:
        fig.savefig(fig_path.as_posix())

def plot_heatmap(heatmap:np.ndarray, fig_path:Path=None, show=True,
                 show_ticks=True, plot_diagonal:bool=False,
                 plot_spec:dict={}):
    """
    Plot an integer heatmap, with [0,0] indexing the lower left corner
    """
    # Merge provided plot_spec with un-provided default values
    old_ps = plot_spec_default
    old_ps.update(plot_spec)
    plot_spec = old_ps

    fig, ax = plt.subplots()

    if plot_diagonal:
        ax.plot((0,heatmap.shape[1]-1), (0,heatmap.shape[0]-1),
                linewidth=plot_spec.get("line_width"))
    im = ax.imshow(
            heatmap,
            cmap=plot_spec.get("cmap"),
            vmax=plot_spec.get("vmax"),
            extent=plot_spec.get("imshow_extent"),
            norm=plot_spec.get("imshow_norm"),
            origin="lower",
            aspect=plot_spec.get("imshow_aspect")
            )
    cbar = fig.colorbar(
            im, orientation=plot_spec.get("cb_orient"),
            label=plot_spec.get("cb_label"), shrink=plot_spec.get("cb_size")
            )
    if not show_ticks:
        plt.tick_params(axis="x", which="both", bottom=False,
                        top=False, labelbottom=False)
        plt.tick_params(axis="y", which="both", bottom=False,
                        top=False, labelbottom=False)
    if plot_spec["imshow_extent"]:
        extent = plot_spec.get("imshow_extent")
        assert len(extent)==4
        plt.xlim(extent[:2])
        plt.ylim(extent[2:])

    #fig.suptitle(plot_spec.get("title"))
    ax.set_title(plot_spec.get("title"))
    ax.set_xlabel(plot_spec.get("xlabel"))
    ax.set_ylabel(plot_spec.get("ylabel"))
    if plot_spec["x_ticks"]:
        ax.set_xticks(plot_spec.get("x_ticks"))
    if plot_spec["y_ticks"]:
        ax.set_xticks(plot_spec.get("y_ticks"))
    if show:
        plt.show()
    if not fig_path is None:
        fig.savefig(fig_path.as_posix(), dpi=plot_spec.get("dpi"),
                    bbox_inches="tight")

def round_to_n(x, n):
    """
    Basic but very useful method to round a number to n significant figures.
    Placed here for rounding float values for labels.
    """
    try:
        return round(x, -int(m.floor(m.log10(x))) + (n - 1))
    except ValueError:
        return 0

def plot_lines(domain, ylines:list, image_path:Path=None,
               labels:list=[], plot_spec={}, show:bool=False):
    """
    Plot a list of 1-d lines that share a domain and codomain.

    :@param domain: 1-d numpy array describing the common domain
    :@param ylines: list of at least 1 1-d array of data values to plot, which
            must be the same size as the domain array.
    :@param image_path: Path to the location to store the figure. If None,
            doesn't store an image.
    :@param labels: list of string labels to include in a legend describing
            each line. If fewer labels than lines are provided, the labels
            will apply to the first of the lines.
    :@param plot_spec: Dictionary of plot options see the geo_plot module
            for plot_spec options, but the defaults are safe.
    :@param show: if True, shows the image in the matplotlib Agg client.
    """
    # Merge provided plot_spec with un-provided default values
    old_ps = plot_spec_default
    old_ps.update(plot_spec)
    plot_spec = old_ps

    # Make sure all codomain arrays equal the domain size
    #if not all((l.size == len(domain) for l in ylines)):
    #    raise ValueError(
    #            f"All codomain arrays must be the same size as the domain.")

    # Plot each
    domain = np.asarray(domain)
    fig, ax = plt.subplots()
    colors = plot_spec.get("colors")
    if colors:
        assert len(ylines)==len(colors)
    for i in range(len(ylines)):
        ax.plot(domain, ylines[i],
                label=labels[i] if len(labels) else "",
                linewidth=plot_spec.get("line_width"),
                color=None if not colors else colors[i])

    ax.set_xlabel(plot_spec.get("xlabel"))
    ax.set_ylabel(plot_spec.get("ylabel"))
    ax.set_title(plot_spec.get("title"))
    ax.set_ylim(plot_spec.get("yrange"))
    ax.set_xlim(plot_spec.get("xrange"))

    if plot_spec.get("xtick_rotation"):
        plt.tick_params(axis="x", **{"labelrotation":plot_spec.get(
            "xtick_rotation")})
    if plot_spec.get("ytick_rotation"):
        plt.tick_params(axis="y", **{"labelrotation":plot_spec.get(
            "ytick_rotation")})

    if len(labels):
        plt.legend(fontsize=plot_spec.get("legend_font_size"),
                   ncol=plot_spec.get("legend_ncols"))
    if show:
        plt.show()
    if not image_path is None:
        print(f"Saving figure to {image_path}")
        fig.savefig(image_path, bbox_inches="tight", dpi=plot_spec.get("dpi"))

def basic_bars(labels, values, xcoords:list=None, err=None, plot_spec:dict={}):
    """
    Make and show a basic bar plot with the provided plot_spec specification.

    :@param labels: list of labels for the x-axis corresponding to each bar.
    :@param values: list of height values for each corresponding bar. There
        must be the same number of values as labels
    :@param xcoords: Optionally provide relative X coordinates for each bar.
        If no xcoords are provided, they will be spaced uniformly.
    :@param err: Optional list of error bar widths in y-coordinate space
    """
    old_ps = plot_spec_default
    old_ps.update(plot_spec)
    plot_spec = old_ps

    print(labels, values)
    assert len(labels)==len(values)
    if xcoords is None:
        xcoords = list(range(len(labels)))
        bar = plt.bar(labels, values, yerr=err,
                      color=plot_spec.get("data_color"))
    else:
        bar = plt.bar(xcoords, values, label=labels, yerr=err,
                      color=plot_spec.get("data_color"))
    plt.title(plot_spec.get("title"))
    plt.xlabel(plot_spec.get("xlabel"))
    plt.ylabel(plot_spec.get("ylabel"))
    plt.show()

def basic_plot(x, y, image_path:Path=None, plot_spec:dict={},
               scatter:bool=False, show:bool=True):
    fig, ax = plt.subplots()
    if scatter:
        ax.scatter(x,y)
    else:
        ax.plot(x,y)
    ax.grid(visible=plot_spec.get("grid"))

    if plot_spec.get("xtick_rotation"):
        plt.tick_params(axis="x", **{"labelrotation":plot_spec.get(
            "xtick_rotation")})
    if plot_spec.get("ytick_rotation"):
        plt.tick_params(axis="y", **{"labelrotation":plot_spec.get(
            "ytick_rotation")})

    ax.set_yscale("log")
    plt.title(plot_spec.get("title"))
    plt.xlabel(plot_spec.get("xlabel"))
    plt.ylabel(plot_spec.get("ylabel"))
    print(f"Saving figure to {image_path}")
    if image_path:
        fig.savefig(image_path)
    if show:
        plt.show()

def generate_raw_image(RGB:np.ndarray, image_path:Path, gif:bool=False,
                       fps:int=5):
    """
    Use imageio to write a raw full-resolution image to image_path
    :param RGB: (H, W, 3) array of RGB values to write as an image, or
            (H, W, T, 3) array of RGB values to write as a gif, if the gif
            attributeis True.
    :param image_path: Path of image file to write
    :param gif: if True, attempts to write as a full-resolution gif
    """
    if not gif:
        imageio.imwrite(image_path.as_posix(), RGB)
        print(f"Generated image at {image_path.as_posix()}")
        return image_path
    RGB = np.moveaxis(RGB, 2, 0)
    '''
    if not RGB.dtype==np.uint8:
        print("converting")
        RGB = (np.moveaxis(RGB, 2, 0)*256).astype(np.uint8)
    '''
    imageio.mimwrite(uri=image_path, ims=RGB, format=".gif", fps=fps)
    print(f"Generated gif at {image_path.as_posix()}")
    return image_path

def geo_rgb_plot(R:np.ndarray, G:np.ndarray, B:np.ndarray, fig_path:Path,
                 xticks:np.ndarray=None, yticks:np.ndarray=None,
                 plot_spec:dict={}, animate:bool=False, extent:list=None):
    """
    Plot RGB values on a lat/lon grid specified by a ndarraay with 2d lat and
    lon coordinate meshes. If animate is False, R/G/B arrays must be 2d
    ndarrays with the same shape, or if animate is True, each of the R/G/B
    arrays must be 3d ndarrays with the same shape with the third axis
    representing a shared time dimension.

    :param data: ndarray. If animate is True, must be 3d with the third
            axes representing time, 2d if animate is False.
    :param lat: 2d ndarray representing latitude values on the data grid
    :param lon: 2d ndarray representing longitude values on the data grid
    :param fig_path: Path to the image generated by this method.
    :param plot_spec: Dictionary of valid plot settings. Options include:
    :param animate: If True and if the DataArray has a "time" dimension,
            animates along the time axis into a GIF, which is saved to
            fig_path.
    :param xticks: List of 2-tuples with the first element containing the int
            value of the pixel to label, and the second value providing the
            label as a string.
    :param yticks: List of 2-tuples with the first element containing the int
            value of the pixel to label, and the second value providing the
            label as a string.
    :param extent: Axes describes axes interval (ie wrt marker locations) and
            aspect ratio as a 4-tuple [left, right, bottom, top]
    """
    fig, ax = plt.subplots(figsize=plot_spec.get("fig_size"))
    #fig, ax = plt.subplots()

    if animate:
        data = np.stack((R,G,B), axis=3)
        if len(data.shape) != 4:
            raise ValueError("Data array must be 4d for animation, " + \
                    "with dimensions ordered like (x,y,time,color)")
        def anim(time):
            im.set_array(data[:,:,time])

        intv = plot_spec.get("anim_delay")
        ani = animation.FuncAnimation(
                fig=fig,
                func=anim,
                frames=data.shape[2],
                interval=intv if intv else 100
                )
        im = ax.imshow(
                data[:, :, 0],
                vmin=plot_spec.get("vmin"),
                vmax=plot_spec.get("vmax"),
                extent=extent # standard extent basis for tick labels
                )

    else:
        data = np.dstack((R,G,B))

        im = ax.imshow(
                data,
                vmin=plot_spec.get("vmin"),
                vmax=plot_spec.get("vmax"),
                extent=extent # standard extent basis for tick labels
                )

    # Set axes ticks if provided
    """
    x_locs, x_labels = (None, None) if xticks is None else list(zip(*xticks))
    y_locs, y_labels = (None, None) if yticks is None else list(zip(*yticks))
    if x_locs and x_labels:
        print(x_locs, x_labels)
        ax.set_xticks(x_locs)
        ax.set_xticklabels(x_labels)
    if y_locs and y_labels:
        ax.set_yticks(y_locs)
        ax.set_yticklabels(y_labels)
    """
    plt.yticks(fontsize=plot_spec.get("ytick_size"))
    plt.xticks(fontsize=plot_spec.get("xtick_size"))

    # markers are 2-tuple pixel coordinates x/y
    if plot_spec.get("markers"):
        marker_char = plot_spec.get("marcher_char")
        marker_char = "x" if not marker_char else marker_char
        marker_color = plot_spec.get("marcher_char")
        marker_color = "red" if not marker_color else marker_color
        for x,y in plot_spec.get("markers"):
            plt.plot(x, y, marker=marker_char, color=marker_color)

    if plot_spec.get("use_ticks") is False:
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())

    dpi = "figure" if not plot_spec.get("dpi") else plot_spec.get("dpi")
    if animate:
        if dpi != "figure":
            fig.set_dpi(dpi)
        ani.save(fig_path.as_posix(), dpi=dpi)
    else:
        plt.savefig(fig_path.as_posix(), dpi=dpi, bbox_inches='tight')

    myfig = plt.gcf()
    size = myfig.get_size_inches()
    print("data:",data.shape)
    print(f"size: {size}; dpi: {dpi}; resolution" + \
            f"{(dpi*size[0], dpi*size[1])}")
    print(f"Generated figure at: {fig_path}")

def geo_scalar_plot(data:np.ndarray, lat:np.ndarray, lon:np.ndarray,
                    fig_path:Path=None, show:bool=True, plot_spec:dict={},
                    animate:bool=False):
    """
    Plot scalar values on a lat/lon grid specified by an xarray Dataset with
    2d coordinates "lat" and "lon". If the Dataset has "plot_spec" attribute
    mapping to a dictionary of supported key/value pairs, they will be used
    to configure the plot. If the Dataset has a "time" dimension and animate
    is True, a gif will be generated at fig_path.

    :param data: ndarray. If animate is True, must be 3d with the third
            axes representing time, 2d if animate is False.
    :param lat: 2d ndarray representing latitude values on the data grid
    :param lon: 2d ndarray representing longitude values on the data grid
    :param fig_path: Path to the image generated by this method.
    :param plot_spec: Dictionary of valid plot settings. Options include:
    :param animate: If True and if the DataArray has a "time" dimension,
            animates along the time axis into a GIF, which is saved to
            fig_path.
    """
    old_ps = plot_spec_default
    old_ps.update(plot_spec)
    plot_spec = old_ps
    fig = plt.figure(figsize=plot_spec.get("fig_size"))
    ax = plt.axes(projection=ccrs.PlateCarree())
    #fig, ax = plt.subplots(1, 1, subplot_kw={"projection":ccrs.PlateCarree()})
    cmap = 'jet' if not plot_spec.get('cb_cmap') else plot_spec.get('cb_cmap')


    if animate:
        if len(data.shape) != 3:
            raise ValueError("Data array must be 3-dimensional for animation")
        def anim(time):
            anim_mesh.set_array(data[:,:,time].flatten())

        intv = plot_spec.get("anim_delay")
        ani = animation.FuncAnimation(
                fig=fig,
                func=anim,
                frames=data.shape[2],
                interval=intv if intv else 100
                )
        anim_mesh = ax.pcolormesh(
                lon,
                lat,
                data[:, :, 0],
                vmin=np.amin(data),
                vmax=np.amax(data),
                #levels=plot_spec.get("cb_levels"),
                #add_colorbar=False,
                cmap=cmap,
                zorder=0,
                )

    if plot_spec.get("borders"):
        linewidth = plot_spec.get("border_width")
        if not linewidth:
            linewidth = 0.5
        b_color=plot_spec.get("border_color")
        ax.coastlines(zorder=1, linewidth=linewidth)
        states_provinces = cf.NaturalEarthFeature(category='cultural',
                name='admin_1_states_provinces_lines',
                scale='10m',
                linewidth=linewidth,
                facecolor='none')
        ax.add_feature(cf.BORDERS, linewidth=linewidth,
                       edgecolor=b_color, zorder=2)
        ax.add_feature(states_provinces, edgecolor=b_color, zorder=3)
        xlocs = plot_spec.get("xtick_count")
        ylocs = plot_spec.get("ytick_count")
        gl = ax.gridlines(
                draw_labels=True,
                linewidth=linewidth,
                xlocs=LongitudeLocator(xlocs) if xlocs else None,
                ylocs=LatitudeLocator(ylocs) if ylocs else None,
                color='gray',
                zorder=4,
                )
        gl.right_labels = False
        gl.top_labels = False
        gl.xlabel_style = {"size":plot_spec.get("xtick_size")}
        gl.ylabel_style = {"size":plot_spec.get("ytick_size")}

    if plot_spec.get("title"):
        ax.set_title(plot_spec.get("title"),
                     fontsize=plot_spec.get("title_size"))
    contour = ax.contourf(
            lon, lat, data,
            levels=plot_spec.get("cb_levels"),
            cmap=cmap,
            zorder=0,
            transform=ccrs.PlateCarree(),
            vmin=np.amin(data),
            vmax=np.amax(data),
            ) if not animate else anim_mesh
    cb_orient = plot_spec.get('cb_orient')
    orientation =  cb_orient if cb_orient else 'vertical'
    fmt = plot_spec.get("cb_label_format")
    pad = 0 if not plot_spec.get("cb_pad") else plot_spec.get("cb_pad")
    shrink = 0 if not plot_spec.get("cb_size") else plot_spec.get("cb_size")
    cbar = fig.colorbar(
            mappable = contour,
            orientation=orientation,
            format=StrMethodFormatter(fmt) if fmt else None,
            pad=pad,
            shrink=shrink,
            ticks=LinearLocator(plot_spec.get("cb_tick_count")),
            )
    if plot_spec.get("cb_tick_size"):
        cbar.ax.tick_params(labelsize=plot_spec.get("cb_tick_size"))
    cbar.set_label(plot_spec.get('cb_label'))
    dpi = "figure" if not plot_spec.get("dpi") else plot_spec.get("dpi")
    if animate:
        ani.save(fig_path.as_posix(), dpi=dpi)
    else:
        if fig_path:
            plt.savefig(fig_path.as_posix(), dpi=dpi, bbox_inches='tight')
            print(f"Generated figure at: {fig_path}")
        if show:
            plt.show()
