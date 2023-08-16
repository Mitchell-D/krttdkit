import numpy as np
from dataclasses import dataclass

def cross_track_area(vza:np.ndarray):
    """
    Calculate the distortion across a scan at constant angular FOV.
    Multiply by the nadir pixel area for all pixel areas.

    :@param vza: Viewing zenith angle in degrees at each pixel
    """
    return np.power(1/np.cos(np.radians(vza)),3)

def get_closest_pixel(latlon:np.ndarray, target:tuple, debug=False):
    """
    Use euclidean distance to determine the closest pixel indeces in latlon
    coordinate space to the provided target. Although geographic coordinates
    are assumed, any 2d monotonic coordinate array should work.
    Haversine formula would be better, but my method is acting odd...

    :@param latlon: 2d monotonic coordinate array.
    :@param target: (lat, lon) or (vertical, horizontal) coordinate to target.
            By array convention, the vertical coordinate is axis 0 of the
            provided latlon array.
    :@return: (y, x) indeces of the closest pixel value.
    """
    lat, lon = latlon[:,:,0], latlon[:,:,1]
    t_lat, t_lon = target

    # Calculate the euclidean distance to each point.
    distance = np.sqrt((lat-t_lat)**2 + (lon-t_lon)**2)
    y_idx, x_idx = tuple(map(lambda x: x[0],
                             np.where(distance==np.amin(distance))))

    if debug:
        print(f"Found pixel ({y_idx}, {x_idx}) closest to {target} at " + \
                f"({lat[y_idx, x_idx]}, {lon[y_idx, x_idx]})")

    # Find pixel closest to target coordinates
    return y_idx, x_idx

def get_geo_range(latlon:np.ndarray, target_latlon:tuple, dx_px:int,dy_px:int,
                  from_center:bool=False, boundary_error:bool=True,
                  debug=False):
    """
    Find indeces closest to the corners of a boundary box described with a
    target location in latlon coordinate space at one corner, and extending
    horizontally/vertically by a pixel width/height of dx_px/dy_py, resp.

    Although the parameters suggest that this method is for lat/lon rasters,
    any 2d monotonic coordinate arrays (ie distance) like (M, N, 2) work.

    :@param latlon: np.ndarray shaped like (ny, nx, 2) of lat/lon values for
            each pixel in the valid domain. See note above.
    :@param target_latlon: 2-Tuple of float values specifying the "anchor"
            point of the boundary box. By default, this is the top left corner
            of the rectangle if dx_px and dy_px are positive. If from_center
            is True, the closest pixel will be the center of the rectangle.
    :@param dx_px: Horizontal pixels from anchor. Positive values correspond to
            an increase in the second axis of the latlon ndarray, which is
            usually rendered as "rightward", or increasing longitude
    :@param dy_px: Vertical pixels from anchor. Positive values correspond to
            an increase in the first axis of the latlon ndarray, which is
            usually rendered as "downward", or decreasing longitude
    :@param from_center: If True, target_latlon describes the center point of
            a rectangle with width dx_px and height dy_px.
    :@param boundary_error: If True, raises a ValueError if the requested pixel
            boundary extends outside of the latlon domain. Otherwise returns
            a boundary at the closest valid pixel. This means if boundary_error
            is False and the grid overlaps a boundary, the returned array will
            not have the requested shape.

    :@return: Nested tuples of coordinates like ((ymin, ymax), (xmin, xmax))
            for the minimum and (non-inclusive) maximum of the requested
            boundary box.
    """
    # Find pixel closest to target coordinates
    target_y, target_x = get_closest_pixel(latlon, target_latlon)

    if from_center:
        if debug:
            print(f"Closest center lat/lon: {latlon[target_y, target_x]}")
        dx_px = abs(dx_px)
        dy_px = abs(dy_px)
        # Returns a non-inclusive index maximum, like range(ymin, ymax)
        ymin = np.floor(target_y-dy_px/2)-1 if dy_px%2 else target_y-dy_px/2
        ymax = np.floor(target_y+dy_px/2)
        xmin = np.floor(target_x-dx_px/2)-1 if dx_px%2 else target_x-dx_px/2
        xmax = np.floor(target_x+dx_px/2)
    else:
        if debug:
            print(f"Closest corner lat/lon: {latlon[target_y, target_x]}")
        ymin, ymax = sorted((target_idx[0][0], target_idx[0][0]+dy_px))
        xmin, xmax = sorted((target_idx[1][0], target_idx[1][0]+dx_px))

    # Check if array is within bounds
    if ymin<0 or ymax>latlon.shape[0]:
        if boundary_error:
            raise ValueError(f"Y-coordinate out of bounds for provided latlon")
        else:
            if ymax>latlon.shape[0]:
                ymax = latlon.shape[0]
            else:
                ymin = 0
    if xmin<0 or xmax>latlon.shape[1]:
        if boundary_error:
            raise ValueError(f"X-coordinate out of bounds for provided latlon")
        else:
            if xmax>latlon.shape[1]:
                xmax = latlon.shape[1]
            else:
                xmin = 0

    # Convert boundaries to integer indeces
    ymin, ymax, xmin, xmax = map(int, (ymin, ymax, xmin, xmax))
    if debug:
        print(f"Found vertical pixel range ({ymin}, {ymax})")
        print(f"Found horizontal pixel range ({xmin}, {xmax})")
    return ((ymin, ymax), (xmin, xmax))


def haversine(i_lat, i_lon, f_lat, f_lon):
    """
    Use the computationally-efficient haversine distance formula to calculate
    the distance between two geographic locations. This function is numpy
    vectorizable, so either lat/lon pair may be a 1d or 2d array.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [i_lon, i_lat, f_lon, f_lat])

    dlon = f_lon - i_lon
    dlat = f_lat - i_lat

    in_sqrt = np.sin(dlat/2.0)**2 + np.cos(i_lat) * \
            np.cos(f_lat) * np.sin(dlon/2.0)**2

    return 2 * np.arcsin(np.sqrt(in_sqrt))

def make_height_map(lat, lon, height):
    """
    Use geographic info arrays to make a surface altitude map, slightly
    colored by the latitude and longitude. Serves as a visual aid.
    """
    r_scale, g_scale, b_scale = (.2, .2, 1)
    blue_height = enhance.linear_gamma_stretch(geo[2])
    red_lat = enhance.linear_gamma_stretch(geo[0])*.2 + blue_height * .4
    green_lon = enhance.linear_gamma_stretch(geo[1])*.2 + blue_height * .5

    return (np.dstack((red_lat, green_lon, blue_height))*256).astype(np.uint8)
