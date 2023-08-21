"""
Module to provide a common library for calculating latitude, longitudes
and viewing zenith angles for geostationary satellites
"""
import numpy as np
import warnings

# Without ignoring, numpy throws lots of runtime warnings due to asymptotic
# trig values off the edge of the Earth.
warnings.filterwarnings('ignore', category=RuntimeWarning)

class GeosGeom:
    """Determine latitude, longitudes and viewing zenith angles from
    geostationary satellite viewing angles"""

    def __init__(self, lon_proj_origin: float, e_w_scan_angles: np.array,
                 n_s_scan_angles: np.array, satellite_alt: float,
                 r_eq: float, r_pol: float, sweep):
        """
        Calculate longitudes, latitudes and viewing zenith angles from
        satellite perspective and scan angles
        :param lon_proj_origin: Satellite subpoint longitude (degrees)
        :param e_w_scan_angles: E/W 2d satellite viewing angles (radians)
        :param n_s_scan_angles: N/S 2d satellite viewing angles (radians)
        :param satellite_alt: Nominal altitude of satellite above MSL (m)
        :param r_eq: Radius of Earth at the equator (m)
        :param r_pol: Radius of Earth at the poles (m)
        :param sweep: Sweep angle axis of the satellite
        """
        self.R_e = 6371000
        self.longitude_of_projection_origin = lon_proj_origin
        self.perspective_point_height = satellite_alt
        self.r_eq = r_eq
        self.r_pol = r_pol
        self.e_w_scan_angles = e_w_scan_angles
        self.n_s_scan_angles = n_s_scan_angles
        self.sweep_angle_axis = sweep
        self._lons, self._lats = self.gen_earth_locations()
        self._vzas = None

    def __repr__(self):
        """ Returns a string reporting lat/lon sizes and ranges """
        return f"GeosGeom:\n" + \
                f"\tLats rng: ({np.amin(np.nan_to_num(self._lats, 99999))},"+\
                f" {np.amax(np.nan_to_num(self._lats, -99999))})  " + \
                f"SIZE: {self._lats.size}  " + \
                f"NaNs: {np.count_nonzero(np.isnan(self._lats))}\n" + \
                f"\tLons rng: ({np.amin(np.nan_to_num(self._lons, 99999))},"+\
                f" {np.amax(np.nan_to_num(self._lons, -99999))})  " + \
                f"SIZE: {self._lons.size}  " + \
                f"NaNs: {np.count_nonzero(np.isnan(self._lons))}"

    def get_closest_latlon(self, lat, lon):
        """ returns the index of the pixel closest to the provided lat/lon """
        masked_lats = np.nan_to_num(self._lats, 999999)
        masked_lons = np.nan_to_num(self._lons, 999999)

        # Get an array of angular distance to the desired lat/lon
        lat_diff = masked_lats-lat
        lon_diff = masked_lons-lon
        total_diff = np.sqrt(lat_diff**2+lon_diff**2)
        min_idx = tuple([ int(c[0]) for c in
            np.where(total_diff == np.amin(total_diff)) ])
        return min_idx

    def get_subgrid_indeces(self, lat_range:tuple=None, lon_range:tuple=None,
                            _debug:bool=False):
        """
        Returns indeces of lat/lon values closest to the provided latitude
        or longitude range

        Both latitude and longitude ranges must be specified, or else the
        terminal indeces of the full lat/lon array will be returned.

        :param lat_range: (min, max) latitude in degrees.
                Defaults to full size.
        :param lon_range: (min, max) longitude in degrees.
                Defaults to full size.
        :param _debug: If True, prints information about the subgrid found.

        :return: latitude and longitude index ranges closest to the desired
                values, using the following tuple format. These indeces are
                reported like a 2d array, so lat_index_0 is actually the
                largest latitude value since 2d arrays count from the "top".
                ( (lat_index_0, lat_index_f),
                    (lon_index_0, lon_index_f) )
        """
        valid_lats = not lat_range is None and len(lat_range)==2 \
                and not next(l is None for l in lat_range)
        valid_lons = not lon_range is None and len(lon_range)==2 \
                and not next(l is None for l in lon_range)
        if not valid_lats or not valid_lons:
            if _debug:
                print("latitude and longitude ranges not specified;" + \
                        " defaulting to full array size.")
            minlat, maxlon = self._lats.shape
            return ((0,minlat), (0, maxlon))

        lr_latlon = [None, None]
        ul_latlon = [None, None]
        lr_latlon[0], ul_latlon[0] = sorted(lat_range)
        ul_latlon[1], lr_latlon[1] = sorted(lon_range)
        lr_latlon, ul_latlon = zip(lat_range, lon_range[::-1])
        ul_index = self.get_closest_latlon(*ul_latlon)
        lr_index = self.get_closest_latlon(*lr_latlon)

        if _debug:
            print("requested lat range: ",lat_range)
            print("requested lon range: ",lon_range)
            print("upper left pixel: ",ul_latlon, ul_index,
                  (self._lats[ul_index], self._lons[ul_index]))
            print("lower right pixel: ",lr_latlon, lr_index,
                  (self._lats[lr_index], self._lons[lr_index]))

        return tuple(zip(ul_index, lr_index))

    @property
    def lats(self) -> np.array:
        """ :return: 1d numpy array of latitude values in degrees """
        return self._lats

    @property
    def lons(self) -> np.array:
        """ :return: 1d numpy array of longitude values in degeres """
        return self._lons

    @property
    def vzas(self) -> np.array:
        """
        Viewing zenith angle getter
        """
        if self._vzas is None:
            self._vzas = self.gen_viewing_zenith_angles()
        return self._vzas

    def gen_earth_locations(self, use_pyproj:bool=False)->(np.array, np.array):
        """
        Calculate latitude, longitude (degrees) values from GOES ABI fixed
        grid (radians).

        See GOES-R PUG Volume 4 Section 7.1.2.8 Navigation of Image Data
        https://www.goes-r.gov/users/docs/PUG-GRB-vol4.pdf,
        or just the wikipedia page on geodetic coordinates.

        :returns: longitudes, latitudes arrays
        """
        # GOES-17 values, GOES-16 values
        # lon_origin  # -137, -75
        # r_eq  # 6378137, 6378137.0
        # r_pol  # 6356752.31414, 6356752.31414
        # h  # 42164160, 42164160.0
        # lambda_0  # -2.3911010752322315, -1.3089969389957472

        lon_origin = self.longitude_of_projection_origin
        r_eq = self.r_eq
        r_pol = self.r_pol
        h = self.perspective_point_height + r_eq
        lambda_0 = (lon_origin * np.pi) / 180.0,
        sweep = self.sweep_angle_axis,

        # Geodedic coordinate transformation
        if use_pyproj:
            print("pyproj-based deprojection isn't supported yet.")
            #proj = Proj(proj='geos',h=str(), lon_0=str(lon_origin),
            #            sweep=sweep, R=self.R_e)

        """
        # Hacky way to get projection for a subset of the grid if the
        # computer keeps running out of memory during trig operations
        nscut = int(self.n_s_scan_angles.shape[0]/2)
        ewcut = int(self.e_w_scan_angles.shape[1]/2)
        sinlatr = np.sin(self.n_s_scan_angles[:nscut,:ewcut])
        sinlonr = np.sin(self.e_w_scan_angles[:nscut,:ewcut])
        coslatr = np.cos(self.n_s_scan_angles[:nscut,:ewcut])
        coslonr = np.cos(self.e_w_scan_angles[:nscut,:ewcut])
        """
        sinlatr = np.sin(self.n_s_scan_angles)
        sinlonr = np.sin(self.e_w_scan_angles)
        coslatr = np.cos(self.n_s_scan_angles)
        coslonr = np.cos(self.e_w_scan_angles)

        # Both GOES sats
        # N/S sa: (.151844, -.151844)
        # E/W sa: (-.151844, .151844)

        r_eq2 = r_eq * r_eq
        r_pol2 = r_pol * r_pol
        a_var = (np.square(sinlonr) +
                    (np.square(coslonr) *
                        (np.square(coslatr) +
                            ((r_eq2 / r_pol2) * np.square(sinlatr)))
                     )
                 )

        b_var = -2.0 * h * coslonr * coslatr
        c_var = h ** 2.0 - r_eq2
        r_s = (-b_var - np.sqrt(
            (b_var ** 2) - (4.0 * a_var * c_var))) / (2.0 * a_var)
        s_x = r_s * coslonr * coslatr
        s_y = -r_s * sinlonr
        s_z = r_s * coslonr * sinlatr
        h_sx = h - s_x
        lats = np.degrees(np.arctan((r_eq2 / r_pol2) * s_z /
                                    np.sqrt((h_sx * h_sx) + (s_y * s_y))))
        lons = np.degrees(lambda_0 - np.arctan(s_y / h_sx))
        #print(f"a_var {a_var}\n", f"b_var {b_var}\n", f"c_var {c_var}\n",
        #      f"r_s {r_s}\n", f"s_x {s_x}\n", f"s_y {s_y}\n", f"s_z {s_z}")

        """
        print("lat min/max:",np.amin(np.nan_to_num(lats, 99999)),
              np.amax(np.nan_to_num(lats,-99999)))
        print("lon min/max:",np.amin(np.nan_to_num(lons, 99999)),
                np.amax(np.nan_to_num(lons,-99999)))
        print("lats size/nancount:",lats.size,
              np.count_nonzero(np.isnan(lats)))
        print("lons size/nancount:",lons.size,
              np.count_nonzero(np.isnan(lons)))
        """
        return lons.astype("float64"), lats.astype("float64")

    def gen_viewing_zenith_angles(self) -> np.array:
        """
        Generate viewing zenith angles for each ABI fixed grid point

        Viewing zenith angle (vza) (simplified version - law of sines)
        https://www.ngs.noaa.gov/CORS/Articles/SolerEisemannJSE.pdf

        :returns: viewing zenith angle array
        """
        r_eq = self.r_eq
        h = self.perspective_point_height + r_eq
        theta_s = np.sqrt(self.e_w_scan_angles ** 2. + self.n_s_scan_angles **
                          2.)
        vzas = np.degrees(np.arcsin((h / r_eq) * np.sin(theta_s)))
        return vzas

