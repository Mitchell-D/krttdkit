
import numpy as np

class FireDetector:
    """
    Executes fire detection algorithm from (Flasse and Ceccato, 1996)
    on arbitrary data grids of brightness temperatures in LWIR and SWIR bands
    and calibrated reflectances (bi-directional reflectance factor) in NIR

    Band : Wavelengths  : Unit
    ------------------------------------------------
    LWIR : 10.3-11.3 um : Brightness temperature (K)
    SWIR : 3.55-3.93 um : Brightness temperature (K)
    NIR  : 0.73-1.00 um : Calibrated reflectance
    """
    def __init__(self):
        #
        self.swir_thresh_lbound = 311 # K
        self.lwir_thresh_lbound = 0 # K
        # Pixels are valid if the LWIR and SWIR are different by 8K;
        # ~3.7um is near the Wein's law peak for fire temperatures.
        self.swir_lwir_diff_lbound = 8 # K
        # Reflectance threshold above which a pixel is just considered
        # a highly reflective surface
        self.nir_ref_ubound = .2
        self._candidates = None
        self._LWIR = None
        self._SWIR = None
        self._NIR = None

    @property
    def candidates(self):
        return self._candidates

    def _get_candidates(self, LWIR, SWIR, NIR):
        """
        """
        self._LWIR, self._SWIR, self._NIR = LWIR, SWIR, NIR
        conditions = [
            SWIR > self.swir_thresh_lbound,
            LWIR > self.lwir_thresh_lbound,
            SWIR-LWIR > self.swir_lwir_diff_lbound,
            NIR < self.nir_ref_ubound
            ]
        return np.where(conditions[0] &
                        conditions[1] &
                        conditions[2] &
                        conditions[3])

    def test_candidates(self, candidates, LWIR, SWIR, NIR):
        fires = np.full_like(LWIR, False, dtype=bool)
        for px in zip(*candidates):
            fires[px[0], px[1]] = self._test_candidate(
                    (px[0],px[1]), LWIR, SWIR, NIR)
        return np.where(fires)

    def _test_candidate(self, px_coords:tuple, LWIR, SWIR, NIR):
        y,x = px_coords
        for i in range(1,7):
            yrange, xrange = ((y-i, y+i), (x-i, x+i))
            if not (yrange[0]>1 and xrange[0]>1 and \
                    yrange[1]<LWIR.shape[0]-1 and xrange[1]<LWIR.shape[1]-1):
                return False
            tmp_lwir = np.copy(LWIR)[yrange[0]:yrange[1], xrange[0]:xrange[1]]
            tmp_swir = np.copy(SWIR)[yrange[0]:yrange[1], xrange[0]:xrange[1]]
            tmp_nir = np.copy(NIR)[yrange[0]:yrange[1], xrange[0]:xrange[1]]
            new_cand = self._get_candidates(tmp_lwir,tmp_swir,tmp_nir)
            # If there aren't  enough background pixels
            if new_cand[0].size/tmp_lwir.size>.75:
                continue
            bg_mask = np.full_like(tmp_lwir, False, dtype=bool)
            bg_mask[new_cand] = True
            rgb = np.dstack([tmp_lwir, tmp_swir, tmp_nir])
            bg_px = rgb[np.where(np.logical_not(bg_mask))]

            bg_nir_mean = np.average(bg_px[:,1])
            bg_nir_std = np.std(bg_px[:,1])
            bg_diff_mean = np.average(bg_px[:,1]-bg_px[:,0])
            bg_diff_std = np.std(bg_px[:,1]-bg_px[:,0])

            condition1 = SWIR[y,x]-(bg_nir_mean+2*bg_nir_std) > 3
            condition2 = SWIR[y,x]-LWIR[y,x] > (bg_diff_mean+2*bg_diff_std)
            if condition1 and condition2:
                return True
        return False


    def get_candidates(self, LWIR:np.ndarray, SWIR:np.ndarray, NIR:np.ndarray,
                   debug=False):
        """
        Returns True if candidate pixels are found, and stores any candidate
        pixels to the self.candidates property, replacing any previous
        candidates (possibly under different threshold conditions).
        """
        self._candidates = self._get_candidates(LWIR, SWIR, NIR)
        if not len(self._candidates):
            if debug: print("No fires detected.")
            return self
        if debug: print(f"{len(self._candidates[0])} candidate pixels found")
        return self
