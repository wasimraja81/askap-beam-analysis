"""
holopy.beam
===========
Per-beam geometry, amplitude, and shape metrics.

BeamAnalyser wraps a HoloCube or HoloTaylor and provides:
  • peak_amplitude()      – peak value and pixel location per beam
  • centroid()            – intensity-weighted centroid offset (pixels / arcmin)
  • fwhm_estimate()       – FWHM along RA and Dec via half-power contour fitting
  • sidelobe_level()      – first sidelobe peak relative to main beam
  • beam_stats()          – aggregate statistics dict for one beam/Stokes
  • compare_beams()       – table of key metrics across all beams
  • radial_profile()      – azimuthally averaged radial profile
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import center_of_mass, label, maximum_filter
from scipy.optimize import curve_fit

from .io import HoloCube, HoloTaylor

# ---------------------------------------------------------------------------
# Gaussian utilities
# ---------------------------------------------------------------------------

def _gaussian2d(xy, amp, x0, y0, sigma_x, sigma_y, theta):
    """Flattened 2-D elliptical Gaussian for curve_fit."""
    x, y = xy
    ct, st = np.cos(theta), np.sin(theta)
    xr = ct * (x - x0) + st * (y - y0)
    yr = -st * (x - x0) + ct * (y - y0)
    return amp * np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))


# ---------------------------------------------------------------------------
# BeamAnalyser
# ---------------------------------------------------------------------------

class BeamAnalyser:
    """
    Compute per-beam shape and quality metrics.

    Parameters
    ----------
    source : HoloCube or HoloTaylor
        Loaded holography data.
    stokes : str, optional
        Which Stokes parameter to work with (default 'I').
    freq_method : str, optional
        How to collapse the frequency axis for HoloCube inputs
        ('mean', 'median', 'max', 'rms').  Ignored for HoloTaylor.
    freq_range_mhz : tuple, optional
        (lo, hi) sub-band selection when collapsing HoloCube.
    """

    def __init__(
        self,
        source: Union[HoloCube, HoloTaylor],
        stokes: str = "I",
        freq_method: str = "mean",
        freq_range_mhz: Optional[Tuple[float, float]] = None,
    ):
        self.source = source
        self.stokes = stokes
        self.freq_method = freq_method
        self.freq_range_mhz = freq_range_mhz
        self._images: Dict[int, np.ndarray] = {}  # cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_image(self, beam: int) -> np.ndarray:
        """Retrieve (or build) the 2-D analysis image for a beam."""
        if beam not in self._images:
            if isinstance(self.source, HoloCube):
                img = self.source.collapsed_image(
                    beam, self.stokes, self.freq_method, self.freq_range_mhz
                )
            else:  # HoloTaylor – use T0 term
                img = self.source.taylor_image(beam, self.stokes, 0)
            self._images[beam] = img
        return self._images[beam]

    def _pixel_scale_arcmin(self) -> Tuple[float, float]:
        return self.source.pixel_scale_arcmin

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def peak_amplitude(self, beam: int) -> Dict:
        """
        Find the peak pixel value and its location.

        Returns
        -------
        dict with keys:
          peak_value, peak_row, peak_col,
          offset_ra_arcmin, offset_dec_arcmin
          (offsets measured from image centre)
        """
        img = self._get_image(beam)
        flat_idx = np.nanargmax(img)
        row, col = np.unravel_index(flat_idx, img.shape)
        peak_val = float(img[row, col])
        cy, cx = np.array(img.shape) / 2.0
        dx_pix, dy_pix = col - cx, row - cy
        dra, ddec = self._pixel_scale_arcmin()
        return {
            "peak_value": peak_val,
            "peak_row": int(row),
            "peak_col": int(col),
            "offset_ra_arcmin": dx_pix * dra,
            "offset_dec_arcmin": dy_pix * ddec,
        }

    def centroid(self, beam: int, threshold_frac: float = 0.1) -> Dict:
        """
        Intensity-weighted centroid above *threshold_frac* × peak.

        Returns
        -------
        dict with keys:
          centroid_row, centroid_col,
          offset_ra_arcmin, offset_dec_arcmin
        """
        img = self._get_image(beam)
        peak = float(np.nanmax(img))
        mask = img >= threshold_frac * peak
        masked = np.where(mask, img, 0.0)
        cy, cx = center_of_mass(masked)
        iy, ix = np.array(img.shape) / 2.0
        dra, ddec = self._pixel_scale_arcmin()
        return {
            "centroid_row": float(cy),
            "centroid_col": float(cx),
            "offset_ra_arcmin": (cx - ix) * dra,
            "offset_dec_arcmin": (cy - iy) * ddec,
        }

    def fwhm_estimate(self, beam: int) -> Dict:
        """
        Estimate FWHM along RA and Dec axes by fitting a 1-D Gaussian
        through the peak pixel.

        Returns
        -------
        dict with keys:
          fwhm_ra_arcmin, fwhm_dec_arcmin, fwhm_mean_arcmin
        """
        img = self._get_image(beam)
        pk = self.peak_amplitude(beam)
        pr, pc = pk["peak_row"], pk["peak_col"]
        dra, ddec = self._pixel_scale_arcmin()

        def _fwhm_1d(profile):
            profile = np.asarray(profile, dtype=float)
            half = profile.max() / 2.0
            idx = np.where(profile >= half)[0]
            if len(idx) < 2:
                return np.nan
            return float(idx[-1] - idx[0] + 1)

        fwhm_ra_pix = _fwhm_1d(img[pr, :])
        fwhm_dec_pix = _fwhm_1d(img[:, pc])

        fwhm_ra = fwhm_ra_pix * dra
        fwhm_dec = fwhm_dec_pix * ddec
        return {
            "fwhm_ra_pix": fwhm_ra_pix,
            "fwhm_dec_pix": fwhm_dec_pix,
            "fwhm_ra_arcmin": fwhm_ra,
            "fwhm_dec_arcmin": fwhm_dec,
            "fwhm_mean_arcmin": float(np.nanmean([fwhm_ra, fwhm_dec])),
        }

    def gaussian_fit_2d(self, beam: int, box_half: int = 20) -> Dict:
        """
        Fit a 2-D elliptical Gaussian around the peak of the beam image.

        Parameters
        ----------
        box_half : int
            Half-width of the fitting box in pixels.

        Returns
        -------
        dict with keys:
          amplitude, x0_pix, y0_pix, sigma_x_pix, sigma_y_pix,
          pa_deg (position angle), fwhm_x_arcmin, fwhm_y_arcmin,
          fit_ok (bool)
        """
        img = self._get_image(beam)
        pk = self.peak_amplitude(beam)
        pr, pc = pk["peak_row"], pk["peak_col"]
        dra, ddec = self._pixel_scale_arcmin()

        r0 = max(0, pr - box_half)
        r1 = min(img.shape[0], pr + box_half + 1)
        c0 = max(0, pc - box_half)
        c1 = min(img.shape[1], pc + box_half + 1)
        sub = img[r0:r1, c0:c1]

        rows, cols = np.mgrid[r0:r1, c0:c1]
        xy = (cols.ravel().astype(float), rows.ravel().astype(float))
        z = sub.ravel().astype(float)

        half_size = box_half / 2.0
        p0 = [pk["peak_value"], float(pc), float(pr), half_size, half_size, 0.0]
        bounds_lo = [0, c0, r0, 1, 1, -np.pi / 2]
        bounds_hi = [np.inf, c1, r1, box_half * 2, box_half * 2, np.pi / 2]

        try:
            popt, _ = curve_fit(
                _gaussian2d, xy, z, p0=p0,
                bounds=(bounds_lo, bounds_hi), maxfev=5000
            )
            amp, x0, y0, sx, sy, theta = popt
            sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
            return {
                "amplitude": float(amp),
                "x0_pix": float(x0),
                "y0_pix": float(y0),
                "sigma_x_pix": float(sx),
                "sigma_y_pix": float(sy),
                "pa_deg": float(np.degrees(theta)),
                "fwhm_x_arcmin": float(sx * sigma_to_fwhm * dra),
                "fwhm_y_arcmin": float(sy * sigma_to_fwhm * ddec),
                "fit_ok": True,
            }
        except Exception as exc:
            return {"fit_ok": False, "error": str(exc)}

    def sidelobe_level(
        self, beam: int, min_distance_frac: float = 0.3
    ) -> Dict:
        """
        Find the highest sidelobe peak outside the main-beam region.

        Parameters
        ----------
        min_distance_frac : float
            Pixels within this fraction of image half-width of the peak
            are masked as the main beam region.

        Returns
        -------
        dict with keys:
          main_peak, sidelobe_peak, sidelobe_ratio_dB,
          sidelobe_row, sidelobe_col
        """
        img = self._get_image(beam)
        pk = self.peak_amplitude(beam)
        pr, pc = pk["peak_row"], pk["peak_col"]
        main_peak = pk["peak_value"]

        # mask main beam
        half = min(img.shape) * min_distance_frac
        rr, cc = np.ogrid[:img.shape[0], :img.shape[1]]
        main_mask = ((rr - pr) ** 2 + (cc - pc) ** 2) <= half ** 2
        side_img = np.where(main_mask, np.nan, img)

        flat_idx = np.nanargmax(side_img)
        sr, sc = np.unravel_index(flat_idx, img.shape)
        sidelobe_peak = float(img[sr, sc])
        ratio = sidelobe_peak / main_peak if main_peak > 0 else np.nan
        ratio_db = 20 * np.log10(ratio) if ratio > 0 else np.nan

        return {
            "main_peak": float(main_peak),
            "sidelobe_peak": sidelobe_peak,
            "sidelobe_ratio": ratio,
            "sidelobe_ratio_dB": ratio_db,
            "sidelobe_row": int(sr),
            "sidelobe_col": int(sc),
        }

    def radial_profile(
        self, beam: int, n_bins: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Azimuthally averaged radial profile centred on the peak.

        Returns
        -------
        radius_arcmin : np.ndarray
        mean_amplitude : np.ndarray
        """
        img = self._get_image(beam)
        pk = self.peak_amplitude(beam)
        pr, pc = pk["peak_row"], pk["peak_col"]
        dra, ddec = self._pixel_scale_arcmin()

        rows, cols = np.mgrid[:img.shape[0], :img.shape[1]]
        r_pix = np.sqrt(((rows - pr) * ddec) ** 2 + ((cols - pc) * dra) ** 2)

        r_max = r_pix[~np.isnan(img)].max()
        bins = np.linspace(0, r_max, n_bins + 1)
        radius_arcmin = 0.5 * (bins[:-1] + bins[1:])
        mean_amp = np.full(n_bins, np.nan)

        for i in range(n_bins):
            mask = (r_pix >= bins[i]) & (r_pix < bins[i + 1])
            vals = img[mask & ~np.isnan(img)]
            if len(vals) > 0:
                mean_amp[i] = float(np.nanmean(vals))

        return radius_arcmin, mean_amp

    def beam_stats(self, beam: int) -> Dict:
        """
        Convenience method returning a combined stats dict for one beam.
        """
        img = self._get_image(beam)
        stats = {
            "beam": beam,
            "stokes": self.stokes,
            "data_min": float(np.nanmin(img)),
            "data_max": float(np.nanmax(img)),
            "data_mean": float(np.nanmean(img)),
            "data_rms": float(np.sqrt(np.nanmean(img**2))),
            "nan_fraction": float(np.isnan(img).sum() / img.size),
        }
        stats.update(self.peak_amplitude(beam))
        stats.update(self.fwhm_estimate(beam))
        stats.update(self.sidelobe_level(beam))
        return stats

    def compare_beams(self) -> List[Dict]:
        """
        Run beam_stats() for every beam; returns list of dicts (one per beam).
        Suitable for building a pandas DataFrame:

            import pandas as pd
            df = pd.DataFrame(analyser.compare_beams())
        """
        return [self.beam_stats(b) for b in range(self.source.n_beams)]
