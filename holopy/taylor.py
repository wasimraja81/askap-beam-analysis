"""
holopy.taylor
=============
Analysis of the Taylor-term holography product (T0, α, β).

TaylorAnalyser wraps a HoloTaylor and provides:
  • amplitude_map()        – T0 image per beam/Stokes
  • spectral_index_map()   – α image per beam/Stokes (Taylor term 1)
  • curvature_map()        – β image per beam/Stokes (Taylor term 2)
  • predicted_beam()       – reconstruct beam at an arbitrary frequency
  • term_stats()           – per-beam statistics for all Taylor terms
  • compare_terms()        – cross-beam comparison table
  • cross_beam_alpha()     – α variation across the beam footprint
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .io import HoloTaylor

TAYLOR_LABELS = ("T0", "alpha", "beta")


class TaylorAnalyser:
    """
    Inspect and derive products from the Taylor-term holography image.

    Parameters
    ----------
    taylor : HoloTaylor
        Loaded Taylor FITS product.
    """

    def __init__(self, taylor: HoloTaylor):
        self.taylor = taylor

    # ------------------------------------------------------------------
    # Basic extractors
    # ------------------------------------------------------------------

    def amplitude_map(self, beam: int, stokes: str = "I") -> np.ndarray:
        """
        Taylor term T0 – beam amplitude at the reference frequency.

        Returns
        -------
        np.ndarray  (n_dec, n_ra)
        """
        return self.taylor.taylor_image(beam, stokes, "T0")

    def spectral_index_map(self, beam: int, stokes: str = "I") -> np.ndarray:
        """
        Taylor term α – first-order spectral index.

        α = d log I / d log ν

        Returns
        -------
        np.ndarray  (n_dec, n_ra)
        """
        if self.taylor.n_taylor < 2:
            raise RuntimeError("Taylor cube has fewer than 2 terms; α unavailable.")
        return self.taylor.taylor_image(beam, stokes, "alpha")

    def curvature_map(self, beam: int, stokes: str = "I") -> np.ndarray:
        """
        Taylor term β – second-order spectral curvature.

        Returns
        -------
        np.ndarray  (n_dec, n_ra)
        """
        if self.taylor.n_taylor < 3:
            raise RuntimeError("Taylor cube has fewer than 3 terms; β unavailable.")
        return self.taylor.taylor_image(beam, stokes, "beta")

    # ------------------------------------------------------------------
    # Beam reconstruction
    # ------------------------------------------------------------------

    def predicted_beam(
        self,
        beam: int,
        freq_mhz: float,
        stokes: str = "I",
    ) -> np.ndarray:
        """
        Reconstruct the beam pattern at *freq_mhz* using the Taylor expansion:

            B(ν) ≈ T0 · (1 + α·x + 0.5·β·x²)

        where x = ln(ν / ν_ref).

        Parameters
        ----------
        freq_mhz : float
            Target frequency in MHz.

        Returns
        -------
        np.ndarray  (n_dec, n_ra)
        """
        t0 = self.amplitude_map(beam, stokes)
        ref = self.taylor.ref_freq_mhz
        x = np.log(freq_mhz / ref)

        result = t0.copy()
        if self.taylor.n_taylor >= 2:
            alpha = self.spectral_index_map(beam, stokes)
            result = result + t0 * alpha * x
        if self.taylor.n_taylor >= 3:
            beta = self.curvature_map(beam, stokes)
            result = result + 0.5 * t0 * beta * x**2
        return result

    def freq_axis_for_reconstruction(
        self, n_points: int = 100
    ) -> Tuple[np.ndarray, float]:
        """
        Convenience: return a sensible frequency grid for reconstruction.

        Returns
        -------
        freq_mhz : np.ndarray
        ref_freq_mhz : float
        """
        ref = self.taylor.ref_freq_mhz
        freq_mhz = np.linspace(ref * 0.85, ref * 1.15, n_points)
        return freq_mhz, ref

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def term_stats(self, beam: int, stokes: str = "I") -> Dict:
        """
        Summary statistics for all Taylor terms for one beam.

        Returns
        -------
        dict with keys: beam, stokes, T0_*, alpha_*, beta_*
        """
        terms = self.taylor.all_taylor_images(beam, stokes)
        stats: Dict = {"beam": beam, "stokes": stokes,
                       "ref_freq_mhz": self.taylor.ref_freq_mhz}

        for label, img in terms.items():
            valid = img[np.isfinite(img)]
            if len(valid) == 0:
                continue
            stats[f"{label}_min"] = float(valid.min())
            stats[f"{label}_max"] = float(valid.max())
            stats[f"{label}_mean"] = float(valid.mean())
            stats[f"{label}_median"] = float(np.median(valid))
            stats[f"{label}_std"] = float(valid.std())
            stats[f"{label}_peak"] = float(np.nanmax(img))

        return stats

    def compare_terms(self, stokes: str = "I") -> List[Dict]:
        """
        Run term_stats() for all beams; returns list of dicts.

        Example usage::

            import pandas as pd
            df = pd.DataFrame(ta.compare_terms())
        """
        return [self.term_stats(b, stokes) for b in range(self.taylor.n_beams)]

    # ------------------------------------------------------------------
    # Cross-beam comparisons
    # ------------------------------------------------------------------

    def cross_beam_alpha(
        self,
        stokes: str = "I",
        pixel: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Spectral index α at a fixed spatial pixel across all beams.

        Parameters
        ----------
        pixel : (row, col), optional
            Spatial location.  Defaults to the image centre.

        Returns
        -------
        np.ndarray  shape (n_beams,)
        """
        if self.taylor.n_taylor < 2:
            raise RuntimeError("No α term available.")
        if pixel is None:
            pixel = (self.taylor.n_dec // 2, self.taylor.n_ra // 2)
        r, c = pixel
        alpha_vals = np.array([
            float(self.spectral_index_map(b, stokes)[r, c])
            for b in range(self.taylor.n_beams)
        ])
        return alpha_vals

    def alpha_gradient(
        self, beam: int, stokes: str = "I"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Spatial gradient of the spectral index map (|∂α/∂x|, |∂α/∂y|).

        Returns
        -------
        grad_ra, grad_dec : np.ndarray (n_dec, n_ra) each
          Units: α / pixel  (multiply by pixel_scale to get α / arcmin)
        """
        alpha = self.spectral_index_map(beam, stokes)
        grad_dec, grad_ra = np.gradient(alpha)
        return grad_ra, grad_dec

    def amplitude_uniformity(
        self,
        stokes: str = "I",
        pixel: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """
        Statistics of T0 amplitude across the beam footprint.

        Parameters
        ----------
        pixel : (row, col), optional
            If provided, sample only this pixel across all beams.
            If None, use the peak pixel from each beam independently.
        """
        T0_vals = []
        for b in range(self.taylor.n_beams):
            img = self.amplitude_map(b, stokes)
            if pixel is not None:
                T0_vals.append(float(img[pixel]))
            else:
                T0_vals.append(float(np.nanmax(img)))
        T0_arr = np.array(T0_vals)
        return {
            "stokes": stokes,
            "n_beams": self.taylor.n_beams,
            "T0_min": float(T0_arr.min()),
            "T0_max": float(T0_arr.max()),
            "T0_mean": float(T0_arr.mean()),
            "T0_std": float(T0_arr.std()),
            "T0_relative_spread_pct": 100 * T0_arr.std() / T0_arr.mean()
            if T0_arr.mean() != 0 else np.nan,
            "T0_per_beam": T0_arr.tolist(),
        }
