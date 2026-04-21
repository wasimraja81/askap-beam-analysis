"""
holopy.stokes
=============
Stokes parameter analysis and polarisation metrics.

StokesAnalyser wraps a HoloCube and provides:
  • stokes_images()        – frequency-collapsed I, Q, U, V for a beam
  • linear_polarisation()  – P = sqrt(Q² + U²) image
  • polarisation_angle()   – χ = 0.5 arctan(U/Q) image (degrees)
  • fractional_pol()       – p = P/I, l = LP/I, c = V/I images
  • leakage_map()          – off-axis Stokes leakage X/I for X ∈ {Q,U,V}
  • stokes_spectrum()      – I/Q/U/V spectra for a beam at a pixel / region
  • polarisation_stats()   – summary statistics dict
  • rotation_measure_estimate() – coarse RM from Q/U vs λ² slope
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np

from .io import HoloCube

SPEED_OF_LIGHT = 2.997924e8  # m/s


class StokesAnalyser:
    """
    Derive polarisation products from a holography frequency cube.

    Parameters
    ----------
    cube : HoloCube
        Loaded holography frequency cube (must contain IQUV).
    freq_method : str
        Collapse method for spatial images ('mean', 'median', 'rms', 'max').
    freq_range_mhz : tuple, optional
        Sub-band (lo, hi) MHz for image collapsing.
    """

    def __init__(
        self,
        cube: HoloCube,
        freq_method: str = "mean",
        freq_range_mhz: Optional[Tuple[float, float]] = None,
    ):
        self.cube = cube
        self.freq_method = freq_method
        self.freq_range_mhz = freq_range_mhz
        self._check_stokes()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_stokes(self):
        for s in ("I", "Q", "U", "V"):
            if s not in self.cube.stokes_labels:
                raise ValueError(
                    f"Stokes {s} not found. Available: {self.cube.stokes_labels}"
                )

    def _img(self, beam: int, stokes: str) -> np.ndarray:
        return self.cube.collapsed_image(
            beam, stokes, self.freq_method, self.freq_range_mhz
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stokes_images(self, beam: int) -> Dict[str, np.ndarray]:
        """
        Return dict of frequency-collapsed 2-D images for I, Q, U, V.
        """
        return {s: self._img(beam, s) for s in ("I", "Q", "U", "V")}

    def linear_polarisation(self, beam: int) -> np.ndarray:
        """
        Linear polarised intensity P = sqrt(Q² + U²).
        """
        Q = self._img(beam, "Q")
        U = self._img(beam, "U")
        return np.sqrt(Q**2 + U**2)

    def polarisation_angle(self, beam: int) -> np.ndarray:
        """
        Electric vector position angle χ = 0.5 * arctan(U/Q) in degrees.
        """
        Q = self._img(beam, "Q")
        U = self._img(beam, "U")
        return np.degrees(0.5 * np.arctan2(U, Q))

    def fractional_polarisation(self, beam: int) -> Dict[str, np.ndarray]:
        """
        Fractional polarisation maps, all normalised by Stokes I.

        Returns
        -------
        dict with keys:
          'LP'  – linear fractional  P/I
          'CP'  – circular fractional V/I
          'TP'  – total fractional   sqrt(Q²+U²+V²)/I
          'Q/I', 'U/I', 'V/I'
        """
        I = self._img(beam, "I")
        Q = self._img(beam, "Q")
        U = self._img(beam, "U")
        V = self._img(beam, "V")
        with np.errstate(invalid="ignore", divide="ignore"):
            LP = np.sqrt(Q**2 + U**2) / I
            CP = V / I
            TP = np.sqrt(Q**2 + U**2 + V**2) / I
            QI = Q / I
            UI = U / I
            VI = V / I
        return {"LP": LP, "CP": CP, "TP": TP, "Q/I": QI, "U/I": UI, "V/I": VI}

    def leakage_map(
        self, beam: int, stokes_numerator: str = "Q"
    ) -> np.ndarray:
        """
        Beam leakage χ/I for χ ∈ {'Q','U','V'}, expressed as a fraction.
        Values near the phase centre should be small for a well-calibrated beam.
        """
        if stokes_numerator not in ("Q", "U", "V"):
            raise ValueError("stokes_numerator must be 'Q', 'U', or 'V'.")
        I = self._img(beam, "I")
        X = self._img(beam, stokes_numerator)
        with np.errstate(invalid="ignore", divide="ignore"):
            leakage = X / I
        return leakage

    def stokes_spectrum(
        self,
        beam: int,
        region: Optional[Tuple[slice, slice]] = None,
        method: str = "mean",
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Spectra for all four Stokes parameters.

        Returns
        -------
        dict mapping Stokes label → (freq_mhz, flux) tuple.
        """
        return {
            s: self.cube.spectrum(beam, s, region, method)
            for s in ("I", "Q", "U", "V")
        }

    def rotation_measure_estimate(
        self,
        beam: int,
        region: Optional[Tuple[slice, slice]] = None,
    ) -> Dict:
        """
        Coarse rotation measure estimate via linear fit of polarisation
        angle vs λ² across the band.

        χ(λ²) = χ₀ + RM · λ²

        Returns
        -------
        dict with keys: RM_rad_m2, chi0_deg, fit_residual_deg
        """
        spectra = self.stokes_spectrum(beam, region)
        freq_hz = spectra["I"][0] * 1e6
        lam2 = (SPEED_OF_LIGHT / freq_hz) ** 2

        Q_spec = spectra["Q"][1]
        U_spec = spectra["U"][1]
        chi = np.degrees(0.5 * np.arctan2(U_spec, Q_spec))  # deg

        # Filter NaN / zero-Q channels
        valid = np.isfinite(chi) & np.isfinite(lam2) & (np.abs(Q_spec) > 0)
        if valid.sum() < 3:
            return {"RM_rad_m2": np.nan, "chi0_deg": np.nan, "fit_residual_deg": np.nan}

        # Convert chi to radians for fitting
        chi_rad = np.radians(chi[valid])
        l2 = lam2[valid]
        coeffs = np.polyfit(l2, np.unwrap(chi_rad), 1)
        chi0 = float(np.degrees(coeffs[1]))
        RM = float(coeffs[0])  # rad/m²
        residuals = np.radians(chi[valid]) - np.polyval(coeffs, l2)
        rms_res = float(np.degrees(np.sqrt(np.mean(residuals**2))))
        return {"RM_rad_m2": RM, "chi0_deg": chi0, "fit_residual_deg": rms_res}

    def polarisation_stats(self, beam: int) -> Dict:
        """
        Summary statistics dict for a single beam.
        """
        I = self._img(beam, "I")
        fp = self.fractional_polarisation(beam)
        lp = self.linear_polarisation(beam)

        def _safe_stats(arr, name):
            valid = arr[np.isfinite(arr)]
            return {
                f"{name}_mean": float(np.nanmean(valid)) if len(valid) else np.nan,
                f"{name}_median": float(np.nanmedian(valid)) if len(valid) else np.nan,
                f"{name}_std": float(np.nanstd(valid)) if len(valid) else np.nan,
                f"{name}_max": float(np.nanmax(valid)) if len(valid) else np.nan,
            }

        stats = {"beam": beam}
        stats.update(_safe_stats(I, "I"))
        stats.update(_safe_stats(fp["LP"], "LP"))
        stats.update(_safe_stats(fp["CP"], "CP"))
        stats.update(_safe_stats(fp["Q/I"], "Q_over_I"))
        stats.update(_safe_stats(fp["U/I"], "U_over_I"))
        stats.update(_safe_stats(fp["V/I"], "V_over_I"))
        return stats
