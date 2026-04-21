"""
holopy.frequency
================
Spectral analysis across the holography band.

FrequencyAnalyser wraps a HoloCube and provides:
  • channel_rms()          – per-channel noise/rms across spatial pixels
  • channel_flags()        – auto-detect bad (outlier) channels
  • band_average_image()   – wideband image with optional channel masking
  • spectral_index_map()   – log-log power-law α per pixel (per beam)
  • bandwidth_smearing()   – theoretical smearing estimate vs. offset
  • channel_coherence()    – pearson-r between adjacent channels
  • freq_stats_table()     – per-beam spectral summary dict
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr

from .io import HoloCube


class FrequencyAnalyser:
    """
    Spectral behaviour analysis for a holography cube.

    Parameters
    ----------
    cube : HoloCube
    stokes : str
        Stokes parameter to analyse (default 'I').
    """

    def __init__(self, cube: HoloCube, stokes: str = "I"):
        self.cube = cube
        self.stokes = stokes
        self._sidx = cube.stokes_labels.index(stokes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cube_data(self, beam: int) -> np.ndarray:
        """Return (n_freq, n_dec, n_ra) for one beam/Stokes."""
        return self.cube.data[beam, self._sidx]

    # ------------------------------------------------------------------
    # Channel statistics
    # ------------------------------------------------------------------

    def channel_rms(self, beam: int) -> np.ndarray:
        """
        Spatial RMS value at each frequency channel.

        Returns
        -------
        np.ndarray  shape (n_freq,)
        """
        c = self._cube_data(beam)
        return np.sqrt(np.nanmean(c**2, axis=(-2, -1)))

    def channel_mean(self, beam: int) -> np.ndarray:
        """Mean value per frequency channel."""
        return np.nanmean(self._cube_data(beam), axis=(-2, -1))

    def channel_peak(self, beam: int) -> np.ndarray:
        """Peak (max) value per frequency channel."""
        return np.nanmax(self._cube_data(beam), axis=(-2, -1))

    def channel_flags(
        self,
        beam: int,
        sigma_thresh: float = 4.0,
        metric: str = "rms",
    ) -> np.ndarray:
        """
        Flag outlier channels based on their spatial RMS or mean.

        Parameters
        ----------
        sigma_thresh : float
            Flag channels deviating more than this many σ from the median.
        metric : {'rms', 'mean', 'peak'}

        Returns
        -------
        flags : np.ndarray of bool, shape (n_freq,)
            True = flagged (bad channel).
        """
        if metric == "rms":
            vals = self.channel_rms(beam)
        elif metric == "mean":
            vals = self.channel_mean(beam)
        elif metric == "peak":
            vals = self.channel_peak(beam)
        else:
            raise ValueError(f"Unknown metric '{metric}'.")

        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
        sigma = 1.4826 * mad  # robust sigma estimate
        flags = np.abs(vals - med) > sigma_thresh * sigma
        return flags

    def flagged_fraction(self, beam: int, sigma_thresh: float = 4.0) -> float:
        """Fraction of channels flagged as bad."""
        flags = self.channel_flags(beam, sigma_thresh)
        return float(flags.sum()) / len(flags)

    # ------------------------------------------------------------------
    # Band-averaged image
    # ------------------------------------------------------------------

    def band_average_image(
        self,
        beam: int,
        exclude_flagged: bool = True,
        sigma_thresh: float = 4.0,
    ) -> np.ndarray:
        """
        Mean image over the full band (optionally excluding flagged channels).

        Returns
        -------
        np.ndarray  shape (n_dec, n_ra)
        """
        c = self._cube_data(beam)
        if exclude_flagged:
            flags = self.channel_flags(beam, sigma_thresh)
            c = c[~flags]
        return np.nanmean(c, axis=0)

    # ------------------------------------------------------------------
    # Spectral index
    # ------------------------------------------------------------------

    def spectral_index_map(
        self,
        beam: int,
        freq_lo_mhz: Optional[float] = None,
        freq_hi_mhz: Optional[float] = None,
        min_snr: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-pixel spectral index α by log-log linear fit:  I ∝ ν^α.

        Pixels below *min_snr* × median RMS are masked (set to NaN).

        Parameters
        ----------
        freq_lo_mhz, freq_hi_mhz : float, optional
            Sub-band limits. Defaults to full band.

        Returns
        -------
        alpha_map : np.ndarray  (n_dec, n_ra)
        alpha_err  : np.ndarray  (n_dec, n_ra)  – 1-σ uncertainty
        """
        freq = self.cube.freq_mhz
        mask_f = np.ones(len(freq), dtype=bool)
        if freq_lo_mhz is not None:
            mask_f &= freq >= freq_lo_mhz
        if freq_hi_mhz is not None:
            mask_f &= freq <= freq_hi_mhz

        c = self._cube_data(beam)[mask_f]  # (n_sel_freq, n_dec, n_ra)
        log_nu = np.log(freq[mask_f] / freq[mask_f].mean())
        n_freq = c.shape[0]

        # noise floor mask on wideband image
        wb = np.nanmean(c, axis=0)
        rms_floor = np.sqrt(np.nanmean(wb**2)) / min_snr

        log_I = np.log(np.abs(c))  # (n_freq, n_dec, n_ra)
        log_I[:, wb < rms_floor] = np.nan

        # Vectorised 1-D polyfit (degree 1) across freq axis
        A = np.column_stack([log_nu, np.ones(n_freq)])  # (n_freq, 2)
        valid_mask = np.all(np.isfinite(log_I), axis=0)

        ny, nx = wb.shape
        alpha_map = np.full((ny, nx), np.nan)
        alpha_err = np.full((ny, nx), np.nan)

        flat_log_I = log_I.reshape(n_freq, -1)  # (n_freq, ny*nx)
        ATA = A.T @ A
        ATb = A.T @ flat_log_I  # (2, ny*nx)
        try:
            sol = np.linalg.solve(ATA, ATb)  # (2, ny*nx)
            alpha_flat = sol[0].reshape(ny, nx)
            alpha_map = np.where(valid_mask, alpha_flat, np.nan)
        except np.linalg.LinAlgError:
            pass

        return alpha_map, alpha_err

    # ------------------------------------------------------------------
    # Channel coherence
    # ------------------------------------------------------------------

    def channel_coherence(
        self, beam: int, lag: int = 1
    ) -> np.ndarray:
        """
        Pearson-r between channel i and channel i+lag (spatial correlation).
        Useful for detecting RFI or calibration artefacts.

        Returns
        -------
        np.ndarray  shape (n_freq - lag,)
        """
        c = self._cube_data(beam)
        n = c.shape[0]
        r = np.full(n - lag, np.nan)
        for i in range(n - lag):
            a = c[i].ravel()
            b = c[i + lag].ravel()
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() > 3:
                r[i] = pearsonr(a[valid], b[valid])[0]
        return r

    # ------------------------------------------------------------------
    # Bandwidth smearing
    # ------------------------------------------------------------------

    def bandwidth_smearing(
        self,
        offsets_arcmin: Optional[np.ndarray] = None,
        chan_width_mhz: Optional[float] = None,
        ref_freq_mhz: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Theoretical bandwidth-smearing amplitude factor R(θ) for a top-hat
        channel (Bridle & Schwab 1999):

            R(θ) ≈ sinc(Δν/ν₀ · θ/θ_synth)  [approximation for extended sources]

        Here we use the simpler integral form for a point source at offset θ:

            R = sin(πΔν·θ/ν₀·) / (πΔν·θ/ν₀)  [sinc in natural units]

        Parameters
        ----------
        offsets_arcmin : array-like
            Angular offsets from phase centre in arcmin.
        chan_width_mhz : float, optional
            Channel bandwidth (defaults to cube channel width).
        ref_freq_mhz : float, optional
            Reference frequency (defaults to cube centre frequency).

        Returns
        -------
        offsets_arcmin : np.ndarray
        smearing_factor : np.ndarray  (1.0 = no smearing)
        """
        if offsets_arcmin is None:
            offsets_arcmin = np.linspace(0, 200, 200)
        offsets_arcmin = np.asarray(offsets_arcmin, dtype=float)

        dnu = (chan_width_mhz or self.cube.channel_width_mhz) * 1e6  # Hz
        nu0 = (ref_freq_mhz or float(np.mean(self.cube.freq_mhz))) * 1e6  # Hz

        # Convert arcmin to radians for generic smearing formula
        theta_rad = np.radians(offsets_arcmin / 60.0)

        # For a single channel: smearing ~ sinc(Δν/ν × θ / θ_resolution)
        # Simplified scalar form (no spatial resolution term):
        x = np.pi * dnu / nu0 * theta_rad
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(x == 0, 1.0, np.sin(x) / x)

        return offsets_arcmin, R

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def freq_stats_table(self, beam: int, sigma_thresh: float = 4.0) -> Dict:
        """
        Return a summary dict of spectral properties for one beam.
        """
        rms = self.channel_rms(beam)
        flags = self.channel_flags(beam, sigma_thresh)
        return {
            "beam": beam,
            "stokes": self.stokes,
            "n_channels": int(self.cube.n_freq),
            "n_flagged": int(flags.sum()),
            "flagged_fraction": float(flags.sum()) / self.cube.n_freq,
            "freq_min_mhz": float(self.cube.freq_mhz.min()),
            "freq_max_mhz": float(self.cube.freq_mhz.max()),
            "bandwidth_mhz": float(self.cube.bandwidth_mhz),
            "channel_width_mhz": float(self.cube.channel_width_mhz),
            "rms_median": float(np.nanmedian(rms)),
            "rms_min": float(np.nanmin(rms)),
            "rms_max": float(np.nanmax(rms)),
            "rms_std": float(np.nanstd(rms)),
        }
