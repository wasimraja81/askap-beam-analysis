"""
holopy.io
=========
Load and introspect the two FITS products from ASKAP holography pipelines.

Classes
-------
HoloCube
    Wraps the frequency cube: (n_beams, 4 Stokes, n_freq, n_dec, n_ra).

HoloTaylor
    Wraps the Taylor-term image: (n_beams, 4 Stokes, 3 Taylor, 1, n_dec, n_ra).

Both classes expose:
  • .data          – numpy array (memory-mapped by default)
  • .header        – astropy FITS header
  • .wcs           – astropy WCS object (spatial + spectral)
  • .summary()     – human-readable text summary
  • .shape_labels  – axis names matching array axes (numpy order)
  • .stokes_labels – list of Stokes parameter names present
  • .beam_index(b) – slice helper: returns data for beam *b* (0-based)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STOKES_MAP = {1: "I", 2: "Q", 3: "U", 4: "V",
              -1: "RR", -2: "LL", -3: "RL", -4: "LR"}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _stokes_labels_from_header(hdr: fits.Header, stokes_axis: int = 4) -> List[str]:
    """
    Return ordered Stokes label list from FITS header.

    Parameters
    ----------
    stokes_axis : int
        1-based FITS axis number corresponding to the Stokes parameter
        (default 4; use 5 for 6-D Taylor cubes where axis 4 is Taylor terms).
    """
    n = hdr.get(f"NAXIS{stokes_axis}", 1)
    crval = hdr.get(f"CRVAL{stokes_axis}", 1)
    cdelt = hdr.get(f"CDELT{stokes_axis}", 1)
    crpix = hdr.get(f"CRPIX{stokes_axis}", 1)
    labels = []
    for i in range(1, n + 1):
        stokes_val = int(round(crval + (i - crpix) * cdelt))
        labels.append(STOKES_MAP.get(stokes_val, f"S{stokes_val}"))
    return labels


def _freq_axis(hdr: fits.Header) -> np.ndarray:
    """Return frequency array in Hz for CTYPE3=FREQ axis."""
    n = hdr.get("NAXIS3", 1)
    crval = hdr.get("CRVAL3", 0.0)
    cdelt = hdr.get("CDELT3", 1.0)
    crpix = hdr.get("CRPIX3", 1)
    return crval + (np.arange(1, n + 1) - crpix) * cdelt   # Hz


# ---------------------------------------------------------------------------
# HoloCube
# ---------------------------------------------------------------------------

@dataclass
class HoloCube:
    """
    Frequency holography cube.

    FITS logical shape: (n_ra, n_dec, n_freq, n_stokes, n_beams)
    NumPy array shape : (n_beams, n_stokes, n_freq, n_dec, n_ra)

    Parameters
    ----------
    path : str or Path
        Path to the cube FITS file.
    memmap : bool
        If True (default) use memory mapping; set False to load fully into RAM.
    """
    path: Path
    memmap: bool = True

    # populated in __post_init__
    data: np.ndarray = field(init=False, repr=False)
    header: fits.Header = field(init=False, repr=False)
    wcs: WCS = field(init=False, repr=False)
    stokes_labels: List[str] = field(init=False)
    freq_hz: np.ndarray = field(init=False, repr=False)
    shape_labels: Tuple[str, ...] = field(init=False)

    def __post_init__(self):
        self.path = Path(self.path)
        with fits.open(self.path, memmap=self.memmap) as hdul:
            self.header = hdul[0].header.copy()
            self.data = hdul[0].data  # (n_beams, n_stokes, n_freq, n_dec, n_ra)
            if not self.memmap:
                self.data = self.data.copy()

        # WCS from the first 4 axes (ra, dec, freq, stokes)
        self.wcs = WCS(self.header, naxis=4)
        self.stokes_labels = _stokes_labels_from_header(self.header)
        self.freq_hz = _freq_axis(self.header)
        self.shape_labels = ("beam", "stokes", "freq", "dec", "ra")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_beams(self) -> int:
        return self.data.shape[0]

    @property
    def n_stokes(self) -> int:
        return self.data.shape[1]

    @property
    def n_freq(self) -> int:
        return self.data.shape[2]

    @property
    def n_dec(self) -> int:
        return self.data.shape[3]

    @property
    def n_ra(self) -> int:
        return self.data.shape[4]

    @property
    def freq_mhz(self) -> np.ndarray:
        """Frequency axis in MHz."""
        return self.freq_hz / 1e6

    @property
    def freq_range_mhz(self) -> Tuple[float, float]:
        """(min_freq_MHz, max_freq_MHz)."""
        return float(self.freq_mhz.min()), float(self.freq_mhz.max())

    @property
    def bandwidth_mhz(self) -> float:
        return float(self.freq_mhz.max() - self.freq_mhz.min())

    @property
    def channel_width_mhz(self) -> float:
        return float(self.header.get("CDELT3", 1.0)) / 1e6

    @property
    def pixel_scale_deg(self) -> Tuple[float, float]:
        """(|CDELT1|, CDELT2) in degrees."""
        return abs(self.header["CDELT1"]), float(self.header["CDELT2"])

    @property
    def pixel_scale_arcmin(self) -> Tuple[float, float]:
        dx, dy = self.pixel_scale_deg
        return dx * 60, dy * 60

    @property
    def date_obs(self) -> Optional[str]:
        return self.header.get("DATE-OBS")

    # ------------------------------------------------------------------
    # Data access helpers
    # ------------------------------------------------------------------

    def beam_data(self, beam: int) -> np.ndarray:
        """
        Return data for a single beam.

        Returns
        -------
        np.ndarray  shape (n_stokes, n_freq, n_dec, n_ra)
        """
        if beam < 0 or beam >= self.n_beams:
            raise IndexError(f"beam index {beam} out of range [0, {self.n_beams})")
        return self.data[beam]

    def stokes_data(self, stokes: str | int) -> np.ndarray:
        """
        Return data for a single Stokes parameter across all beams and
        frequencies.

        Parameters
        ----------
        stokes : str or int
            Label ('I','Q','U','V') or 0-based integer index.

        Returns
        -------
        np.ndarray  shape (n_beams, n_freq, n_dec, n_ra)
        """
        idx = stokes if isinstance(stokes, int) else self.stokes_labels.index(stokes)
        return self.data[:, idx, :, :, :]

    def channel_image(self, beam: int, stokes: str | int, chan: int) -> np.ndarray:
        """
        Single-channel image for a given beam, Stokes, and channel index.

        Returns
        -------
        np.ndarray  shape (n_dec, n_ra)
        """
        sidx = stokes if isinstance(stokes, int) else self.stokes_labels.index(stokes)
        return self.data[beam, sidx, chan, :, :]

    def collapsed_image(
        self,
        beam: int,
        stokes: str | int = "I",
        method: str = "mean",
        freq_range_mhz: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Frequency-collapsed image for one beam/Stokes.

        Parameters
        ----------
        method : {'mean', 'median', 'max', 'rms'}
        freq_range_mhz : optional (lo, hi) MHz sub-band selection

        Returns
        -------
        np.ndarray  shape (n_dec, n_ra)
        """
        sidx = stokes if isinstance(stokes, int) else self.stokes_labels.index(stokes)
        cube = self.data[beam, sidx, :, :, :]  # (n_freq, n_dec, n_ra)

        if freq_range_mhz is not None:
            lo, hi = freq_range_mhz
            mask = (self.freq_mhz >= lo) & (self.freq_mhz <= hi)
            cube = cube[mask]

        if method == "mean":
            return np.nanmean(cube, axis=0)
        if method == "median":
            return np.nanmedian(cube, axis=0)
        if method == "max":
            return np.nanmax(cube, axis=0)
        if method == "rms":
            return np.sqrt(np.nanmean(cube**2, axis=0))
        raise ValueError(f"Unknown method '{method}'. Use mean/median/max/rms.")

    def spectrum(
        self,
        beam: int,
        stokes: str | int = "I",
        region: Optional[Tuple[slice, slice]] = None,
        method: str = "mean",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Spatially averaged spectrum for one beam / Stokes.

        Parameters
        ----------
        region : optional (row_slice, col_slice) to restrict the spatial area
        method : {'mean', 'median', 'max', 'peak'}

        Returns
        -------
        freq_mhz : np.ndarray  (n_freq,)
        flux     : np.ndarray  (n_freq,)
        """
        sidx = stokes if isinstance(stokes, int) else self.stokes_labels.index(stokes)
        cube = self.data[beam, sidx, :, :, :]
        if region is not None:
            rs, cs = region
            cube = cube[:, rs, cs]
        if method == "mean":
            flux = np.nanmean(cube, axis=(-2, -1))
        elif method == "median":
            flux = np.nanmedian(cube, axis=(-2, -1))
        elif method in ("max", "peak"):
            flux = np.nanmax(cube, axis=(-2, -1))
        else:
            raise ValueError(f"Unknown method '{method}'.")
        return self.freq_mhz, flux

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"HoloCube : {self.path.name}",
            "=" * 60,
            f"  Array shape          : {self.data.shape}",
            f"  Axis order           : {self.shape_labels}",
            f"  Beams                : {self.n_beams}",
            f"  Stokes               : {self.stokes_labels}",
            f"  Frequency channels   : {self.n_freq}",
            f"  Freq range           : {self.freq_range_mhz[0]:.2f} – {self.freq_range_mhz[1]:.2f} MHz",
            f"  Bandwidth            : {self.bandwidth_mhz:.1f} MHz",
            f"  Channel width        : {self.channel_width_mhz:.3f} MHz",
            f"  Spatial pixels       : {self.n_ra} × {self.n_dec}",
            f"  Pixel scale          : {self.pixel_scale_arcmin[0]:.3f} × {self.pixel_scale_arcmin[1]:.3f} arcmin",
            f"  Field of view        : {self.n_ra * self.pixel_scale_arcmin[0] / 60:.2f} × {self.n_dec * self.pixel_scale_arcmin[1] / 60:.2f} deg",
            f"  DATE-OBS             : {self.date_obs}",
            f"  SPECSYS              : {self.header.get('SPECSYS', 'N/A')}",
            f"  EQUINOX              : {self.header.get('EQUINOX', 'N/A')}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"HoloCube(beams={self.n_beams}, stokes={self.stokes_labels}, "
            f"freq={self.n_freq}ch, spatial={self.n_ra}×{self.n_dec})"
        )


# ---------------------------------------------------------------------------
# HoloTaylor
# ---------------------------------------------------------------------------

@dataclass
class HoloTaylor:
    """
    Taylor-term holography image.

    FITS logical shape: (n_ra, n_dec, 1, n_taylor, n_stokes, n_beams)
    NumPy array shape : (n_beams, n_stokes, n_taylor, 1, n_dec, n_ra)

    Taylor terms (0-based index → physical meaning):
      0 → T0  : beam amplitude at reference frequency
      1 → α   : spectral index  (d log I / d log ν)
      2 → β   : spectral curvature

    Parameters
    ----------
    path : str or Path
        Path to the Taylor FITS file.
    """
    path: Path
    memmap: bool = True

    data: np.ndarray = field(init=False, repr=False)
    header: fits.Header = field(init=False, repr=False)
    wcs: WCS = field(init=False, repr=False)
    stokes_labels: List[str] = field(init=False)
    ref_freq_hz: float = field(init=False)
    shape_labels: Tuple[str, ...] = field(init=False)
    taylor_labels: Tuple[str, ...] = field(init=False, default=("T0", "alpha", "beta"))

    def __post_init__(self):
        self.path = Path(self.path)
        with fits.open(self.path, memmap=self.memmap) as hdul:
            self.header = hdul[0].header.copy()
            self.data = hdul[0].data   # (n_beams, n_stokes, n_taylor, 1, n_dec, n_ra)
            if not self.memmap:
                self.data = self.data.copy()

        self.wcs = WCS(self.header, naxis=4)
        # Stokes axis is NAXIS5 in the 6-D Taylor cube (NAXIS4 = Taylor terms)
        self.stokes_labels = _stokes_labels_from_header(self.header, stokes_axis=5)
        self.ref_freq_hz = float(self.header.get("CRVAL3", self.header.get("RESTFRQ", 0)))
        self.shape_labels = ("beam", "stokes", "taylor", "freq_dummy", "dec", "ra")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_beams(self) -> int:
        return self.data.shape[0]

    @property
    def n_stokes(self) -> int:
        return self.data.shape[1]

    @property
    def n_taylor(self) -> int:
        return self.data.shape[2]

    @property
    def n_dec(self) -> int:
        return self.data.shape[4]

    @property
    def n_ra(self) -> int:
        return self.data.shape[5]

    @property
    def ref_freq_mhz(self) -> float:
        return self.ref_freq_hz / 1e6

    @property
    def pixel_scale_deg(self) -> Tuple[float, float]:
        return abs(self.header["CDELT1"]), float(self.header["CDELT2"])

    @property
    def pixel_scale_arcmin(self) -> Tuple[float, float]:
        dx, dy = self.pixel_scale_deg
        return dx * 60, dy * 60

    # ------------------------------------------------------------------
    # Data access helpers
    # ------------------------------------------------------------------

    def beam_data(self, beam: int) -> np.ndarray:
        """
        Returns (n_stokes, n_taylor, 1, n_dec, n_ra) for a single beam.
        """
        return self.data[beam]

    def taylor_image(
        self, beam: int, stokes: str | int, term: int | str
    ) -> np.ndarray:
        """
        2-D image for a specific beam, Stokes parameter, and Taylor term.

        Parameters
        ----------
        term : int (0/1/2) or str ('T0', 'alpha', 'beta')

        Returns
        -------
        np.ndarray  shape (n_dec, n_ra)
        """
        sidx = stokes if isinstance(stokes, int) else self.stokes_labels.index(stokes)
        if isinstance(term, str):
            tidx = list(self.taylor_labels).index(term)
        else:
            tidx = term
        return self.data[beam, sidx, tidx, 0, :, :]

    def all_taylor_images(
        self, beam: int, stokes: str | int = "I"
    ) -> dict:
        """
        Return a dict {label: 2D array} for all Taylor terms.
        """
        sidx = stokes if isinstance(stokes, int) else self.stokes_labels.index(stokes)
        return {
            lbl: self.data[beam, sidx, tidx, 0, :, :]
            for tidx, lbl in enumerate(self.taylor_labels)
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"HoloTaylor : {self.path.name}",
            "=" * 60,
            f"  Array shape          : {self.data.shape}",
            f"  Axis order           : {self.shape_labels}",
            f"  Beams                : {self.n_beams}",
            f"  Stokes               : {self.stokes_labels}",
            f"  Taylor terms         : {list(self.taylor_labels)}",
            f"  Reference frequency  : {self.ref_freq_mhz:.3f} MHz",
            f"  Spatial pixels       : {self.n_ra} × {self.n_dec}",
            f"  Pixel scale          : {self.pixel_scale_arcmin[0]:.3f} × {self.pixel_scale_arcmin[1]:.3f} arcmin",
            f"  Field of view        : {self.n_ra * self.pixel_scale_arcmin[0] / 60:.2f} × {self.n_dec * self.pixel_scale_arcmin[1] / 60:.2f} deg",
            f"  DATE-OBS             : {self.header.get('DATE-OBS', 'N/A')}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"HoloTaylor(beams={self.n_beams}, stokes={self.stokes_labels}, "
            f"taylor={list(self.taylor_labels)}, ref={self.ref_freq_mhz:.1f} MHz)"
        )
