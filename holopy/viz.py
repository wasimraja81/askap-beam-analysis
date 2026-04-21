"""
holopy.viz
==========
Publication-ready plotting helpers for holography data.

All functions return (fig, axes) so callers can further customise.

Functions
---------
plot_beam_image()        – single 2-D beam pattern image
plot_all_beams()         – grid of all beam images (one panel per beam)
plot_stokes_panel()      – I / Q / U / V side-by-side for one beam
plot_spectrum()          – frequency spectrum with optional flagging
plot_radial_profile()    – azimuthally averaged radial profile
plot_taylor_terms()      – T0 / α / β side-by-side for one beam
plot_beam_comparison()   – scatter/bar of a metric across all beams
plot_freq_rms()          – per-channel RMS diagnostics
plot_smearing_curve()    – bandwidth smearing factor vs. angular offset
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CMAP_AMPLITUDE = "inferno"
DEFAULT_CMAP_SIGNED = "RdBu_r"
DEFAULT_CMAP_ALPHA = "coolwarm"


def _add_colorbar(ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    return plt.colorbar(im, cax=cax)


def _pixel_extent(data_shape, pixel_scale_arcmin):
    """Return imshow extent in arcmin centred at (0, 0)."""
    ny, nx = data_shape
    dx, dy = pixel_scale_arcmin
    half_x = nx * dx / 2.0
    half_y = ny * dy / 2.0
    return [-half_x, half_x, -half_y, half_y]


# ---------------------------------------------------------------------------
# Single beam image
# ---------------------------------------------------------------------------

def plot_beam_image(
    image: np.ndarray,
    pixel_scale_arcmin: Tuple[float, float] = (1.0, 1.0),
    title: str = "",
    cmap: str = DEFAULT_CMAP_AMPLITUDE,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    log_scale: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Display a single 2-D beam image.

    Parameters
    ----------
    image            : 2-D numpy array
    pixel_scale_arcmin : (dx, dy) in arcmin
    log_scale        : if True, use logarithmic colour scale (absolute values)
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    extent = _pixel_extent(image.shape, pixel_scale_arcmin)
    disp = np.abs(image) if log_scale else image
    norm = mcolors.LogNorm(vmin=vmin or disp[disp > 0].min(), vmax=vmax or disp.max()) \
        if log_scale else mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        disp, origin="lower", extent=extent,
        cmap=cmap, norm=norm, interpolation="nearest"
    )
    _add_colorbar(ax, im)
    ax.set_xlabel("ΔRA (arcmin)")
    ax.set_ylabel("ΔDec (arcmin)")
    ax.set_title(title)
    if fig is not None:
        fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# All-beams grid
# ---------------------------------------------------------------------------

def plot_all_beams(
    images: List[np.ndarray],
    pixel_scale_arcmin: Tuple[float, float] = (1.0, 1.0),
    ncols: int = 6,
    cmap: str = DEFAULT_CMAP_AMPLITUDE,
    suptitle: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    log_scale: bool = False,
    beam_labels: Optional[List[str]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot a grid of beam images.

    Parameters
    ----------
    images : list of 2-D arrays, one per beam (length = n_beams)
    ncols  : columns in the grid
    """
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3 * ncols, 3.2 * nrows), squeeze=False
    )

    extent = _pixel_extent(images[0].shape, pixel_scale_arcmin)

    # Common colour scale
    all_vals = np.concatenate([img[np.isfinite(img)].ravel() for img in images])
    _vmin = vmin if vmin is not None else float(all_vals.min())
    _vmax = vmax if vmax is not None else float(all_vals.max())

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx >= n:
            ax.axis("off")
            continue
        img = images[idx]
        disp = np.abs(img) if log_scale else img
        norm = mcolors.LogNorm(
            vmin=max(_vmin, 1e-10), vmax=_vmax
        ) if log_scale else mcolors.Normalize(vmin=_vmin, vmax=_vmax)
        ax.imshow(
            disp, origin="lower", extent=extent,
            cmap=cmap, norm=norm, interpolation="nearest"
        )
        label = beam_labels[idx] if beam_labels else f"Beam {idx}"
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Stokes panel
# ---------------------------------------------------------------------------

def plot_stokes_panel(
    stokes_images: Dict[str, np.ndarray],
    pixel_scale_arcmin: Tuple[float, float] = (1.0, 1.0),
    beam: int = 0,
    suptitle: str = "",
    log_scale_I: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot I, Q, U, V images side by side.

    Parameters
    ----------
    stokes_images : dict from StokesAnalyser.stokes_images()
    log_scale_I   : use log scale for Stokes I
    """
    labels = [s for s in ("I", "Q", "U", "V") if s in stokes_images]
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4))
    if len(labels) == 1:
        axes = [axes]

    for ax, s in zip(axes, labels):
        img = stokes_images[s]
        cmap = DEFAULT_CMAP_AMPLITUDE if s == "I" else DEFAULT_CMAP_SIGNED
        log = log_scale_I and s == "I"
        plot_beam_image(img, pixel_scale_arcmin, title=f"Stokes {s}",
                        cmap=cmap, log_scale=log, ax=ax)

    fig.suptitle(suptitle or f"Stokes IQUV – Beam {beam}")
    fig.tight_layout()
    return fig, np.asarray(axes)


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------

def plot_spectrum(
    freq_mhz: np.ndarray,
    flux: np.ndarray,
    flags: Optional[np.ndarray] = None,
    stokes: str = "I",
    beam: int = 0,
    ax: Optional[plt.Axes] = None,
    ref_freq_mhz: Optional[float] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a frequency spectrum, optionally highlighting flagged channels.

    Parameters
    ----------
    flags : bool array, True = flagged channel
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    else:
        fig = ax.figure

    ax.plot(freq_mhz, flux, lw=0.8, color="steelblue", label=f"Stokes {stokes}")
    if flags is not None and flags.any():
        ax.scatter(
            freq_mhz[flags], flux[flags],
            s=10, color="red", zorder=5, label="Flagged"
        )
    if ref_freq_mhz is not None:
        ax.axvline(ref_freq_mhz, ls="--", lw=0.8, color="orange", label="ν₀")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel(f"Amplitude  [{stokes}]")
    ax.set_title(f"Spectrum – Beam {beam}, Stokes {stokes}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Radial profile
# ---------------------------------------------------------------------------

def plot_radial_profile(
    radius_arcmin: np.ndarray,
    amplitude: np.ndarray,
    beam: int = 0,
    stokes: str = "I",
    log_scale: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot azimuthally averaged radial beam profile."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.plot(radius_arcmin, amplitude, lw=1.2, color="teal")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Angular Offset (arcmin)")
    ax.set_ylabel("Mean Amplitude")
    ax.set_title(f"Radial Profile – Beam {beam}, Stokes {stokes}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Taylor terms
# ---------------------------------------------------------------------------

def plot_taylor_terms(
    taylor_dict: Dict[str, np.ndarray],
    pixel_scale_arcmin: Tuple[float, float] = (1.0, 1.0),
    beam: int = 0,
    stokes: str = "I",
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot T0, α, β images side by side.

    Parameters
    ----------
    taylor_dict : from TaylorAnalyser.all_taylor_images() or
                  {'T0': ..., 'alpha': ..., 'beta': ...}
    """
    labels = [l for l in ("T0", "alpha", "beta") if l in taylor_dict]
    cmaps = {"T0": DEFAULT_CMAP_AMPLITUDE, "alpha": DEFAULT_CMAP_ALPHA,
             "beta": DEFAULT_CMAP_SIGNED}
    nice = {"T0": "T₀ (amplitude)", "alpha": "α (spectral index)",
            "beta": "β (curvature)"}
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4.5))
    if len(labels) == 1:
        axes = [axes]

    for ax, lbl in zip(axes, labels):
        plot_beam_image(
            taylor_dict[lbl], pixel_scale_arcmin,
            title=nice.get(lbl, lbl),
            cmap=cmaps.get(lbl, "viridis"),
            ax=ax,
        )
    fig.suptitle(f"Taylor Terms – Beam {beam}, Stokes {stokes}")
    fig.tight_layout()
    return fig, np.asarray(axes)


# ---------------------------------------------------------------------------
# Beam comparison bar chart
# ---------------------------------------------------------------------------

def plot_beam_comparison(
    beam_stats: List[Dict],
    metric: str = "fwhm_mean_arcmin",
    ylabel: str = "",
    title: str = "",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Bar chart of a chosen metric across all beams.

    Parameters
    ----------
    beam_stats : list of dicts from BeamAnalyser.compare_beams()
    metric     : key to extract from each dict
    """
    beams = [d["beam"] for d in beam_stats]
    vals = [d.get(metric, np.nan) for d in beam_stats]
    fig, ax = plt.subplots(figsize=(max(8, len(beams) * 0.5), 4))
    ax.bar(beams, vals, color="steelblue", edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Beam index")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"{metric} per beam")
    ax.set_xticks(beams)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Per-channel RMS
# ---------------------------------------------------------------------------

def plot_freq_rms(
    freq_mhz: np.ndarray,
    rms: np.ndarray,
    flags: Optional[np.ndarray] = None,
    beam: int = 0,
    stokes: str = "I",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot per-channel spatial RMS with flagged channels highlighted.
    """
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(freq_mhz, rms, lw=0.8, color="indigo", label="Channel RMS")
    if flags is not None and flags.any():
        ax.scatter(
            freq_mhz[flags], rms[flags],
            s=15, color="red", zorder=5, label="Flagged"
        )
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Spatial RMS")
    ax.set_title(f"Per-Channel RMS – Beam {beam}, Stokes {stokes}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Smearing curve
# ---------------------------------------------------------------------------

def plot_smearing_curve(
    offsets_arcmin: np.ndarray,
    smearing_factor: np.ndarray,
    ref_freq_mhz: float = 943.0,
    chan_width_mhz: float = 1.0,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the bandwidth smearing amplitude factor vs. angular offset.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(offsets_arcmin, smearing_factor, lw=1.5, color="darkred")
    ax.axhline(1.0, ls="--", lw=0.8, color="gray")
    ax.axhline(0.99, ls=":", lw=0.8, color="orange", label="99% level")
    ax.set_xlabel("Angular Offset (arcmin)")
    ax.set_ylabel("Smearing factor R")
    ax.set_title(
        f"Bandwidth Smearing  |  ν₀ = {ref_freq_mhz:.0f} MHz, "
        f"Δν = {chan_width_mhz:.2f} MHz"
    )
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
