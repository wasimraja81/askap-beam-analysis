"""
holopy – ASKAP Holography Beam Analysis Package
================================================
Modules
-------
io          : load FITS cubes, parse headers, build WCS
beam        : per-beam geometry, peak, FWHM, sidelobe metrics
stokes      : Stokes parameter extraction, polarisation leakage
frequency   : spectral behaviour across the band
taylor      : Taylor-term cube (T0, alpha, beta) inspection
viz         : publication-ready plotting helpers
"""

from .io import HoloCube, HoloTaylor          # noqa: F401
from .beam import BeamAnalyser                # noqa: F401
from .stokes import StokesAnalyser            # noqa: F401
from .frequency import FrequencyAnalyser      # noqa: F401
from .taylor import TaylorAnalyser            # noqa: F401
from . import viz                             # noqa: F401

__version__ = "0.1.0"
