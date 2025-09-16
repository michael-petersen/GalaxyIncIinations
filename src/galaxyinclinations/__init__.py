"""
galaxyinclinations

Using Fourier-Laguerre basis function expansions to measure galaxy inclinations

See README file and online documentation (https://galaxyinclinations.readthedocs.io)
for further details and usage instructions.
"""
from .morphology import galaxymorphology
from importlib.metadata import version

__version__ = version("galaxyinclinations")
__all__ = ["galaxymorphology"]
