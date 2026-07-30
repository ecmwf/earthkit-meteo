"""
Wind related functions.

The API is split into two layers:

- Low-level interfaces are in the ``array``, ``xarray`` and ``fieldlist`` submodules.
- High-level functions are in this module and dispatch to backend implementations
  based on input type.
"""

from .wind import *  # noqa
