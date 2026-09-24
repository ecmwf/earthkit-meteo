# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

"""
Vertical computation functions.

The API is split into two layers:

- Low-level interfaces are in the ``array``, ``xarray`` and ``fieldlist`` submodules.
- High-level functions are in this module and dispatch to backend implementations
  based on input type.

Please note that the majority of the functions in this module are still under development and
not available for all the supported input formats. See the individual functions for details.
"""

from .vertical import *  # noqa
