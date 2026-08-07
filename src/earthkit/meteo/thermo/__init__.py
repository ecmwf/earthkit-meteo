# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

"""
Thermodynamic functions.

The API is split into two layers:

- Low-level implementations are in the ``array``, ``xarray`` and ``fieldlist`` submodules.
- High-level functions are in this module and dispatch to backend implementations
  based on input type.
"""

from .thermo import *  # noqa
