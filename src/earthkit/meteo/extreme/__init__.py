# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

"""
Extreme index functions.

The API is split into two layers:

- Low-level implementations are in the ``array`` and ``xarray`` submodules.
- High-level functions are in this module and dispatch to backend implementations
  based on input type.
"""

from .cpf import cpf  # noqa
from .efi import efi  # noqa
from .sot import sot  # noqa
from .sot import sot_unsorted  # noqa

__all__ = [
    "cpf",
    "efi",
    "sot",
    "sot_unsorted",
]
