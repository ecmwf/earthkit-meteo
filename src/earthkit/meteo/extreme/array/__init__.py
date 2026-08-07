# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

"""
Extreme index functions operating on numpy arrays.
"""

from .cpf import cpf  # noqa
from .efi import efi  # noqa
from .sot import sot  # noqa
from .sot import sot_func  # noqa
from .sot import sot_unsorted  # noqa

__all__ = [
    "cpf",
    "efi",
    "sot",
    "sot_unsorted",
    "sot_func",
]
