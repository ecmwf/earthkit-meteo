# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

"""
Weather regimes based on projections onto spatial patterns.

- To define a collection or generator of patterns, use the provided generator
  classes to implement your desired scheme or define your own pattern generator
  scheme based on the abstract base class :py:class:`Patterns`.
- To compute regime indices, use the functions :py:func:`project` and
  :py:func:`regime_index` together with a given pattern collection/generator.
"""

from . import array
from .index import project, regime_index
from .patterns import ConstantPatterns, ModulatedPatterns, Patterns

__all__ = [
    "ConstantPatterns",
    "ModulatedPatterns",
    "Patterns",
    "project",
    "regime_index",
    "array",
]
