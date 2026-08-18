# (C) Copyright 2025- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""
Weather regimes based on projections onto spatial patterns.

- To define a collection or generator of patterns, use the provided generator
  classes to implement your desired scheme or define your own pattern generator
  scheme based on the abstract base class :py:class:`Patterns`.
- To compute regime indices, use the functions :py:func:`project` and
  :py:func:`regime_index` together with a given pattern collection/generator.
  These functions dispatch to backend implementations in the ``array``,
  ``xarray`` and ``fieldlist`` submodules based on the input type.

Requires :py:mod:`earthkit.geo`.
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
