# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Statistical functions."""

from .array.extreme_values import GumbelDistribution
from .extreme_values import (
    fit_gumbel,
    return_period_to_value,
    value_to_return_period,
)
from .numpy_extended import nanaverage
from .quantiles import iter_quantiles

__all__ = [
    "GumbelDistribution",
    "fit_gumbel",
    "return_period_to_value",
    "value_to_return_period",
    "nanaverage",
    "iter_quantiles",
]
