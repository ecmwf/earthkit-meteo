# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from ..utils.decorators import dispatch


def iter_quantiles(arr, which=100, axis=0, method="sort"):
    """Iterate over the quantiles of a large array


    .. admonition:: Implementations

        Depending on the type of argument `arr`, this function calls:

        - :py:func:`earthkit.meteo.stats.array.iter_quantiles` for ``array_like``
    """
    dispatched = dispatch(iter_quantiles, array=True)
    return dispatched(arr, which, axis, method)
