# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from ..utils.decorators import dispatch


def nanaverage(data, weights=None, **kwargs):
    """A merge of the functionality of np.nanmean and np.average.


    .. admonition:: Implementations

        Depending on the type of argument `data`, this function calls:

        - :py:func:`earthkit.meteo.stats.xarray.nanaverage` for ``xarray.DataArray``
        - :py:func:`earthkit.meteo.stats.array.nanaverage` for ``array_like``
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(nanaverage, xarray=True, array=True)
    return dispatched(data, weights, **kwargs)
