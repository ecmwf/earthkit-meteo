# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from functools import partial

import array_api_compat.torch as _xp
from array_api_compat.torch import *  # noqa: F403

from ..compute import percentile
from ..compute import polyval

# make these methods available on the namespace
percentile = partial(percentile, _xp)
polyval = partial(polyval, _xp)


def sign(x, *args, **kwargs):
    """Reimplement the sign function to handle NaNs.

    The problem is that torch.sign returns 0 for NaNs, but the array API
    standard requires NaNs to be propagated.
    """
    x = _xp.asarray(x)
    r = _xp.sign(x, *args, **kwargs)
    r = _xp.asarray(r)
    r[_xp.isnan(x)] = _xp.nan
    return r
