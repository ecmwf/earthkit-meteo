# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from functools import partial

import array_api_compat.cupy as _xp
from array_api_compat.cupy import *  # noqa: F403

# make polyval available on the namespace
from cupy.polynomial.polynomial import polyval  # noqa: F401

from earthkit.meteo.utils.compute import seterr

# make these methods available on the namespace
seterr = partial(seterr, _xp)
