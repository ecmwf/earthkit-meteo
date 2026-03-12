# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""
Forecast scoring functions.

The API is split into two layers:

- Low-level interfaces are in the ``array`` and ``xarray`` submodules.
- High-level functions are in this module and dispatch to the ``array`` and
  ``xarray`` implementations based on input type.
"""

from .deterministic import *
from .ensemble import *
