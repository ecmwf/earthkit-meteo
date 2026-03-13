# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""
Vertical computation functions.

The API is organised in layers:

- Core numerical routines live in the ``array`` submodule.
- Functions exposed from this module provide the high-level entry points for the
  vertical API.

For xarray interpolation workflows, see :mod:`earthkit.meteo.vertical.interpolation`.
"""

from .vertical import *  # noqa
