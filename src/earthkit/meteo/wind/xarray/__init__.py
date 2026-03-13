# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""
Wind related functions operating on xarray objects.
"""

from .wind import coriolis
from .wind import direction
from .wind import polar_to_xy
from .wind import speed
from .wind import w_from_omega
from .wind import windrose
from .wind import xy_to_polar

__all__ = [
    "coriolis",
    "direction",
    "polar_to_xy",
    "speed",
    "w_from_omega",
    "windrose",
    "xy_to_polar",
]
