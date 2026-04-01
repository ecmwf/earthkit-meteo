# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import Any, TypeAlias

from earthkit.meteo import constants

ArrayLike: TypeAlias = Any


def celsius_to_kelvin(t: ArrayLike) -> ArrayLike:
    """Convert temperature values from Celsius to Kelvin.

    Parameters
    ----------
    t : number or array-like
        Temperature in Celsius units

    Returns
    -------
    number or array-like
        Temperature in Kelvin units

    """
    return t + constants.T_C2K


def kelvin_to_celsius(t: ArrayLike) -> ArrayLike:
    """Convert temperature values from Kelvin to Celsius.

    Parameters
    ----------
    t : number or array-like
        Temperature in Kelvin units

    Returns
    -------
    number or array-like
        Temperature in Celsius units

    """
    return t - constants.T_C2K
