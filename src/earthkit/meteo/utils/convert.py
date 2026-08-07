# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

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
