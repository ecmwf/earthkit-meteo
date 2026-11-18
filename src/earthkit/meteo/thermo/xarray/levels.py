# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Heights where a temperature profile falls below a threshold, operating on xarray objects."""

from __future__ import annotations

import xarray as xr

from earthkit.meteo import constants
from earthkit.meteo.vertical.xarray import interpolate_sleve_to_coord_levels

__all__ = [
    "freezing_level",
    "snowfall_limit",
]


def freezing_level(
    t: xr.DataArray,
    z: xr.DataArray,
    t_sfc: xr.DataArray | float | None = None,
    z_sfc: xr.DataArray | float | None = None,
    threshold: float = constants.T_C2K,
    lapse_rate: float | None = None,
    min_height: float | None = None,
    vertical_dim: str = "z",
) -> xr.DataArray:
    r"""Compute the freezing level: the height where the temperature falls below 0 °C.

    The freezing level is the uppermost height where the temperature profile
    crosses ``threshold`` from above (warmer below, colder above), linearly
    interpolated in height. With several warm layers in a column, the top of the
    highest one is returned.

    Parameters
    ----------
    t : xarray.DataArray
        Temperature on the vertical levels (K).
    z : xarray.DataArray
        Height of the vertical levels (m), e.g. geometric height above sea level.
        The levels can be ordered bottom-up or top-down.
    t_sfc : xarray.DataArray or float, optional
        Temperature at the surface, e.g. the 2 m temperature (K). Must be given
        together with ``z_sfc``.
    z_sfc : xarray.DataArray or float, optional
        Height of the surface (m), in the same reference as ``z``, e.g. the orography.
        ``t_sfc`` is taken as the value at this height, and the levels not above it
        (or with a missing height) are treated as below the surface. Must be given
        together with ``t_sfc``.
    threshold : float, optional
        Temperature defining the freezing level (K). Defaults to 0 °C (273.15 K).
    lapse_rate : float, optional
        Rate at which the temperature decreases with height (K m-1), e.g. 0.0065.
        When given, columns without a crossing are extrapolated from the surface:
        :math:`z_{sfc} - (threshold - t_{sfc}) / lapse\_rate`. This lies below
        ``z_sfc`` when the column is colder than ``threshold`` everywhere. When not
        given, these columns are NaN. Requires ``t_sfc`` and ``z_sfc``.
    min_height : float, optional
        Lower bound of the result (m). When not given, the result is not bounded.
    vertical_dim : str, optional
        Name of the vertical dimension.

    Returns
    -------
    xarray.DataArray
        Height of the freezing level (m), in the same reference as ``z``, with the
        vertical dimension removed. NaN where there is no crossing and no
        extrapolation, and in columns containing missing values.

    Notes
    -----
    When ``t_sfc`` and ``z_sfc`` are given, the surface value is added as the lowest
    point of the profile, and the levels that are not above ``z_sfc`` (e.g. pressure
    levels below the ground) are replaced by it. The crossing can then also lie
    between the surface and the lowest level above it. Levels with a missing height
    are replaced too, so another criterion can be applied by setting their height to
    NaN, e.g. ``z.where(p < p_sfc)`` for pressure levels.

    The crossing is found with
    :func:`earthkit.meteo.vertical.interpolate_sleve_to_coord_levels` using
    ``folding_mode="high_fold"``.
    """
    return _height_of_uppermost_crossing(
        t,
        z,
        threshold,
        values_sfc=t_sfc,
        z_sfc=z_sfc,
        lapse_rate=lapse_rate,
        min_height=min_height,
        vertical_dim=vertical_dim,
    )


def snowfall_limit(
    tw: xr.DataArray,
    z: xr.DataArray,
    tw_sfc: xr.DataArray | float | None = None,
    z_sfc: xr.DataArray | float | None = None,
    threshold: float = constants.T_C2K + 1.3,
    lapse_rate: float | None = None,
    min_height: float | None = None,
    vertical_dim: str = "z",
) -> xr.DataArray:
    r"""Compute the snowfall limit: the height where the wet-bulb temperature falls below 1.3 °C.

    The snowfall limit is the uppermost height where the wet-bulb temperature
    profile crosses ``threshold`` from above (warmer below, colder above), linearly
    interpolated in height. Below it, falling snow is expected to have mostly melted.

    Parameters
    ----------
    tw : xarray.DataArray
        Wet-bulb temperature on the vertical levels (K), e.g. computed with
        :func:`earthkit.meteo.thermo.wet_bulb_temperature_from_specific_humidity`.
    z : xarray.DataArray
        Height of the vertical levels (m), e.g. geometric height above sea level.
        The levels can be ordered bottom-up or top-down.
    tw_sfc : xarray.DataArray or float, optional
        Wet-bulb temperature at the surface, e.g. at 2 m (K). Must be given together
        with ``z_sfc``.
    z_sfc : xarray.DataArray or float, optional
        Height of the surface (m), in the same reference as ``z``, e.g. the orography.
        ``tw_sfc`` is taken as the value at this height, and the levels not above it
        (or with a missing height) are treated as below the surface. Must be given
        together with ``tw_sfc``.
    threshold : float, optional
        Wet-bulb temperature defining the snowfall limit (K). Defaults to 1.3 °C
        (274.45 K).
    lapse_rate : float, optional
        Rate at which the wet-bulb temperature decreases with height (K m-1), e.g.
        0.005. When given, columns without a crossing are extrapolated from the
        surface: :math:`z_{sfc} - (threshold - tw_{sfc}) / lapse\_rate`. This lies
        below ``z_sfc`` when the column is colder than ``threshold`` everywhere. When
        not given, these columns are NaN. Requires ``tw_sfc`` and ``z_sfc``.
    min_height : float, optional
        Lower bound of the result (m). When not given, the result is not bounded.
    vertical_dim : str, optional
        Name of the vertical dimension.

    Returns
    -------
    xarray.DataArray
        Height of the snowfall limit (m), in the same reference as ``z``, with the
        vertical dimension removed. NaN where there is no crossing and no
        extrapolation, and in columns containing missing values.

    Notes
    -----
    When ``tw_sfc`` and ``z_sfc`` are given, the surface value is added as the lowest
    point of the profile, and the levels that are not above ``z_sfc`` (e.g. pressure
    levels below the ground) are replaced by it. The crossing can then also lie
    between the surface and the lowest level above it. Levels with a missing height
    are replaced too, so another criterion can be applied by setting their height to
    NaN, e.g. ``z.where(p < p_sfc)`` for pressure levels.

    The crossing is found with
    :func:`earthkit.meteo.vertical.interpolate_sleve_to_coord_levels` using
    ``folding_mode="high_fold"``.
    """
    return _height_of_uppermost_crossing(
        tw,
        z,
        threshold,
        values_sfc=tw_sfc,
        z_sfc=z_sfc,
        lapse_rate=lapse_rate,
        min_height=min_height,
        vertical_dim=vertical_dim,
    )


def _height_of_uppermost_crossing(
    values: xr.DataArray,
    z: xr.DataArray,
    threshold: float,
    *,
    values_sfc: xr.DataArray | float | None,
    z_sfc: xr.DataArray | float | None,
    lapse_rate: float | None,
    min_height: float | None,
    vertical_dim: str,
) -> xr.DataArray:
    """Height of the uppermost crossing of a profile through ``threshold`` from above.

    Implements :func:`freezing_level` and :func:`snowfall_limit`; see them for the
    parameters.
    """
    has_surface = values_sfc is not None
    if has_surface != (z_sfc is not None):
        raise ValueError("the surface value and the surface height must be given together")
    if lapse_rate is not None:
        if not has_surface:
            raise ValueError("lapse_rate requires the surface value and the surface height")
        if lapse_rate <= 0:
            raise ValueError(f"lapse_rate must be positive, got {lapse_rate}")

    values, z = _bottom_up(values, z, vertical_dim)
    if has_surface:
        values_sfc, z_sfc = xr.DataArray(values_sfc), xr.DataArray(z_sfc)
        values, z = _add_surface_point(values, z, values_sfc, z_sfc, vertical_dim)

    height = interpolate_sleve_to_coord_levels(
        z, z, values, [threshold], folding_mode="high_fold", vertical_dim=vertical_dim
    ).isel({vertical_dim: 0}, drop=True)

    # "high_fold" also accepts a crossing where the profile rises through the
    # threshold. The uppermost crossing is from above only if the profile ends
    # below the threshold.
    column_top = values.isel({vertical_dim: -1}, drop=True)
    height = height.where(column_top < threshold)

    if lapse_rate is not None:
        height = height.fillna(z_sfc - (threshold - values_sfc) / lapse_rate)

    is_complete = values.notnull().all(vertical_dim) & z.notnull().all(vertical_dim)
    height = height.where(is_complete)

    if min_height is not None:
        height = height.clip(min=min_height)

    height.name = None
    height.attrs = {}
    return height


def _bottom_up(values: xr.DataArray, z: xr.DataArray, vertical_dim: str) -> tuple[xr.DataArray, xr.DataArray]:
    """Broadcast ``values`` and ``z`` and order their levels by increasing height."""
    values, z = xr.broadcast(values, z)
    if (z.isel({vertical_dim: -1}) - z.isel({vertical_dim: 0})).mean() < 0:
        reverse = {vertical_dim: slice(None, None, -1)}
        values, z = values.isel(reverse), z.isel(reverse)
    return values, z


def _add_surface_point(
    values: xr.DataArray,
    z: xr.DataArray,
    values_sfc: xr.DataArray,
    z_sfc: xr.DataArray,
    vertical_dim: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Add the surface as the lowest point of bottom-up profiles.

    The levels not above ``z_sfc`` are replaced by the surface point. The resulting
    zero-thickness layers at ``z_sfc`` add no crossing above the surface, so the
    surface connects directly to the lowest level above it.
    """
    above_surface = z > z_sfc
    columns = values.isel({vertical_dim: 0}, drop=True)

    def with_surface(levels: xr.DataArray, surface: xr.DataArray) -> xr.DataArray:
        levels = levels.where(above_surface, surface)
        # The surface point has no vertical coordinate, so drop those of the levels,
        # and any scalar coordinate of the surface named like the vertical dimension.
        levels = levels.drop_vars([name for name, coord in levels.coords.items() if vertical_dim in coord.dims])
        surface = surface.drop_vars(vertical_dim, errors="ignore")
        surface = surface.broadcast_like(columns).expand_dims(vertical_dim).transpose(*levels.dims)
        return xr.concat([surface, levels], dim=vertical_dim)

    return with_surface(values, values_sfc), with_surface(z, z_sfc)
