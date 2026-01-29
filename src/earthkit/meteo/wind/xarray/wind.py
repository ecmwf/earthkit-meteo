# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import Iterable

import numpy as np
import xarray as xr

from .. import array


def _sample_arg(arg: object) -> np.ndarray:
    if hasattr(arg, "dtype"):
        return np.zeros((), dtype=arg.dtype)
    return np.zeros((), dtype=float)


def _infer_output_dtypes(func, *args, **kwargs) -> list[np.dtype]:
    sample_args = [_sample_arg(arg) for arg in args]
    res = func(*sample_args, **kwargs)
    if isinstance(res, tuple):
        return [np.asarray(item).dtype for item in res]
    return [np.asarray(res).dtype]


def _apply_ufunc(func, num, *args, **kwargs):
    output_dtypes = _infer_output_dtypes(func, *args, **kwargs)

    opt = {}
    # TODO: temporary fix for xarray ufunc with multiple outputs
    # to make wind tests pass
    if num > 1:
        input_core_dims = [x.dims for x in args]
        output_core_dims = [x.dims for x in args]

        opt = {
            "input_core_dims": input_core_dims,
            "output_core_dims": output_core_dims,
        }

    return xr.apply_ufunc(
        func,
        *args,
        kwargs=kwargs,
        dask="parallelized",
        output_dtypes=output_dtypes,
        **opt,
        keep_attrs=True,
    )


def speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    r"""Compute the wind speed/vector magnitude.

    Parameters
    ----------
    u: xarray.DataArray
        u wind/x vector component
    v: xarray.DataArray
        v wind/y vector component (same units as ``u``)

    Returns
    -------
    xarray.DataArray
        Wind speed/magnitude (same units as ``u`` and ``v``)
    """
    res = _apply_ufunc(array.speed, 1, u, v)
    res.name = "wind_speed"
    res.attrs["standard_name"] = "wind_speed"
    res.attrs["long_name"] = "Wind Speed"
    return res


def direction(
    u: xr.DataArray, v: xr.DataArray, convention: str = "meteo", to_positive: bool = True
) -> xr.DataArray:
    r"""Compute the direction/angle of a vector quantity.

    Parameters
    ----------
    u: xarray.DataArray
        u wind/x vector component
    v: xarray.DataArray
        v wind/y vector component (same units as ``u``)
    convention: str, optional
        Specify how the direction/angle is interpreted. The possible values are as follows:

        * "meteo": the direction is the meteorological wind direction (see below for explanation)
        * "polar": the direction is measured anti-clockwise from the x axis (East/right) to the vector

    to_positive: bool, optional
        If True, the resulting values are mapped into the [0, 360] range when
        ``convention`` is "polar". Otherwise they lie in the [-180, 180] range.

    Returns
    -------
    xarray.DataArray
        Direction/angle (degrees)

    Notes
    -----
    The meteorological wind direction is the direction from which the wind is
    blowing. Wind direction increases clockwise such that a northerly wind
    is 0°, an easterly wind is 90°, a southerly wind is 180°, and a westerly
    wind is 270°. The figure below illustrates how it is related to the actual
    orientation of the wind vector:

    .. image:: /_static/wind_direction.png
        :width: 400px

    """
    return _apply_ufunc(array.direction, 1, u, v, convention=convention, to_positive=to_positive)


def xy_to_polar(
    x: xr.DataArray, y: xr.DataArray, convention: str = "meteo"
) -> tuple[xr.DataArray, xr.DataArray]:
    r"""Convert wind/vector data from xy representation to polar representation.

    Parameters
    ----------
    x: xarray.DataArray
        u wind/x vector component
    y: xarray.DataArray
        v wind/y vector component (same units as ``x``)
    convention: str
        Specify how the direction/angle component of the target polar coordinate
        system is interpreted. The possible values are as follows:

        * "meteo": the direction is the meteorological wind direction
        * "polar": the direction is measured anti-clockwise from the x axis (East/right) to the vector

    Returns
    -------
    xarray.DataArray
        Magnitude (same units as ``x``)
    xarray.DataArray
        Direction (degrees)


    Notes
    -----
    In the target xy representation the x axis points East while the y axis points North.

    """
    return _apply_ufunc(array.xy_to_polar, 2, x, y, convention=convention)


def polar_to_xy(
    magnitude: xr.DataArray, direction: xr.DataArray, convention: str = "meteo"
) -> tuple[xr.DataArray, xr.DataArray]:
    r"""Convert wind/vector data from polar representation to xy representation.

    Parameters
    ----------
    magnitude: xr.DataArray
        Speed/magnitude of the vector
    direction: xr.DataArray
        Direction of the vector (degrees)
    convention: str
        Specify how ``direction`` is interpreted. The possible values are as follows:

        * "meteo": ``direction`` is the meteorological wind direction
        * "polar": ``direction`` is the angle measured anti-clockwise from the x axis
          (East/right) to the vector

    Returns
    -------
    xarray.DataArray
        X vector component (same units as ``magnitude``)
    xarray.DataArray
        Y vector component (same units as ``magnitude``)


    Notes
    -----
    In the target xy representation the x axis points East while the y axis points North.

    """
    return _apply_ufunc(array.polar_to_xy, 2, magnitude, direction, convention=convention)


def w_from_omega(omega: xr.DataArray, t: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    r"""Compute the hydrostatic vertical velocity from pressure velocity, temperature and pressure.

    Parameters
    ----------
    omega : xarray.DataArray
        Hydrostatic pressure velocity (Pa/s)
    t : xarray.DataArray
        Temperature (K)
    p : xarray.DataArray
        Pressure (Pa)

    Returns
    -------
    xarray.DataArray
        Hydrostatic vertical velocity (m/s)


    Notes
    -----
    The computation is based on the following hydrostatic formula:

    .. math::

        w = - \frac{\omega\; t R_{d}}{p g}

    where

        * :math:`R_{d}` is the specific gas constant for dry air (see :data:`earthkit.meteo.constants.Rd`).
        * :math:`g` is the gravitational acceleration (see :data:`earthkit.meteo.constants.g`)

    """
    return _apply_ufunc(array.w_from_omega, 1, omega, t, p)


def coriolis(lat: xr.DataArray) -> xr.DataArray:
    r"""Compute the Coriolis parameter.

    Parameters
    ----------
    lat : xarray.DataArray
        Latitude (degrees)

    Returns
    -------
    xarray.DataArray
        The Coriolis parameter (:math:`s^{-1}`)


    Notes
    -----
    The Coriolis parameter is defined by the following formula:

    .. math::

        f = 2 \Omega sin(\phi)

    where :math:`\Omega` is the rotation rate of Earth
    (see :data:`earthkit.meteo.constants.omega`) and :math:`\phi` is the latitude.

    """
    return _apply_ufunc(array.coriolis, 1, lat)


def windrose(
    speed: xr.DataArray,
    direction: xr.DataArray,
    sectors: int = 16,
    speed_bins: Iterable[float] | None = None,
    percent: bool = True,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Generate windrose data.

    Parameters
    ----------
    speed : xarray.DataArray
        Speed
    direction : xarray.DataArray
        Meteorological wind direction (degrees). See :func:`earthkit.meteo.wind.direction` for details.
        Values must be between 0 and 360.
    sectors: number
        Number of sectors the 360 degrees direction range is split into.
    speed_bin: array-like
        Speed bins
    percent: bool
        If False, returns the number of valid samples in each bin. If True, returns
        the percentage of the number of samples in each bin with respect to the total
        number of valid samples.

    Returns
    -------
    xarray.DataArray
        The bi-dimensional histogram of ``speed`` and ``direction``.  Values in
        ``speed`` are histogrammed along the first dimension and values in ``direction``
        are histogrammed along the second dimension.
    xarray.DataArray
        The direction bins (i.e. the sectors) (degrees)


    Notes
    -----
    The ``sectors`` parameter defines the number of direction bins the 360 degreesThe sectors do not start at 0 degrees (North) but are shifted by half a sector size.
    E.g. if ``sectors`` is 4 the sectors are defined as:

    .. image:: /_static/wind_sector.png
        :width: 350px

    """
    import xarray as xr

    speed_bins = [] if speed_bins is None else speed_bins
    speed_np = speed.to_numpy() if hasattr(speed, "to_numpy") else np.asarray(speed)
    direction_np = direction.to_numpy() if hasattr(direction, "to_numpy") else np.asarray(direction)

    res, dir_bins = array.windrose(
        speed_np,
        direction_np,
        sectors=sectors,
        speed_bins=speed_bins,
        percent=percent,
    )

    speed_bins = np.asarray(speed_bins, dtype=res.dtype)
    dir_bins = np.asarray(dir_bins, dtype=res.dtype)

    res_da = xr.DataArray(
        res,
        dims={"speed_bin": res.shape[0], "direction_bin": res.shape[1]},
        coords={
            "speed_bin": speed_bins[:-1],
            "direction_bin": dir_bins[:-1],
        },
    )

    dir_bins_da = xr.DataArray(dir_bins, dims=("direction_bin_edge",))

    return res_da, dir_bins_da
