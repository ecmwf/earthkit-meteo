# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,  # noqa: F401
    TypeAlias,
    overload,
)

from earthkit.meteo import constants
from earthkit.meteo.utils.decorators import dispatch

if TYPE_CHECKING:
    import xarray  # type: ignore[import]
    from earthkit.data import Field, FieldList  # type: ignore[import]


ArrayLike: TypeAlias = Any


@overload
def surface_downward_shortwave_radiation(
    diffuse: "ArrayLike",
    direct: "ArrayLike",
) -> "ArrayLike": ...


@overload
def surface_downward_shortwave_radiation(
    diffuse: "xarray.DataArray",
    direct: "xarray.DataArray",
) -> "xarray.DataArray": ...


@overload
def surface_downward_shortwave_radiation(
    diffuse: "FieldList",
    direct: "FieldList",
) -> "FieldList": ...


@overload
def surface_downward_shortwave_radiation(
    diffuse: "Field",
    direct: "Field",
) -> "Field": ...


def surface_downward_shortwave_radiation(
    diffuse: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
    direct: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
) -> "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field":
    r"""Compute the global downward shortwave radiation flux at the surface.

    Parameters
    ----------
    diffuse : array-like | xarray.DataArray | FieldList | Field
        Downward diffuse shortwave radiation flux on a horizontal plane (W/m2)
    direct : array-like | xarray.DataArray | FieldList | Field
        Downward direct shortwave radiation flux on a horizontal plane (W/m2).
        This is the direct beam projected onto the horizontal, not the
        direct normal irradiance.

    Returns
    -------
    array-like | xarray.DataArray | FieldList | Field
        Downward shortwave radiation flux (W/m2)


    The result is the sum of the two components:

    .. math::

        R_{sw} = R_{diffuse} + R_{direct}

    The result is clipped to non-negative values, since a downward flux cannot
    be negative.


    Implementations
    ------------------------
    :func:`surface_downward_shortwave_radiation` calls one of the following implementations depending on
    the type of the input arguments:

    - :py:meth:`earthkit.meteo.radiation.array.surface_downward_shortwave_radiation` for array-like
    - :py:meth:`earthkit.meteo.radiation.xarray.surface_downward_shortwave_radiation`
      for xarray.DataArray
    - :py:meth:`earthkit.meteo.radiation.fieldlist.surface_downward_shortwave_radiation`
      for FieldList | Field

    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(surface_downward_shortwave_radiation, array=True)
    return dispatched(diffuse, direct)


@overload
def surface_downward_longwave_flux(
    net_longwave: "ArrayLike",
    surface_temperature: "ArrayLike",
    emissivity: float = ...,
) -> "ArrayLike": ...


@overload
def surface_downward_longwave_flux(
    net_longwave: "xarray.DataArray",
    surface_temperature: "xarray.DataArray",
    emissivity: float = ...,
) -> "xarray.DataArray": ...


@overload
def surface_downward_longwave_flux(
    net_longwave: "FieldList",
    surface_temperature: "FieldList",
    emissivity: float = ...,
) -> "FieldList": ...


@overload
def surface_downward_longwave_flux(
    net_longwave: "Field",
    surface_temperature: "Field",
    emissivity: float = ...,
) -> "Field": ...


def surface_downward_longwave_flux(
    net_longwave: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
    surface_temperature: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
    emissivity: float = constants.emissivity_surface,
) -> "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field":
    r"""Compute the downward longwave flux at the surface.

    Parameters
    ----------
    net_longwave : array-like | xarray.DataArray | FieldList | Field
        Net (downward minus upward) longwave radiation flux at the surface (W/m2).
        Downward fluxes are positive.
    surface_temperature : array-like | xarray.DataArray | FieldList | Field
        Surface (skin) temperature (K)
    emissivity : number
        Broadband longwave emissivity of the surface (1). Defaults to
        :data:`earthkit.meteo.constants.emissivity_surface`.

    Returns
    -------
    array-like | xarray.DataArray | FieldList | Field
        Downwelling longwave flux at the surface (W/m2)


    The surface longwave budget assumes a grey body radiating at the surface
    temperature, so that the net flux is the absorbed fraction of the incoming
    radiation minus the emitted one:

    .. math::

        R_{net} = \epsilon R_{lwd} - \epsilon \sigma T_{s}^{4}

    Solving for the downward flux gives:

    .. math::

        R_{lwd} = \frac{R_{net}}{\epsilon} + \sigma T_{s}^{4}

    where :math:`\sigma` is the Stefan-Boltzmann constant
    (:data:`earthkit.meteo.constants.sigma`).


    Implementations
    ------------------------
    :func:`surface_downward_longwave_flux` calls one of the following implementations
    depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.radiation.array.surface_downward_longwave_flux` for array-like
    - :py:meth:`earthkit.meteo.radiation.xarray.surface_downward_longwave_flux`
      for xarray.DataArray
    - :py:meth:`earthkit.meteo.radiation.fieldlist.surface_downward_longwave_flux`
      for FieldList | Field

    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(surface_downward_longwave_flux, array=True)
    return dispatched(net_longwave, surface_temperature, emissivity=emissivity)
