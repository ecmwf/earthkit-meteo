# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from __future__ import annotations

from earthkit.data import Field, FieldList  # type: ignore[import]

from earthkit.meteo import constants
from earthkit.meteo.utils.decorators import fieldlist_ufunc

from .. import array


def surface_downward_shortwave_radiation(diffuse: FieldList | Field, direct: FieldList | Field) -> FieldList | Field:
    r"""Compute the global downward shortwave radiation flux at the surface.

    Parameters
    ----------
    diffuse : FieldList|Field
        Downward diffuse shortwave radiation flux on a horizontal plane (W/m2)
    direct : FieldList|Field
        Downward direct shortwave radiation flux on a horizontal plane (W/m2).
        This is the direct beam projected onto the horizontal, not the
        direct normal irradiance.

    Returns
    -------
    FieldList|Field
        Downward shortwave radiation flux (W/m2). The result has the same type as the
        input ``diffuse`` (FieldList or Field).


    The result is the sum of the two components:

    .. math::

        R_{sw} = R_{diffuse} + R_{direct}

    The result is clipped to non-negative values, since a downward flux cannot
    be negative.

    """
    fieldlist_ufunc_kwargs = {"default_variable": "surface_downward_shortwave_radiation"}

    return fieldlist_ufunc(
        array.surface_downward_shortwave_radiation, diffuse, direct, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def surface_downward_longwave_flux(
    net_longwave: FieldList | Field,
    surface_temperature: FieldList | Field,
    emissivity: float = constants.emissivity_surface,
) -> FieldList | Field:
    r"""Compute the downward longwave flux at the surface.

    Parameters
    ----------
    net_longwave : FieldList|Field
        Net (downward minus upward) longwave radiation flux at the surface (W/m2).
        Downward fluxes are positive.
    surface_temperature : FieldList|Field
        Surface (skin) temperature (K)
    emissivity : number
        Broadband longwave emissivity of the surface (1). Defaults to
        :data:`earthkit.meteo.constants.emissivity_surface`.

    Returns
    -------
    FieldList|Field
        Downwelling longwave flux at the surface (W/m2). The result has the same
        type as the input ``net_longwave`` (FieldList or Field).


    The result is computed from the surface longwave budget of a grey body:

    .. math::

        R_{lwd} = \frac{R_{net}}{\epsilon} + \sigma T_{s}^{4}

    """
    fieldlist_ufunc_kwargs = {"default_variable": "surface_downward_longwave_flux"}

    return fieldlist_ufunc(
        array.surface_downward_longwave_flux,
        net_longwave,
        surface_temperature,
        emissivity=emissivity,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
    )
