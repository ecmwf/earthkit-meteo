# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import xarray as xr

from earthkit.meteo import constants
from earthkit.meteo.utils.decorators import xarray_ufunc

from .. import array


def surface_downward_shortwave_radiation(diffuse: xr.DataArray, direct: xr.DataArray) -> xr.DataArray:
    r"""Compute the global downward shortwave radiation flux at the surface.

    Parameters
    ----------
    diffuse : xarray.DataArray
        Downward diffuse shortwave radiation flux on a horizontal plane (W/m2)
    direct : xarray.DataArray
        Downward direct shortwave radiation flux on a horizontal plane (W/m2).
        This is the direct beam projected onto the horizontal, not the
        direct normal irradiance.

    Returns
    -------
    xarray.DataArray
        Downward shortwave radiation flux (W/m2)


    The result is the sum of the two components:

    .. math::

        R_{sw} = R_{diffuse} + R_{direct}

    The result is clipped to non-negative values, since a downward flux cannot
    be negative.

    """
    return xarray_ufunc(array.surface_downward_shortwave_radiation, diffuse, direct).assign_attrs({
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
        "units": "W m-2",
    })


def surface_downward_longwave_flux(
    net_longwave: xr.DataArray,
    surface_temperature: xr.DataArray,
    emissivity: float = constants.emissivity_surface,
) -> xr.DataArray:
    r"""Compute the downward longwave flux at the surface.

    Parameters
    ----------
    net_longwave : xarray.DataArray
        Net (downward minus upward) longwave radiation flux at the surface (W/m2).
        Downward fluxes are positive.
    surface_temperature : xarray.DataArray
        Surface (skin) temperature (K)
    emissivity : number
        Broadband longwave emissivity of the surface (1). Defaults to
        :data:`earthkit.meteo.constants.emissivity_surface`.

    Returns
    -------
    xarray.DataArray
        Downwelling longwave flux at the surface (W/m2)


    The result is computed from the surface longwave budget of a grey body:

    .. math::

        R_{lwd} = \frac{R_{net}}{\epsilon} + \sigma T_{s}^{4}

    """
    return xarray_ufunc(
        array.surface_downward_longwave_flux,
        net_longwave,
        surface_temperature,
        emissivity=emissivity,
    ).assign_attrs({
        "standard_name": "surface_downwelling_longwave_flux_in_air",
        "units": "W m-2",
    })
