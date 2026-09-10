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

from earthkit.utils.array import array_namespace

from earthkit.meteo import constants

ArrayLike: TypeAlias = Any


def surface_downward_shortwave_radiation(diffuse: ArrayLike, direct: ArrayLike) -> ArrayLike:
    r"""Compute the global downward shortwave radiation flux at the surface.

    Parameters
    ----------
    diffuse : number or array-like
        Downward diffuse shortwave radiation flux on a horizontal plane (W/m2)
    direct : number or array-like
        Downward direct shortwave radiation flux on a horizontal plane (W/m2).
        This is the direct beam projected onto the horizontal, not the
        direct normal irradiance.

    Returns
    -------
    number or array-like
        Downward shortwave radiation flux (W/m2)


    The result is the sum of the two components:

    .. math::

        R_{sw} = R_{diffuse} + R_{direct}

    Both inputs must refer to the same plane and use the same units, so the
    function equally applies to time-integrated fluxes (J/m2).

    The result is clipped to non-negative values, since a downward flux cannot
    be negative.
    """
    xp = array_namespace(diffuse, direct)
    diffuse = xp.asarray(diffuse)
    direct = xp.asarray(direct)

    return xp.clip(diffuse + direct, min=0.0)


def surface_downward_longwave_flux(
    net_longwave: ArrayLike,
    surface_temperature: ArrayLike,
    emissivity: float = constants.emissivity_surface,
) -> ArrayLike:
    r"""Compute the downward longwave flux at the surface.

    Parameters
    ----------
    net_longwave : number or array-like
        Net (downward minus upward) longwave radiation flux at the surface (W/m2).
        Downward fluxes are positive.
    surface_temperature : number or array-like
        Surface (skin) temperature (K)
    emissivity : number
        Broadband longwave emissivity of the surface (1). Defaults to
        :data:`earthkit.meteo.constants.emissivity_surface`.

    Returns
    -------
    number or array-like
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
    """
    xp = array_namespace(net_longwave, surface_temperature)
    net_longwave = xp.asarray(net_longwave)
    surface_temperature = xp.asarray(surface_temperature)

    return net_longwave / emissivity + constants.sigma * surface_temperature**4
