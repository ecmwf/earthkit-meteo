# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import xarray as xr

from earthkit.meteo.utils.decorators import xarray_ufunc

from .. import array


def celsius_to_kelvin(t: xr.DataArray) -> xr.DataArray:
    r"""Convert temperature values from Celsius to Kelvin.

    Parameters
    ----------
    t : xarray.DataArray
        Temperature in Celsius units

    Returns
    -------
    xarray.DataArray
        Temperature in Kelvin units

    """
    return xarray_ufunc(array.celsius_to_kelvin, t).assign_attrs(
        {
            "standard_name": "air_temperature",
            "units": "K",
        }
    )


def kelvin_to_celsius(t: xr.DataArray) -> xr.DataArray:
    r"""Convert temperature values from Kelvin to Celsius.

    Parameters
    ----------
    t : xarray.DataArray
        Temperature in Kelvin units

    Returns
    -------
    xarray.DataArray
        Temperature in Celsius units

    """
    return xarray_ufunc(array.kelvin_to_celsius, t).assign_attrs(
        {
            "standard_name": "air_temperature",
            "units": "degC",
        }
    )


def specific_humidity_from_mixing_ratio(w: xr.DataArray) -> xr.DataArray:
    r"""Compute the specific humidity from mixing ratio.

    Parameters
    ----------
    w : xarray.DataArray
        Mixing ratio (kg/kg)

    Returns
    -------
    xarray.DataArray
        Specific humidity (kg/kg)


    The result is the specific humidity in kg/kg units. The computation is based on
    the following definition [Wallace2006]_:

    .. math::

        q = \frac {w}{1+w}

    """
    return xarray_ufunc(array.specific_humidity_from_mixing_ratio, w).assign_attrs(
        {
            "standard_name": "specific_humidity",
            "units": "kg kg-1",
        }
    )


def mixing_ratio_from_specific_humidity(q: xr.DataArray) -> xr.DataArray:
    r"""Compute the mixing ratio from specific humidity.

    Parameters
    ----------
    q : xarray.DataArray
        Specific humidity (kg/kg)

    Returns
    -------
    xarray.DataArray
        Mixing ratio (kg/kg)


    The result is the mixing ratio in kg/kg units. The computation is based on
    the following definition [Wallace2006]_:

    .. math::

        w = \frac {q}{1-q}

    """
    return xarray_ufunc(array.mixing_ratio_from_specific_humidity, q).assign_attrs(
        {
            "standard_name": "humidity_mixing_ratio",
            "units": "kg kg-1",
        }
    )


def vapour_pressure_from_specific_humidity(q: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    r"""Compute the vapour pressure from specific humidity.

    Parameters
    ----------
    q: xarray.DataArray
        Specific humidity (kg/kg)
    p: xarray.DataArray
        Pressure (Pa)

    Returns
    -------
    xarray.DataArray
        Vapour pressure (Pa)


    The computation is based on the following formula [Wallace2006]_:

    .. math::

        e = \frac{p\;q}{\epsilon\; (1 + q(\frac{1}{\epsilon} -1 ))}

    with :math:`\epsilon =  R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).

    """
    return xarray_ufunc(array.vapour_pressure_from_specific_humidity, q, p).assign_attrs(
        {
            "standard_name": "water_vapour_partial_pressure_in_air",
            "units": "Pa",
        }
    )


def vapour_pressure_from_mixing_ratio(w: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    r"""Compute the vapour pressure from mixing ratio.

    Parameters
    ----------
    w: xarray.DataArray
        Mixing ratio (kg/kg)
    p: xarray.DataArray
        Pressure (Pa)

    Returns
    -------
    xarray.DataArray
        Vapour pressure (Pa)


    The computation is based on the following formula:

    .. math::

        e = \frac{p\;w}{\epsilon + w}

    with :math:`\epsilon =  R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).

    """
    return xarray_ufunc(array.vapour_pressure_from_mixing_ratio, w, p).assign_attrs(
        {
            "standard_name": "water_vapour_partial_pressure_in_air",
            "units": "Pa",
        }
    )


def specific_humidity_from_vapour_pressure(
    e: xr.DataArray, p: xr.DataArray, eps: float = 1e-4
) -> xr.DataArray:
    r"""Compute the specific humidity from vapour pressure.

    Parameters
    ----------
    e: xarray.DataArray
        Vapour pressure (Pa)
    p: xarray.DataArray
        Pressure (Pa)
    eps: number
        Where p - e < ``eps`` nan is returned.

    Returns
    -------
    xarray.DataArray
        Specific humidity (kg/kg)


    The computation is based on the following formula:

    .. math::

       q = \frac{\epsilon e}{p + e(\epsilon-1)}

    with :math:`\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).

    """
    return xarray_ufunc(array.specific_humidity_from_vapour_pressure, e, p, eps=eps).assign_attrs(
        {
            "standard_name": "specific_humidity",
            "units": "kg kg-1",
        }
    )


def mixing_ratio_from_vapour_pressure(e: xr.DataArray, p: xr.DataArray, eps: float = 1e-4) -> xr.DataArray:
    r"""Compute the mixing ratio from vapour pressure.

    Parameters
    ----------
    e: xarray.DataArray
        Vapour pressure (Pa)
    p: xarray.DataArray
        Pressure (Pa)
    eps: number
        Where p - e < ``eps`` nan is returned.

    Returns
    -------
    xarray.DataArray
        Mixing ratio (kg/kg).


    The computation is based on the following formula:

    .. math::

       w = \frac{\epsilon e}{p - e}

    with :math:`\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).

    """
    return xarray_ufunc(array.mixing_ratio_from_vapour_pressure, e, p, eps=eps).assign_attrs(
        {
            "standard_name": "humidity_mixing_ratio",
            "units": "kg kg-1",
        }
    )


def saturation_vapour_pressure(t: xr.DataArray, phase: str = "mixed") -> xr.DataArray:
    r"""Compute the saturation vapour pressure from temperature with respect to a phase.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    phase: str, optional
        Define the phase with respect to the saturation vapour pressure is computed.
        It is either “water”, “ice” or “mixed”.

    Returns
    -------
    xarray.DataArray
        Saturation vapour pressure (Pa)


    The algorithm was taken from the IFS model [IFS-CY47R3-PhysicalProcesses]_ (see Chapter 12).
    It uses the following formula when ``phase`` is "water" or "ice":

    .. math::

        e_{sat} = a_{1}\;exp \left(a_{3}\frac{t-273.16}{t-a_{4}}\right)

    where the parameters are set as follows:

    * ``phase`` = "water": :math:`a_{1}` =611.21 Pa, :math:`a_{3}` =17.502 and :math:`a_{4}` =32.19 K
    * ``phase`` = "ice": :math:`a_{1}` =611.21 Pa, :math:`a_{3}` =22.587 and :math:`a_{4}` =-0.7 K

    When ``phase`` is "mixed" the formula is based on the value of ``t``:

    * if :math:`t <= t_{i}`: the formula for ``phase`` = "ice" is used (:math:`t_{i} = 250.16 K`)
    * if :math:`t >= t_{0}`: the formula for ``phase`` = "water" is used (:math:`t_{0} = 273.16 K`)
    * for the range :math:`t_{i} < t < t_{0}` an interpolation is used between the "ice" and "water" phases:

    .. math::

        \alpha(t) e_{wsat}(t) + (1 - \alpha(t)) e_{isat}(t)

    with :math:`\alpha(t) = (\frac{t-t_{i}}{t_{0}-t_{i}})^2`.

    """
    return xarray_ufunc(array.saturation_vapour_pressure, t, phase=phase).assign_attrs(
        {
            "standard_name": "",  # no standard name
            "units": "Pa",
            "long_name": f"Saturation vapour pressure w.r.t. {phase} phase",
        }
    )


def saturation_mixing_ratio(t: xr.DataArray, p: xr.DataArray, phase: str = "mixed") -> xr.DataArray:
    r"""Compute the saturation mixing ratio from temperature with respect to a phase.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    p: xarray.DataArray
        Pressure (Pa)
    phase: str
        Define the phase with respect to the :func:`saturation_vapour_pressure` is computed.
        It is either “water”, “ice” or “mixed”.

    Returns
    -------
    xarray.DataArray
        Saturation mixing ratio (kg/kg)


    Equivalent to the following code:

    .. code-block:: python

        e = saturation_vapour_pressure(t, phase=phase)
        return mixing_ratio_from_vapour_pressure(e, p)

    """
    return xarray_ufunc(array.saturation_mixing_ratio, t, p, phase=phase).assign_attrs(
        {
            "standard_name": "",  # no standard name
            "units": "kg kg-1",
            "long_name": f"Saturation mixing ratio w.r.t. {phase} phase",
        }
    )


def saturation_specific_humidity(t: xr.DataArray, p: xr.DataArray, phase: str = "mixed") -> xr.DataArray:
    r"""Compute the saturation specific humidity from temperature with respect to a phase.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    p: xarray.DataArray
        Pressure (Pa)
    phase: str, optional
        Define the phase with respect to the :func:`saturation_vapour_pressure` is computed.
        It is either “water”, “ice” or “mixed”.

    Returns
    -------
    xarray.DataArray
        Saturation specific humidity (kg/kg)


    Equivalent to the following code:

    .. code-block:: python

        e = saturation_vapour_pressure(t, phase=phase)
        return specific_humidity_from_vapour_pressure(e, p)

    """
    return xarray_ufunc(array.saturation_specific_humidity, t, p, phase=phase).assign_attrs(
        {
            "standard_name": "",  # no standard name
            "units": "kg kg-1",
            "long_name": f"Saturation specific humidity w.r.t. {phase} phase",
        }
    )


def saturation_vapour_pressure_slope(t: xr.DataArray, phase: str = "mixed") -> xr.DataArray:
    r"""Compute the slope of saturation vapour pressure with respect to temperature.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    phase: str, optional
        Define the phase with respect to the computation will be performed.
        It is either “water”, “ice” or “mixed”. See :func:`saturation_vapour_pressure`
        for details.

    Returns
    -------
    xarray.DataArray
        Slope of saturation vapour pressure (Pa/K)

    """
    return xarray_ufunc(array.saturation_vapour_pressure_slope, t, phase=phase).assign_attrs(
        {
            "standard_name": "",  # no standard name
            "units": "Pa K-1",
            "long_name": f"Derivative of saturation vapour pressure w.r.t. temperature and {phase} phase",
        }
    )


def saturation_mixing_ratio_slope(
    t: xr.DataArray,
    p: xr.DataArray,
    es: xr.DataArray | None = None,
    es_slope: xr.DataArray | None = None,
    phase: str = "mixed",
    eps: float = 1e-4,
) -> xr.DataArray:
    r"""Compute the slope of saturation mixing ratio with respect to temperature.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    p: xarray.DataArray
        Pressure (Pa)
    es: xarray.DataArray or None, optional
        :func:`saturation_vapour_pressure` pre-computed for the given ``phase`` (Pa)
    es_slope: xarray.DataArray or None, optional
        :func:`saturation_vapour_pressure_slope` pre-computed for the given ``phase`` (Pa/K)
    phase: str, optional
        Define the phase with respect to the computation will be performed.
        It is either “water”, “ice” or “mixed”. See :func:`saturation_vapour_pressure`
    eps: number
        Where p - es < ``eps`` nan is returned.

    Returns
    -------
    xarray.DataArray
        Slope of saturation mixing ratio (:math:`kg kg^{-1} K^{-1}`)


    The computation is based on the following formula:

    .. math::

        \frac{\partial w_{s}}{\partial t} = \frac{\epsilon\; p}{(p-e_{s})^{2}} \frac{d e_{s}}{d t}

    where

        * :math:`\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).
        * :math:`e_{s}` is the :func:`saturation_vapour_pressure` for the given ``phase``

    """
    return xarray_ufunc(
        array.saturation_mixing_ratio_slope,
        t,
        p,
        es,
        es_slope,
        phase=phase,
        eps=eps,
    ).assign_attrs(
        {
            "standard_name": "",  # no standard name
            "units": "kg kg-1 K-1",
            "long_name": f"Derivative of saturation mixing ratio w.r.t. temperature and {phase} phase",
        }
    )


def saturation_specific_humidity_slope(
    t: xr.DataArray,
    p: xr.DataArray,
    es: xr.DataArray | None = None,
    es_slope: xr.DataArray | None = None,
    phase: str = "mixed",
    eps: float = 1e-4,
) -> xr.DataArray:
    r"""Compute the slope of saturation specific humidity with respect to temperature.

    Parameters
    ----------
    t:  xarray.DataArray
        Temperature (K)
    p:  xarray.DataArray
        Pressure (Pa)
    es:  xarray.DataArray or None, optional
        :func:`saturation_vapour_pressure` pre-computed for the given ``phase`` (Pa)
    es_slope: xarray.DataArray or None, optional
        :func:`saturation_vapour_pressure_slope` pre-computed for the given ``phase`` (Pa/K)
    phase: str, optional
        Define the phase with respect to the computation will be performed.
        It is either “water”, “ice” or “mixed”. See :func:`saturation_vapour_pressure`
    eps: number
        Where p - es < ``eps`` nan is returned.

    Returns
    -------
    xarray.DataArray
        Slope of saturation specific humidity (:math:`kg kg^{-1} K^{-1}`)


    The computation is based on the following formula:

    .. math::

        \frac{\partial q_{s}}{\partial t} =
        \frac{\epsilon\; p}{(p+e_{s}(\epsilon - 1))^{2}} \frac{d e_{s}}{d t}

    where

        * :math:`\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).
        * :math:`e_{s}` is the :func:`saturation_vapour_pressure` for the given ``phase``

    """
    return xarray_ufunc(
        array.saturation_specific_humidity_slope,
        t,
        p,
        es,
        es_slope,
        phase=phase,
        eps=eps,
    ).assign_attrs(
        {
            "standard_name": "",  # no standard name
            "units": "kg kg-1 K-1",
            "long_name": f"Derivative of saturation specific humidity w.r.t. temperature and {phase} phase",
        }
    )


def temperature_from_saturation_vapour_pressure(es: xr.DataArray) -> xr.DataArray:
    r"""Compute the temperature from saturation vapour pressure.

    Parameters
    ----------
    es: xarray.DataArray
        :func:`saturation_vapour_pressure` (Pa)

    Returns
    -------
    xarray.DataArray
        Temperature (K). For zero ``es`` values returns nan.


    The computation is always based on the "water" phase of
    the :func:`saturation_vapour_pressure` formulation irrespective of the
    phase ``es`` was computed to.

    """
    return xarray_ufunc(array.temperature_from_saturation_vapour_pressure, es).assign_attrs(
        {
            "standard_name": "air_temperature",
            "units": "K",
        }
    )


def relative_humidity_from_dewpoint(t: xr.DataArray, td: xr.DataArray) -> xr.DataArray:
    r"""Compute the relative humidity from dewpoint temperature.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    td: xarray.DataArray
        Dewpoint (K)

    Returns
    -------
    xarray.DataArray
        Relative humidity (%)


    The computation is based on the following formula:

    .. math::

        r = 100 \frac {e_{wsat}(td)}{e_{wsat}(t)}

    where :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water.

    """
    return xarray_ufunc(array.relative_humidity_from_dewpoint, t, td).assign_attrs(
        {
            "standard_name": "relative_humidity",
            "units": "%",
        }
    )


def relative_humidity_from_specific_humidity(
    t: xr.DataArray, q: xr.DataArray, p: xr.DataArray
) -> xr.DataArray:
    r"""Compute the relative humidity from specific humidity.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    q: xarray.DataArray
        Specific humidity (kg/kg)
    p: xarray.DataArray
        Pressure (Pa)

    Returns
    -------
    xarray.DataArray
        Relative humidity (%)


    The computation is based on the following formula:

    .. math::

        r = 100 \frac {e(q, p)}{e_{msat}(t)}

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure_from_specific_humidity`)
        * :math:`e_{msat}` is the :func:`saturation_vapour_pressure` based on the "mixed" phase

    """
    return xarray_ufunc(array.relative_humidity_from_specific_humidity, t, q, p).assign_attrs(
        {
            "standard_name": "relative_humidity",
            "units": "%",
        }
    )


def specific_humidity_from_dewpoint(td: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    r"""Compute the specific humidity from dewpoint.

    Parameters
    ----------
    td: xarray.DataArray
        Dewpoint (K)
    p: xarray.DataArray
        Pressure (Pa)

    Returns
    -------
    xarray.DataArray
        Specific humidity (kg/kg)


    The computation starts with determining the vapour pressure:

    .. math::

        e(q, p) = e_{wsat}(td)

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure_from_specific_humidity`)
        * :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water
        * :math:`q` is the specific humidity

    Then `q` is computed from :math:`e` using :func:`specific_humidity_from_vapour_pressure`.

    """
    return xarray_ufunc(array.specific_humidity_from_dewpoint, td, p).assign_attrs(
        {
            "standard_name": "specific_humidity",
            "units": "kg kg-1",
        }
    )


def mixing_ratio_from_dewpoint(td: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    r"""Compute the mixing ratio from dewpoint.

    Parameters
    ----------
    td: xarray.DataArray
        Dewpoint (K)
    p: xarray.DataArray
        Pressure (Pa)

    Returns
    -------
    xarray.DataArray
        Specific humidity (kg/kg)


    The computation starts with determining the vapour pressure:

    .. math::

        e(w, p) = e_{wsat}(td)

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure_from_mixing_ratio`)
        * :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water
        * :math:`w` is the mixing ratio

    Then `w` is computed from :math:`e` using :func:`mixing_ratio_from_vapour_pressure`.

    """
    return xarray_ufunc(array.mixing_ratio_from_dewpoint, td, p).assign_attrs(
        {
            "standard_name": "humidity_mixing_ratio",
            "units": "kg kg-1",
        }
    )


def specific_humidity_from_relative_humidity(
    t: xr.DataArray, r: xr.DataArray, p: xr.DataArray
) -> xr.DataArray:
    r"""Compute the specific humidity from relative_humidity.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    r: xarray.DataArray
        Relative humidity(%)
    p: xarray.DataArray
        Pressure (Pa)

    Returns
    -------
    xarray.DataArray
        Specific humidity (kg/kg) units


    The computation starts with determining the the vapour pressure:

    .. math::

        e(q, p) = r\; \frac{e_{msat}(t)}{100}

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure`)
        * :math:`e_{msat}` is the :func:`saturation_vapour_pressure` based on the "mixed" phase
        * :math:`q` is the specific humidity

    Then :math:`q` is computed from :math:`e` using :func:`specific_humidity_from_vapour_pressure`.

    """
    return xarray_ufunc(array.specific_humidity_from_relative_humidity, t, r, p).assign_attrs(
        {
            "standard_name": "specific_humidity",
            "units": "kg kg-1",
        }
    )


def dewpoint_from_relative_humidity(t: xr.DataArray, r: xr.DataArray) -> xr.DataArray:
    r"""Compute the dewpoint temperature from relative humidity.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    r: xarray.DataArray
        Relative humidity (%)

    Returns
    -------
    xarray.DataArray
        Dewpoint temperature (K). For zero ``r`` values returns nan.


    The computation starts with determining the the saturation vapour pressure over
    water at the dewpoint temperature:

    .. math::

        e_{wsat}(td) = \frac{r\; e_{wsat}(t)}{100}

    where:

    * :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water
    * :math:`td` is the dewpoint.

    Then :math:`td` is computed from :math:`e_{wsat}(td)` by inverting the
    equations used in :func:`saturation_vapour_pressure`.

    """
    return xarray_ufunc(array.dewpoint_from_relative_humidity, t, r).assign_attrs(
        {
            "standard_name": "dew_point_temperature",
            "units": "K",
        }
    )


def dewpoint_from_specific_humidity(q: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    r"""Compute the dewpoint temperature from specific humidity.

    Parameters
    ----------
    q: xarray.DataArray
        Specific humidity (kg/kg)
    p: xarray.DataArray
        Pressure (Pa)

    Returns
    -------
    xarray.DataArray
        Dewpoint temperature (K). For zero ``q`` values returns nan.


    The computation starts with determining the the saturation vapour pressure over
    water at the dewpoint temperature:

    .. math::

        e_{wsat}(td) = e(q, p)

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure_from_specific_humidity`)
        * :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water
        * :math:`td` is the dewpoint

    Then :math:`td` is computed from :math:`e_{wsat}(td)` by inverting the equations
    used in :func:`saturation_vapour_pressure`.

    """
    return xarray_ufunc(array.dewpoint_from_specific_humidity, q, p).assign_attrs(
        {
            "standard_name": "dew_point_temperature",
            "units": "K",
        }
    )


def virtual_temperature(t: xr.DataArray, q: xr.DataArray) -> xr.DataArray:
    r"""Compute the virtual temperature from temperature and specific humidity.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)s
    q: xarray.DataArray
        Specific humidity (kg/kg)

    Returns
    -------
    xarray.DataArray
        Virtual temperature (K)


    The computation is based on the following formula [Wallace2006]_:

    .. math::

        t_{v} = t (1 + \frac{1 - \epsilon}{\epsilon} q)

    with :math:`\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).

    """
    return xarray_ufunc(array.virtual_temperature, t, q).assign_attrs(
        {
            "standard_name": "virtual_temperature",
            "units": "K",
        }
    )


def virtual_potential_temperature(t: xr.DataArray, q: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    r"""Compute the virtual potential temperature from temperature and specific humidity.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    q: xarray.DataArray
        Specific humidity (kg/kg)
    p: xarray.DataArray
        Pressure (Pa)

    Returns
    -------
    xarray.DataArray
        Virtual potential temperature (K)


    The computation is based on the following formula:

    .. math::

        \Theta_{v} = \theta (1 + \frac{1 - \epsilon}{\epsilon} q)

    where:

        * :math:`\Theta` is the :func:`potential_temperature`
        * :math:`\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).

    """
    return xarray_ufunc(array.virtual_potential_temperature, t, q, p).assign_attrs(
        {
            "standard_name": "",  # no standard name
            "units": "K",
            "long_name": "Virtual potential temperature",
        }
    )


def potential_temperature(t: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    r"""Compute the potential temperature.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    p: xarray.DataArray
        Pressure (Pa)

    Returns
    -------
    xarray.DataArray
        Potential temperature (K)


    The computation is based on the following formula [Wallace2006]_:

    .. math::

       \theta = t (\frac{10^{5}}{p})^{\kappa}

    with :math:`\kappa = R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`).

    """
    return xarray_ufunc(array.potential_temperature, t, p).assign_attrs(
        {
            "standard_name": "air_potential_temperature",
            "units": "K",
        }
    )


def temperature_from_potential_temperature(th: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    r"""Compute the temperature from potential temperature.

    Parameters
    ----------
    th: xarray.DataArray
        Potential temperature (K)
    p: xarray.DataArray
        Pressure (Pa)

    Returns
    -------
    xarray.DataArray
        Temperature (K)


    The computation is based on the following formula:

    .. math::

       t = \theta (\frac{p}{10^{5}})^{\kappa}

    with :math:`\kappa = R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`).

    """
    return xarray_ufunc(array.temperature_from_potential_temperature, th, p).assign_attrs(
        {
            "standard_name": "air_temperature",
            "units": "K",
        }
    )


def pressure_on_dry_adiabat(t: xr.DataArray, t_def: xr.DataArray, p_def: xr.DataArray) -> xr.DataArray:
    r"""Compute the pressure on a dry adiabat.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature on the dry adiabat (K)
    t_def: xarray.DataArray
        Temperature defining the dry adiabat (K)
    p_def: xarray.DataArray
        Pressure defining the dry adiabat (Pa)

    Returns
    -------
    xarray.DataArray
        Pressure on the dry adiabat (Pa)


    The computation is based on the following formula:

    .. math::

       p = p_{def} (\frac{t}{t_{def}})^{\frac{1}{\kappa}}

    with :math:`\kappa =  R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`).

    """
    return xarray_ufunc(array.pressure_on_dry_adiabat, t, t_def, p_def).assign_attrs(
        {
            "standard_name": "",  # no standard name
            "units": "Pa",
            "long_name": "Pressure on the dry adiabat",
        }
    )


def temperature_on_dry_adiabat(p: xr.DataArray, t_def: xr.DataArray, p_def: xr.DataArray) -> xr.DataArray:
    r"""Compute the temperature on a dry adiabat.

    Parameters
    ----------
    p: xarray.DataArray
        Pressure on the dry adiabat (Pa)
    t_def: xarray.DataArray
        Temperature defining the dry adiabat (K)
    p_def: xarray.DataArray
        Pressure defining the dry adiabat (Pa)

    Returns
    -------
    xarray.DataArray
        Temperature on the dry adiabat (K)


    The computation is based on the following formula:

    .. math::

       t = t_{def} (\frac{p}{p_{def}})^{\kappa}

    with :math:`\kappa =  R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`).

    """
    return xarray_ufunc(array.temperature_on_dry_adiabat, p, t_def, p_def).assign_attrs(
        {
            "standard_name": "",  # no standard name
            "units": "K",
            "long_name": "Temperature on the dry adiabat",
        }
    )


def lcl_temperature(t: xr.DataArray, td: xr.DataArray, method: str = "davies") -> xr.DataArray:
    r"""Compute the Lifting Condenstaion Level (LCL) temperature from dewpoint.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature at the start level (K)
    td: xarray.DataArray
        Dewpoint at the start level (K)
    method: str, optional
        The computation method: "davies" or "bolton".

    Returns
    -------
    xarray.DataArray
        Temperature of the LCL (K)


    The actual computation is based on the ``method``:

    * "davies": the formula by [DaviesJones1983]_ is used (it is also used by the IFS model):

        .. math::

            t_{LCL} =
            td - (0.212 + 1.571\times 10^{-3} (td - t_{0}) - 4.36\times 10^{-4} (t - t_{0})) (t - td)

      where :math:`t_{0}` is the triple point of water (see :data:`earthkit.meteo.constants.T0`).

    * "bolton": the formula by [Bolton1980]_ is used:

        .. math::

            t_{LCL} = 56.0 +  \frac{1}{\frac{1}{td - 56} + \frac{log(\frac{t}{td})}{800}}

    """
    return xarray_ufunc(array.lcl_temperature, t, td, method=method)


def lcl(
    t: xr.DataArray, td: xr.DataArray, p: xr.DataArray, method: str = "davies"
) -> tuple[xr.DataArray, xr.DataArray]:
    r"""Compute the temperature and pressure of the Lifting Condenstaion Level (LCL) from dewpoint.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature at the start level (K)
    td: xarray.DataArray
        Dewpoint at the start level (K)
    p: xarray.DataArray
        Pressure at the start level (Pa)
        method: str
    method: str, optional
        The computation method: "davies" or "bolton".

    Returns
    -------
    xarray.DataArray
        Temperature of the LCL (K)
    xarray.DataArray
        Pressure of the LCL (Pa)


    The LCL temperature is determined by :func:`lcl_temperature` with the given ``method``
    and the pressure is computed with :math:`t_{LCL}` using :func:`pressure_on_dry_adiabat`.

    """
    return xarray_ufunc(array.lcl, t, td, p, method=method)


def ept_from_dewpoint(
    t: xr.DataArray, td: xr.DataArray, p: xr.DataArray, method: str = "ifs"
) -> xr.DataArray:
    r"""Compute the equivalent potential temperature from dewpoint.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    td: xarray.DataArray
        Dewpoint (K)
    p: xarray.DataArray
        Pressure (Pa)
    method: str, optional
        Specifies the computation method. The possible values are: "ifs", "bolton35", "bolton39".

    Returns
    -------
    xarray.DataArray
        Equivalent potential temperature (K)


    The actual computation is based on the value of ``method``:

    * "ifs": the formula from the IFS model [IFS-CY47R3-PhysicalProcesses]_ (Chapter 6.11) is used:

        .. math::

            \Theta_{e} = \Theta\; exp(\frac{L_{v}\; q}{c_{pd}\; t_{LCL}})

    * "bolton35": Eq (35) from [Bolton1980]_ is used:


        .. math::

            \Theta_{e} = \Theta (\frac{10^{5}}{p})^{\kappa 0.28 w} exp(\frac{2675 w}{t_{LCL}})

    * "bolton39": Eq (39) from [Bolton1980]_ is used:

        .. math::

            \Theta_{e} =
            t (\frac{10^{5}}{p-e})^{\kappa} (\frac{t}{t_{LCL}})^{0.28 w} exp[(\frac{3036}{t_{LCL}} -
            1.78)w(1+0.448\; w)]

    where:

        * :math:`\Theta` is the :func:`potential_temperature`
        * :math:`t_{LCL}` is the temperature at the Lifting Condestation Level computed
          with :func:`lcl_temperature` using option:

            * method="davis" when ``method`` is "ifs"
            * method="bolton" when ``method`` is "bolton35" or "bolton39"
        * :math:`q` is the specific humidity computed with :func:`specific_humidity_from_dewpoint`
        * :math:`w`: is the mixing ratio computed with :func:`mixing_ratio_from_dewpoint`
        * :math:`e` is the vapour pressure computed with :func:`vapour_pressure_from_mixing_ratio`
        * :math:`L_{v}`: is the latent heat of vaporisation
          (see :data:`earthkit.meteo.constants.Lv`)
        * :math:`c_{pd}` is the specific heat of dry air on constant pressure
          (see :data:`earthkit.meteo.constants.c_pd`)
        * :math:`\kappa = R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`)

    """
    return xarray_ufunc(array.ept_from_dewpoint, t, td, p, method=method)


def ept_from_specific_humidity(
    t: xr.DataArray, q: xr.DataArray, p: xr.DataArray, method: str = "ifs"
) -> xr.DataArray:
    r"""Compute the equivalent potential temperature from specific humidity.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    q: xarray.DataArray
        Specific humidity (kg/kg)
    p: xarray.DataArray
        Pressure (Pa)
    method: str, optional
        Specifies the computation method. The possible values are: "ifs",
        "bolton35", "bolton39. See :func:`ept_from_dewpoint` for details.

    Returns
    -------
    xarray.DataArray
        Equivalent potential temperature (K)


    The computations are the same as in :func:`ept_from_dewpoint`
    (the dewpoint is computed from q with :func:`dewpoint_from_specific_humidity`).

    """
    return xarray_ufunc(array.ept_from_specific_humidity, t, q, p, method=method)


def saturation_ept(t: xr.DataArray, p: xr.DataArray, method: str = "ifs") -> xr.DataArray:
    r"""Compute the saturation equivalent potential temperature.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    p: xarray.DataArray
        Pressure (Pa)
    method: str, optional
        Specifies the computation method. The possible values are: "ifs", "bolton35", "bolton39".

    Returns
    -------
    xarray.DataArray
        Saturation equivalent potential temperature (K)


    The actual computation is based on the ``method``:

    * "ifs": The formula is based on the equivalent potential temperature definition used
       in the IFS model [IFS-CY47R3-PhysicalProcesses]_ (see Chapter 6.11) :

        .. math::

            \Theta_{esat} = \Theta\; exp(\frac{L_{v}\; q_{sat}}{c_{pd}\; t})

    * "bolton35": Eq (35) from [Bolton1980]_ is used:

        .. math::

            \Theta_{e} = \Theta (\frac{10^{5}}{p})^{\kappa 0.28 w_{sat}}\; exp(\frac{2675\; w_{sat}}{t})

    * "bolton39": Eq (39) from [Bolton1980]_ is used:

        .. math::

            \Theta_{e} =
            t (\frac{10^{5}}{p-e_{sat}})^{\kappa} exp[(\frac{3036}{t} - 1.78)w_{sat}(1+0.448\; w_{sat})]

    where:

        * :math:`\Theta` is the :func:`potential_temperature`
        * :math:`e_{sat}` is the :func:`saturation_vapor_pressure`
        * :math:`q_{sat}` is the :func:`saturation_specific_humidity`
        * :math:`w_{sat}` is the :func:`saturation_mixing_ratio`
        * :math:`L_{v}` is the specific latent heat of vaporization (see :data:`earthkit.meteo.constants.Lv`)
        * :math:`c_{pd}` is the specific heat of dry air on constant pressure
          (see :data:`earthkit.meteo.constants.c_pd`)

    """
    return xarray_ufunc(array.saturation_ept, t, p, method=method)


def temperature_on_moist_adiabat(
    ept: xr.DataArray,
    p: xr.DataArray,
    ept_method: str = "ifs",
    t_method: str = "bisect",
) -> xr.DataArray:
    r"""Compute the temperature on a moist adiabat (pseudoadiabat)

    Parameters
    ----------
    ept: xarray.DataArray
        Equivalent potential temperature defining the moist adiabat (K)
    p: xarray.DataArray
        Pressure on the moist adiabat (Pa)
    ept_method: str, optional
        Specifies the computation method that was used to compute ``ept``. The possible
        values are: "ifs", "bolton35", "bolton39".
        (See :func:`ept_from_dewpoint` for details.)
    t_method: str, optional
        Specifies the iteration method along the moist adiabat to find the temperature
        for the given ``p`` pressure. The possible values are as follows:

        * "bisect": a bisection method is used as defined in [Stipanuk1973]_
        * "newton": Newtons's method is used as defined by Eq (2.6) in [DaviesJones2008]_.
          For extremely hot and humid conditions (``ept`` > 800 K) depending on
          ``ept_method`` the computation might not be carried out
          and nan will be returned.


    Returns
    -------
    xarray.DataArray
        Temperature on the moist adiabat (K). For values where the computation cannot
        be carried out nan is returned.

    """
    return xarray_ufunc(
        array.temperature_on_moist_adiabat,
        ept,
        p,
        ept_method=ept_method,
        t_method=t_method,
    ).assign_attrs(
        {
            "standard_name": "",  # no standard name
            "units": "K",
            "long_name": "Temperature on the moist adiabat",
        }
    )


def wet_bulb_temperature_from_dewpoint(
    t: xr.DataArray,
    td: xr.DataArray,
    p: xr.DataArray,
    ept_method: str = "ifs",
    t_method: str = "bisect",
) -> xr.DataArray:
    r"""Compute the pseudo adiabatic wet bulb temperature from dewpoint.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    td: xarray.DataArray
        Dewpoint (K)
    p: xarray.DataArray
        Pressure (Pa)
    ept_method: str, optional
        Specifies the computation method for the equivalent potential temperature.
        The possible values are: "ifs", "bolton35", "bolton39".
        (See :func:`ept_from_dewpoint` for details.)
    t_method: str, optional
        Specifies the method to find the temperature along the moist adiabat defined
        by the equivalent potential temperature. The possible values are as follows:

        * "bisect": :func:`temperature_on_moist_adiabat` with ``t_method`` = "bisect" is used
        * "newton": :func:`temperature_on_moist_adiabat` with ``t_method`` = "newton" is used

    Returns
    -------
    xarray.DataArray
        Wet bulb temperature (K)


    The computation is based on Normand's rule [Wallace2006]_ (Chapter 3.5.6):

    * first the equivalent potential temperature is computed with the given
      ``ept_method`` (using :func:`ept_from_dewpoint`). This defines the moist adiabat.
    * then the wet bulb potential temperature is determined as the temperature at
      pressure ``p`` on the moist adiabat with the given ``t_method``.

    """
    return xarray_ufunc(
        array.wet_bulb_temperature_from_dewpoint,
        t,
        td,
        p,
        ept_method=ept_method,
        t_method=t_method,
    ).assign_attrs(
        {
            "standard_name": "wet_bulb_temperature",
            "units": "K",
        }
    )


def wet_bulb_temperature_from_specific_humidity(
    t: xr.DataArray,
    q: xr.DataArray,
    p: xr.DataArray,
    ept_method: str = "ifs",
    t_method: str = "bisect",
) -> xr.DataArray:
    r"""Compute the pseudo adiabatic wet bulb temperature from specific humidity.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    q: xarray.DataArray
        Specific humidity (kg/kg)
    p: xarray.DataArray
        Pressure (Pa)
    ept_method: str, optional
        Specifies the computation method for the equivalent potential temperature.
        The possible values are: "ifs", "bolton35", "bolton39".
        (See :func:`ept_from_dewpoint` for details.)
    t_method: str, optional
        Specifies the method to find the temperature along the moist adiabat
        defined by the equivalent potential temperature. The possible values are
        as follows:

        * "bisect": :func:`temperature_on_moist_adiabat` with ``t_method`` = "bisect" is used
        * "newton": :func:`temperature_on_moist_adiabat` with ``t_method`` = "newton" is used

    Returns
    -------
    xarray.DataArray
        Wet bulb temperature (K)


    The computation is based on Normand's rule [Wallace2006]_ (Chapter 3.5.6):

    * first the equivalent potential temperature is computed with the given
      ``ept_method`` (using :func:`ept_from_dewpoint`). This defines the moist adiabat.
    * then the wet bulb potential temperature is determined as the temperature at
      pressure ``p`` on the moist adiabat with the given ``t_method``.

    """
    return xarray_ufunc(
        array.wet_bulb_temperature_from_specific_humidity,
        t,
        q,
        p,
        ept_method=ept_method,
        t_method=t_method,
    ).assign_attrs(
        {
            "standard_name": "wet_bulb_temperature",
            "units": "K",
        }
    )


def wet_bulb_potential_temperature_from_dewpoint(
    t: xr.DataArray,
    td: xr.DataArray,
    p: xr.DataArray,
    ept_method: str = "ifs",
    t_method: str = "direct",
) -> xr.DataArray:
    r"""Compute the pseudo adiabatic wet bulb potential temperature from dewpoint.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    td: xarray.DataArray
        Dewpoint (K)
    p: xarray.DataArray
        Pressure (Pa)
    ept_method: str, optional
        Specifies the computation method for the equivalent potential temperature.
        The possible values are: "ifs", "bolton35", "bolton39".
        (See :func:`ept_from_dewpoint` for details.)
    t_method: str, optional
        Specifies the method to find the temperature along the moist adiabat defined
        by the equivalent potential temperature. The possible values are as follows:

        * "direct": the rational formula defined by Eq (3.8) in [DaviesJones2008]_ is used
        * "bisect": :func:`temperature_on_moist_adiabat` with ``t_method`` = "bisect" is used
        * "newton": :func:`temperature_on_moist_adiabat` with ``t_method`` = "newton" is used

    Returns
    -------
    xarray.DataArray
        Wet bulb potential temperature (K)


    The computation is based on Normand's rule [Wallace2006]_ (Chapter 3.5.6):

    * first the equivalent potential temperature is computed with the given
      ``ept_method`` (using :func:`ept_from_dewpoint`). This defines the moist adiabat.
    * then the wet bulb potential temperature is determined as the temperature at
      pressure :math:`10^{5}` Pa on the moist adiabat with the given ``t_method``.

    """
    return xarray_ufunc(
        array.wet_bulb_potential_temperature_from_dewpoint,
        t,
        td,
        p,
        ept_method=ept_method,
        t_method=t_method,
    ).assign_attrs(
        {
            "standard_name": "wet_bulb_potential_temperature",
            "units": "K",
        }
    )


def wet_bulb_potential_temperature_from_specific_humidity(
    t: xr.DataArray,
    q: xr.DataArray,
    p: xr.DataArray,
    ept_method: str = "ifs",
    t_method: str = "direct",
) -> xr.DataArray:
    r"""Compute the pseudo adiabatic wet bulb potential temperature from specific humidity.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    q: xarray.DataArray
        Specific humidity (kg/kg)
    p: xarray.DataArray
        Pressure (Pa)
    ept_method: str, optional
        Specifies the computation method for the equivalent potential temperature.
        The possible values are: "ifs", "bolton35", "bolton39".
        (See :func:`ept_from_dewpoint` for details.)
    t_method: str, optional
        Specifies the method to find the temperature along the moist adiabat
        defined by the equivalent potential temperature. The possible values are as follows:

        * "direct": the rational formula defined by Eq (3.8) in [DaviesJones2008]_ is used
        * "bisect": :func:`temperature_on_moist_adiabat` with ``t_method`` = "bisect" is used
        * "newton": :func:`temperature_on_moist_adiabat` with ``t_method`` = "newton" is used

    Returns
    -------
    xarray.DataArray
        Wet bulb potential temperature (K)


    The computations are the same as in
    :func:`wet_bulb_potential_temperature_from_dewpoint`
    (the dewpoint is computed from q with :func:`dewpoint_from_specific_humidity`).

    """
    return xarray_ufunc(
        array.wet_bulb_potential_temperature_from_specific_humidity,
        t,
        q,
        p,
        ept_method=ept_method,
        t_method=t_method,
    ).assign_attrs(
        {
            "standard_name": "wet_bulb_potential_temperature",
            "units": "K",
        }
    )


def specific_gas_constant(q: xr.DataArray) -> xr.DataArray:
    r"""Compute the specific gas constant of moist air.

    Specific content of cloud particles and hydrometeors are neglected.

    Parameters
    ----------
    q: xarray.DataArray
        Specific humidity (kg/kg)

    Returns
    -------
    xarray.DataArray
        Specific gas constant of moist air (J kg-1 K-1)


    The computation is based on the following formula:

    .. math::

        R = R_{d} + (R_{v} - R_{d}) q

    where:

        * :math:`R_{d}` is the gas constant for dry air (see :data:`earthkit.meteo.constants.Rd`)
        * :math:`R_{v}` is the gas constant for water vapour (see :data:`earthkit.meteo.constants.Rv`)

    """
    return xarray_ufunc(array.specific_gas_constant, q).assign_attrs(
        {
            "standard_name": "",  # no standard name
            "units": "J kg-1 K-1",
            "long_name": "Specific gas constant of moist air",
        }
    )
