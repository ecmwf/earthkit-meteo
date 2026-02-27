# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import overload

from earthkit.meteo.utils.decorators import dispatch

if TYPE_CHECKING:
    import xarray  # type: ignore[import]


@overload
def celsius_to_kelvin(t: "xarray.DataArray") -> "xarray.DataArray": ...


def celsius_to_kelvin(t: "xarray.DataArray") -> "xarray.DataArray":
    r"""Convert temperature values from Celsius to Kelvin.

    Parameters
    ----------
    t : xarray.DataArray
        Temperature in Celsius units

    Returns
    -------
    xarray.DataArray
        Temperature in Kelvin units


    Implementations
    ------------------------
    :func:`celsius_to_kelvin` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.celsius_to_kelvin` for xarray.DataArray

    """
    return dispatch(celsius_to_kelvin, t)


@overload
def kelvin_to_celsius(t: "xarray.DataArray") -> "xarray.DataArray": ...


def kelvin_to_celsius(t: "xarray.DataArray") -> "xarray.DataArray":
    r"""Convert temperature values from Kelvin to Celsius.

    Parameters
    ----------
    t : xarray.DataArray
        Temperature in Kelvin units

    Returns
    -------
    xarray.DataArray
        Temperature in Celsius units


    Implementations
    ------------------------
    :func:`kelvin_to_celsius` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.kelvin_to_celsius` for xarray.DataArray

    """
    return dispatch(kelvin_to_celsius, t)


@overload
def specific_humidity_from_mixing_ratio(w: "xarray.DataArray") -> "xarray.DataArray": ...


def specific_humidity_from_mixing_ratio(w: "xarray.DataArray") -> "xarray.DataArray":
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


    Implementations
    ------------------------
    :func:`specific_humidity_from_mixing_ratio` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.specific_humidity_from_mixing_ratio` for xarray.DataArray

    """
    return dispatch(specific_humidity_from_mixing_ratio, w)


@overload
def mixing_ratio_from_specific_humidity(q: "xarray.DataArray") -> "xarray.DataArray": ...


def mixing_ratio_from_specific_humidity(q: "xarray.DataArray") -> "xarray.DataArray":
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


    Implementations
    ------------------------
    :func:`mixing_ratio_from_specific_humidity` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.mixing_ratio_from_specific_humidity` for xarray.DataArray

    """
    return dispatch(mixing_ratio_from_specific_humidity, q)


@overload
def vapour_pressure_from_specific_humidity(
    q: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray": ...


def vapour_pressure_from_specific_humidity(
    q: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray":
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


    Implementations
    ------------------------
    :func:`vapour_pressure_from_specific_humidity` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.vapour_pressure_from_specific_humidity` for xarray.DataArray

    """
    return dispatch(vapour_pressure_from_specific_humidity, q, p)


@overload
def vapour_pressure_from_mixing_ratio(
    w: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray": ...


def vapour_pressure_from_mixing_ratio(
    w: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray":
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


    Implementations
    ------------------------
    :func:`vapour_pressure_from_mixing_ratio` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.vapour_pressure_from_mixing_ratio` for xarray.DataArray

    """
    return dispatch(vapour_pressure_from_mixing_ratio, w, p)


@overload
def specific_humidity_from_vapour_pressure(
    e: "xarray.DataArray",
    p: "xarray.DataArray",
    eps: float = 1e-4,
) -> "xarray.DataArray": ...


def specific_humidity_from_vapour_pressure(
    e: "xarray.DataArray",
    p: "xarray.DataArray",
    eps: float = 1e-4,
) -> "xarray.DataArray":
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


    Implementations
    ------------------------
    :func:`specific_humidity_from_vapour_pressure` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.specific_humidity_from_vapour_pressure` for xarray.DataArray

    """
    return dispatch(specific_humidity_from_vapour_pressure, e, p, eps=eps)


@overload
def mixing_ratio_from_vapour_pressure(
    e: "xarray.DataArray",
    p: "xarray.DataArray",
    eps: float = 1e-4,
) -> "xarray.DataArray": ...


def mixing_ratio_from_vapour_pressure(
    e: "xarray.DataArray",
    p: "xarray.DataArray",
    eps: float = 1e-4,
) -> "xarray.DataArray":
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


    Implementations
    ------------------------
    :func:`mixing_ratio_from_vapour_pressure` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.mixing_ratio_from_vapour_pressure` for xarray.DataArray

    """
    return dispatch(mixing_ratio_from_vapour_pressure, e, p, eps=eps)


@overload
def saturation_vapour_pressure(t: "xarray.DataArray", phase: str = "mixed") -> "xarray.DataArray": ...


def saturation_vapour_pressure(
    t: "xarray.DataArray",
    phase: str = "mixed",
) -> "xarray.DataArray":
    r"""Compute the saturation vapour pressure from temperature with respect to a phase.

    Parameters
    ----------
    t: array-like
        Temperature (K)
    phase: str, optional
        Define the phase with respect to the saturation vapour pressure is computed.
        It is either “water”, “ice” or “mixed”.

    Returns
    -------
    array-like
        Saturation vapour pressure (Pa)


    The algorithm was taken from the IFS model [IFS-CY47R3-PhysicalProcesses]_ (see Chapter 12).
    It uses the following formula when ``phase`` is \"water\" or \"ice\":

    .. math::

        e_{sat} = a_{1}\\;exp \\left(a_{3}\\frac{t-273.16}{t-a_{4}}\\right)

    where the parameters are set as follows:

    * ``phase`` = \"water\": :math:`a_{1}` =611.21 Pa, :math:`a_{3}` =17.502 and :math:`a_{4}` =32.19 K
    * ``phase`` = \"ice\": :math:`a_{1}` =611.21 Pa, :math:`a_{3}` =22.587 and :math:`a_{4}` =-0.7 K

    When ``phase`` is \"mixed\" the formula is based on the value of ``t``:

    * if :math:`t <= t_{i}`: the formula for ``phase`` = \"ice\" is used (:math:`t_{i} = 250.16 K`)
    * if :math:`t >= t_{0}`: the formula for ``phase`` = \"water\" is used (:math:`t_{0} = 273.16 K`)
    * for the range :math:`t_{i} < t < t_{0}` an interpolation is used between the \"ice\" and \"water\" phases:

    .. math::

        \\alpha(t) e_{wsat}(t) + (1 - \\alpha(t)) e_{isat}(t)

    with :math:`\\alpha(t) = (\\frac{t-t_{i}}{t_{0}-t_{i}})^2`.


    Implementations
    ------------------------
    :func:`saturation_vapour_pressure` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.saturation_vapour_pressure` for xarray.DataArray

    """
    return dispatch(saturation_vapour_pressure, t, phase=phase)


@overload
def saturation_mixing_ratio(
    t: "xarray.DataArray",
    p: "xarray.DataArray",
    phase: str = "mixed",
) -> "xarray.DataArray": ...


def saturation_mixing_ratio(
    t: "xarray.DataArray",
    p: "xarray.DataArray",
    phase: str = "mixed",
) -> "xarray.DataArray":
    r"""Compute the saturation mixing ratio from temperature with respect to a phase.

    Parameters
    ----------
    t: array-like
        Temperature (K)
    p: array-like
        Pressure (Pa)
    phase: str
        Define the phase with respect to the :func:`saturation_vapour_pressure` is computed.
        It is either “water”, “ice” or “mixed”.

    Returns
    -------
    array-like
        Saturation mixing ratio (kg/kg)


    Equivalent to the following code:

    .. code-block:: python

        e = saturation_vapour_pressure(t, phase=phase)
        return mixing_ratio_from_vapour_pressure(e, p)


    Implementations
    ------------------------
    :func:`saturation_mixing_ratio` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.saturation_mixing_ratio` for xarray.DataArray

    """
    return dispatch(saturation_mixing_ratio, t, p, phase=phase)


@overload
def saturation_specific_humidity(
    t: "xarray.DataArray",
    p: "xarray.DataArray",
    phase: str = "mixed",
) -> "xarray.DataArray": ...


def saturation_specific_humidity(
    t: "xarray.DataArray",
    p: "xarray.DataArray",
    phase: str = "mixed",
) -> "xarray.DataArray":
    r"""Compute the saturation specific humidity from temperature with respect to a phase.

    Parameters
    ----------
    t: array-like
        Temperature (K)
    p: array-like
        Pressure (Pa)
    phase: str, optional
        Define the phase with respect to the :func:`saturation_vapour_pressure` is computed.
        It is either “water”, “ice” or “mixed”.

    Returns
    -------
    array-like
        Saturation specific humidity (kg/kg)


    Equivalent to the following code:

    .. code-block:: python

        e = saturation_vapour_pressure(t, phase=phase)
        return specific_humidity_from_vapour_pressure(e, p)


    Implementations
    ------------------------
    :func:`saturation_specific_humidity` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.saturation_specific_humidity` for xarray.DataArray

    """
    return dispatch(saturation_specific_humidity, t, p, phase=phase)


@overload
def saturation_vapour_pressure_slope(
    t: "xarray.DataArray",
    phase: str = "mixed",
) -> "xarray.DataArray": ...


def saturation_vapour_pressure_slope(
    t: "xarray.DataArray",
    phase: str = "mixed",
) -> "xarray.DataArray":
    r"""Compute the slope of saturation vapour pressure with respect to temperature.

    Parameters
    ----------
    t: array-like
        Temperature (K)
    phase: str, optional
        Define the phase with respect to the computation will be performed.
        It is either “water”, “ice” or “mixed”. See :func:`saturation_vapour_pressure`
        for details.

    Returns
    -------
    array-like
        Slope of saturation vapour pressure (Pa/K)


    Implementations
    ------------------------
    :func:`saturation_vapour_pressure_slope` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.saturation_vapour_pressure_slope` for xarray.DataArray

    """
    return dispatch(saturation_vapour_pressure_slope, t, phase=phase)


@overload
def saturation_mixing_ratio_slope(
    t: "xarray.DataArray",
    p: "xarray.DataArray",
    es: "xarray.DataArray" | None = None,
    es_slope: "xarray.DataArray" | None = None,
    phase: str = "mixed",
    eps: float = 1e-4,
) -> "xarray.DataArray": ...


def saturation_mixing_ratio_slope(
    t: "xarray.DataArray",
    p: "xarray.DataArray",
    es: "xarray.DataArray" | None = None,
    es_slope: "xarray.DataArray" | None = None,
    phase: str = "mixed",
    eps: float = 1e-4,
) -> "xarray.DataArray":
    r"""Compute the slope of saturation mixing ratio with respect to temperature.

    Parameters
    ----------
    t: array-like
        Temperature (K)
    p: array-like
        Pressure (Pa)
    es: array-like or None, optional
        :func:`saturation_vapour_pressure` pre-computed for the given ``phase`` (Pa)
    es_slope: array-like or None, optional
        :func:`saturation_vapour_pressure_slope` pre-computed for the given ``phase`` (Pa/K)
    phase: str, optional
        Define the phase with respect to the computation will be performed.
        It is either “water”, “ice” or “mixed”. See :func:`saturation_vapour_pressure`
        for details.
    eps: number
        Where p - es < ``eps`` nan is returned.

    Returns
    -------
    array-like
        Slope of saturation mixing ratio (:math:`kg kg^{-1} K^{-1}`)


    The computation is based on the following formula:

    .. math::

        \\frac{\\partial w_{s}}{\\partial t} = \\frac{\\epsilon\\; p}{(p-e_{s})^{2}} \\frac{d e_{s}}{d t}

    where

        * :math:`\\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).
        * :math:`e_{s}` is the :func:`saturation_vapour_pressure` for the given ``phase``


    Implementations
    ------------------------
    :func:`saturation_mixing_ratio_slope` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.saturation_mixing_ratio_slope` for xarray.DataArray

    """
    return dispatch(
        saturation_mixing_ratio_slope,
        t,
        p,
        es,
        es_slope,
        phase=phase,
        eps=eps,
    )


@overload
def saturation_specific_humidity_slope(
    t: "xarray.DataArray",
    p: "xarray.DataArray",
    es: "xarray.DataArray" | None = None,
    es_slope: "xarray.DataArray" | None = None,
    phase: str = "mixed",
    eps: float = 1e-4,
) -> "xarray.DataArray": ...


def saturation_specific_humidity_slope(
    t: "xarray.DataArray",
    p: "xarray.DataArray",
    es: "xarray.DataArray" | None = None,
    es_slope: "xarray.DataArray" | None = None,
    phase: str = "mixed",
    eps: float = 1e-4,
) -> "xarray.DataArray":
    r"""Compute the slope of saturation specific humidity with respect to temperature.

    Parameters
    ----------
    t: array-like
        Temperature (K)
    p: array-like
        Pressure (Pa)
    es: array-like or None, optional
        :func:`saturation_vapour_pressure` pre-computed for the given ``phase`` (Pa)
    es_slope: array-like or None, optional
        :func:`saturation_vapour_pressure_slope` pre-computed for the given ``phase`` (Pa/K)
    phase: str, optional
        Define the phase with respect to the computation will be performed.
        It is either “water”, “ice” or “mixed”. See :func:`saturation_vapour_pressure`
        for details.
    eps: number
        Where p - es < ``eps`` nan is returned.

    Returns
    -------
    array-like
        Slope of saturation specific humidity (:math:`kg kg^{-1} K^{-1}`)


    The computation is based on the following formula:

    .. math::

        \\frac{\\partial q_{s}}{\\partial t} =
        \\frac{\\epsilon\\; p}{(p+e_{s}(\\epsilon - 1))^{2}} \\frac{d e_{s}}{d t}

    where

        * :math:`\\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).
        * :math:`e_{s}` is the :func:`saturation_vapour_pressure` for the given ``phase``


    Implementations
    ------------------------
    :func:`saturation_specific_humidity_slope` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.saturation_specific_humidity_slope` for xarray.DataArray

    """
    return dispatch(
        saturation_specific_humidity_slope,
        t,
        p,
        es,
        es_slope,
        phase=phase,
        eps=eps,
    )


@overload
def temperature_from_saturation_vapour_pressure(
    es: "xarray.DataArray",
) -> "xarray.DataArray": ...


def temperature_from_saturation_vapour_pressure(
    es: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the temperature from saturation vapour pressure.

    Parameters
    ----------
    es: array-like
        :func:`saturation_vapour_pressure` (Pa)

    Returns
    -------
    array-like
        Temperature (K). For zero ``es`` values returns nan.


    The computation is always based on the "water" phase of
    the :func:`saturation_vapour_pressure` formulation irrespective of the
    phase ``es`` was computed to.


    Implementations
    ------------------------
    :func:`temperature_from_saturation_vapour_pressure` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.temperature_from_saturation_vapour_pressure` for xarray.DataArray

    """
    return dispatch(temperature_from_saturation_vapour_pressure, es)


@overload
def relative_humidity_from_dewpoint(
    t: "xarray.DataArray",
    td: "xarray.DataArray",
) -> "xarray.DataArray": ...


def relative_humidity_from_dewpoint(
    t: "xarray.DataArray",
    td: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the relative humidity from dewpoint temperature.

    Parameters
    ----------
    t: array-like
        Temperature (K)
    td: array-like
        Dewpoint (K)

    Returns
    -------
    array-like
        Relative humidity (%)


    The computation is based on the following formula:

    .. math::

        r = 100 \\frac {e_{wsat}(td)}{e_{wsat}(t)}

    where :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water.


    Implementations
    ------------------------
    :func:`relative_humidity_from_dewpoint` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.relative_humidity_from_dewpoint` for xarray.DataArray

    """
    return dispatch(relative_humidity_from_dewpoint, t, td)


@overload
def relative_humidity_from_specific_humidity(
    t: "xarray.DataArray",
    q: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray": ...


def relative_humidity_from_specific_humidity(
    t: "xarray.DataArray",
    q: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the relative humidity from specific humidity.

    Parameters
    ----------
    t: array-like
        Temperature (K)
    q: array-like
        Specific humidity (kg/kg)
    p: array-like
        Pressure (Pa)

    Returns
    -------
    array-like
        Relative humidity (%)


    The computation is based on the following formula:

    .. math::

        r = 100 \\frac {e(q, p)}{e_{msat}(t)}

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure_from_specific_humidity`)
        * :math:`e_{msat}` is the :func:`saturation_vapour_pressure` based on the \"mixed\" phase


    Implementations
    ------------------------
    :func:`relative_humidity_from_specific_humidity` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.relative_humidity_from_specific_humidity` for xarray.DataArray

    """
    return dispatch(relative_humidity_from_specific_humidity, t, q, p)


@overload
def specific_humidity_from_dewpoint(
    td: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray": ...


def specific_humidity_from_dewpoint(
    td: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the specific humidity from dewpoint.

    Parameters
    ----------
    td: array-like
        Dewpoint (K)
    p: array-like
        Pressure (Pa)

    Returns
    -------
    array-like
        Specific humidity (kg/kg)


    The computation starts with determining the vapour pressure:

    .. math::

        e(q, p) = e_{wsat}(td)

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure_from_specific_humidity`)
        * :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water
        * :math:`q` is the specific humidity

    Then `q` is computed from :math:`e` using :func:`specific_humidity_from_vapour_pressure`.


    Implementations
    ------------------------
    :func:`specific_humidity_from_dewpoint` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.specific_humidity_from_dewpoint` for xarray.DataArray

    """
    return dispatch(specific_humidity_from_dewpoint, td, p)


@overload
def mixing_ratio_from_dewpoint(
    td: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray": ...


def mixing_ratio_from_dewpoint(
    td: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the mixing ratio from dewpoint.

    Parameters
    ----------
    td: array-like
        Dewpoint (K)
    p: array-like
        Pressure (Pa)

    Returns
    -------
    array-like
        Specific humidity (kg/kg)


    The computation starts with determining the vapour pressure:

    .. math::

        e(w, p) = e_{wsat}(td)

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure_from_mixing_ratio`)
        * :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water
        * :math:`w` is the mixing ratio

    Then `w` is computed from :math:`e` using :func:`mixing_ratio_from_vapour_pressure`.


    Implementations
    ------------------------
    :func:`mixing_ratio_from_dewpoint` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.mixing_ratio_from_dewpoint` for xarray.DataArray

    """
    return dispatch(mixing_ratio_from_dewpoint, td, p)


@overload
def specific_humidity_from_relative_humidity(
    t: "xarray.DataArray",
    r: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray": ...


def specific_humidity_from_relative_humidity(
    t: "xarray.DataArray",
    r: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the specific humidity from relative_humidity.

    Parameters
    ----------
    t: array-like
        Temperature (K)
    r: array-like
        Relative humidity(%)
    p: array-like
        Pressure (Pa)

    Returns
    -------
    array-like
        Specific humidity (kg/kg) units


    The computation starts with determining the the vapour pressure:

    .. math::

        e(q, p) = r\; \frac{e_{msat}(t)}{100}

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure`)
        * :math:`e_{msat}` is the :func:`saturation_vapour_pressure` based on the \"mixed\" phase
        * :math:`q` is the specific humidity

    Then :math:`q` is computed from :math:`e` using :func:`specific_humidity_from_vapour_pressure`.


    Implementations
    ------------------------
    :func:`specific_humidity_from_relative_humidity` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.specific_humidity_from_relative_humidity` for xarray.DataArray

    """
    return dispatch(specific_humidity_from_relative_humidity, t, r, p)


@overload
def dewpoint_from_relative_humidity(
    t: "xarray.DataArray",
    r: "xarray.DataArray",
) -> "xarray.DataArray": ...


def dewpoint_from_relative_humidity(
    t: "xarray.DataArray",
    r: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the dewpoint temperature from relative humidity.

    Parameters
    ----------
    t: array-like
        Temperature (K)
    r: array-like
        Relative humidity (%)

    Returns
    -------
    array-like
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


    Implementations
    ------------------------
    :func:`dewpoint_from_relative_humidity` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.dewpoint_from_relative_humidity` for xarray.DataArray

    """
    return dispatch(dewpoint_from_relative_humidity, t, r)


@overload
def dewpoint_from_specific_humidity(
    q: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray": ...


def dewpoint_from_specific_humidity(
    q: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the dewpoint temperature from specific humidity.

    Parameters
    ----------
    q: array-like
        Specific humidity (kg/kg)
    p: array-like
        Pressure (Pa)

    Returns
    -------
    array-like
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


    Implementations
    ------------------------
    :func:`dewpoint_from_specific_humidity` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.dewpoint_from_specific_humidity` for xarray.DataArray

    """
    return dispatch(dewpoint_from_specific_humidity, q, p)


@overload
def virtual_temperature(
    t: "xarray.DataArray",
    q: "xarray.DataArray",
) -> "xarray.DataArray": ...


def virtual_temperature(
    t: "xarray.DataArray",
    q: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the virtual temperature from temperature and specific humidity.

    Parameters
    ----------
    t: number or array-like
        Temperature (K)s
    q: number or array-like
        Specific humidity (kg/kg)

    Returns
    -------
    number or array-like
        Virtual temperature (K)


    The computation is based on the following formula [Wallace2006]_:

    .. math::

        t_{v} = t (1 + \frac{1 - \epsilon}{\epsilon} q)

    with :math:`\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).


    Implementations
    ------------------------
    :func:`virtual_temperature` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.virtual_temperature` for xarray.DataArray

    """
    return dispatch(virtual_temperature, t, q)


@overload
def virtual_potential_temperature(
    t: "xarray.DataArray",
    q: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray": ...


def virtual_potential_temperature(
    t: "xarray.DataArray",
    q: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the virtual potential temperature from temperature and specific humidity.

    Parameters
    ----------
    t: number or array-like
        Temperature (K)
    q: number or array-like
        Specific humidity (kg/kg)
    p: number or array-like
        Pressure (Pa)

    Returns
    -------
    number or array-like
        Virtual potential temperature (K)


    The computation is based on the following formula:

    .. math::

        \Theta_{v} = \theta (1 + \frac{1 - \epsilon}{\epsilon} q)

    where:

        * :math:`\Theta` is the :func:`potential_temperature`
        * :math:`\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).


    Implementations
    ------------------------
    :func:`virtual_potential_temperature` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.virtual_potential_temperature` for xarray.DataArray

    """
    return dispatch(virtual_potential_temperature, t, q, p)


@overload
def potential_temperature(
    t: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray": ...


def potential_temperature(
    t: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the potential temperature.

    Parameters
    ----------
    t: number or array-like
        Temperature (K)
    p: number or array-like
        Pressure (Pa)

    Returns
    -------
    number or array-like
        Potential temperature (K)


    The computation is based on the following formula [Wallace2006]_:

    .. math::

       \theta = t (\frac{10^{5}}{p})^{\kappa}

    with :math:`\kappa = R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`).


    Implementations
    ------------------------
    :func:`potential_temperature` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.potential_temperature` for xarray.DataArray

    """
    return dispatch(potential_temperature, t, p)


@overload
def temperature_from_potential_temperature(
    th: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray": ...


def temperature_from_potential_temperature(
    th: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the temperature from potential temperature.

    Parameters
    ----------
    th: number or array-like
        Potential temperature (K)
    p: number or array-like
        Pressure (Pa)

    Returns
    -------
    number or array-like
        Temperature (K)


    The computation is based on the following formula:

    .. math::

       t = \theta (\frac{p}{10^{5}})^{\kappa}

    with :math:`\kappa = R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`).


    Implementations
    ------------------------
    :func:`temperature_from_potential_temperature` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.temperature_from_potential_temperature` for xarray.DataArray

    """
    return dispatch(temperature_from_potential_temperature, th, p)


@overload
def pressure_on_dry_adiabat(
    t: "xarray.DataArray",
    t_def: "xarray.DataArray",
    p_def: "xarray.DataArray",
) -> "xarray.DataArray": ...


def pressure_on_dry_adiabat(
    t: "xarray.DataArray",
    t_def: "xarray.DataArray",
    p_def: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the pressure on a dry adiabat.

    Parameters
    ----------
    t: number or array-like
        Temperature on the dry adiabat (K)
    t_def: number or array-like
        Temperature defining the dry adiabat (K)
    p_def: number or array-like
        Pressure defining the dry adiabat (Pa)

    Returns
    -------
    number or array-like
        Pressure on the dry adiabat (Pa)


    The computation is based on the following formula:

    .. math::

       p = p_{def} (\frac{t}{t_{def}})^{\frac{1}{\kappa}}

    with :math:`\kappa =  R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`).


    Implementations
    ------------------------
    :func:`pressure_on_dry_adiabat` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.pressure_on_dry_adiabat` for xarray.DataArray

    """
    return dispatch(pressure_on_dry_adiabat, t, t_def, p_def)


@overload
def temperature_on_dry_adiabat(
    p: "xarray.DataArray",
    t_def: "xarray.DataArray",
    p_def: "xarray.DataArray",
) -> "xarray.DataArray": ...


def temperature_on_dry_adiabat(
    p: "xarray.DataArray",
    t_def: "xarray.DataArray",
    p_def: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the temperature on a dry adiabat.

    Parameters
    ----------
    p: number or array-like
        Pressure on the dry adiabat (Pa)
    t_def: number or array-like
        Temperature defining the dry adiabat (K)
    p_def: number or array-like
        Pressure defining the dry adiabat (Pa)

    Returns
    -------
    number or array-like
        Temperature on the dry adiabat (K)


    The computation is based on the following formula:

    .. math::

       t = t_{def} (\frac{p}{p_{def}})^{\kappa}

    with :math:`\kappa =  R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`).


    Implementations
    ------------------------
    :func:`temperature_on_dry_adiabat` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.temperature_on_dry_adiabat` for xarray.DataArray

    """
    return dispatch(temperature_on_dry_adiabat, p, t_def, p_def)


@overload
def lcl_temperature(
    t: "xarray.DataArray",
    td: "xarray.DataArray",
    method: str = "davies",
) -> "xarray.DataArray": ...


def lcl_temperature(
    t: "xarray.DataArray",
    td: "xarray.DataArray",
    method: str = "davies",
) -> "xarray.DataArray":
    r"""Compute the Lifting Condenstaion Level (LCL) temperature from dewpoint.

    Parameters
    ----------
    t: number or array-like
        Temperature at the start level (K)
    td: number or array-like
        Dewpoint at the start level (K)
    method: str, optional
        The computation method: "davies" or "bolton".

    Returns
    -------
    number or array-like
        Temperature of the LCL (K)


    The actual computation is based on the ``method``:

    * "davies": the formula by [DaviesJones1983]_ is used (it is also used by the IFS model):

        .. math::

            t_{LCL} =
            td - (0.212 + 1.571\\times 10^{-3} (td - t_{0}) - 4.36\\times 10^{-4} (t - t_{0})) (t - td)

      where :math:`t_{0}` is the triple point of water (see :data:`earthkit.meteo.constants.T0`).

    * "bolton": the formula by [Bolton1980]_ is used:

        .. math::

            t_{LCL} = 56.0 +  \\frac{1}{\\frac{1}{td - 56} + \\frac{log(\\frac{t}{td})}{800}}


    Implementations
    ------------------------
    :func:`lcl_temperature` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.lcl_temperature` for xarray.DataArray

    """
    return dispatch(lcl_temperature, t, td, method=method)


@overload
def lcl(
    t: "xarray.DataArray",
    td: "xarray.DataArray",
    p: "xarray.DataArray",
    method: str = "davies",
) -> tuple["xarray.DataArray", "xarray.DataArray"]: ...


def lcl(
    t: "xarray.DataArray",
    td: "xarray.DataArray",
    p: "xarray.DataArray",
    method: str = "davies",
) -> tuple["xarray.DataArray", "xarray.DataArray"]:
    r"""Compute the temperature and pressure of the Lifting Condenstaion Level (LCL) from dewpoint.

    Parameters
    ----------
    t: number or array-like
        Temperature at the start level (K)
    td: number or array-like
        Dewpoint at the start level (K)
    p: number or array-like
        Pressure at the start level (Pa)
        method: str
    method: str, optional
        The computation method: "davies" or "bolton".

    Returns
    -------
    number or array-like
        Temperature of the LCL (K)
    number or array-like
        Pressure of the LCL (Pa)


    The LCL temperature is determined by :func:`lcl_temperature` with the given ``method``
    and the pressure is computed with :math:`t_{LCL}` using :func:`pressure_on_dry_adiabat`.


    Implementations
    ------------------------
    :func:`lcl` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.lcl` for xarray.DataArray

    """
    return dispatch(lcl, t, td, p, method=method)


@overload
def ept_from_dewpoint(
    t: "xarray.DataArray",
    td: "xarray.DataArray",
    p: "xarray.DataArray",
    method: str = "ifs",
) -> "xarray.DataArray": ...


def ept_from_dewpoint(
    t: "xarray.DataArray",
    td: "xarray.DataArray",
    p: "xarray.DataArray",
    method: str = "ifs",
) -> "xarray.DataArray":
    r"""Compute the equivalent potential temperature from dewpoint.

    Parameters
    ----------
    t: number or array-like
        Temperature (K)
    td: number or array-like
        Dewpoint (K)
    p: number or array-like
        Pressure (Pa)
    method: str, optional
        Specifies the computation method. The possible values are: "ifs", "bolton35", "bolton39".

    Returns
    -------
    number or array-like
        Equivalent potential temperature (K)


    The actual computation is based on the value of ``method``:

    * "ifs": the formula from the IFS model [IFS-CY47R3-PhysicalProcesses]_ (Chapter 6.11) is used:

        .. math::

            \\Theta_{e} = \\Theta\\; exp(\\frac{L_{v}\\; q}{c_{pd}\\; t_{LCL}})

    * "bolton35": Eq (35) from [Bolton1980]_ is used:

        .. math::

            \\Theta_{e} = \\Theta (\\frac{10^{5}}{p})^{\\kappa 0.28 w} exp(\\frac{2675 w}{t_{LCL}})

    * "bolton39": Eq (39) from [Bolton1980]_ is used:

        .. math::

            \\Theta_{e} =
            t (\\frac{10^{5}}{p-e})^{\\kappa} (\\frac{t}{t_{LCL}})^{0.28 w} exp[(\\frac{3036}{t_{LCL}} -
            1.78)w(1+0.448\\; w)]

    where:

        * :math:`\\Theta` is the :func:`potential_temperature`
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
        * :math:`\\kappa = R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`)


    Implementations
    ------------------------
    :func:`ept_from_dewpoint` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.ept_from_dewpoint` for xarray.DataArray

    """
    return dispatch(ept_from_dewpoint, t, td, p, method=method)


@overload
def ept_from_specific_humidity(
    t: "xarray.DataArray",
    q: "xarray.DataArray",
    p: "xarray.DataArray",
    method: str = "ifs",
) -> "xarray.DataArray": ...


def ept_from_specific_humidity(
    t: "xarray.DataArray",
    q: "xarray.DataArray",
    p: "xarray.DataArray",
    method: str = "ifs",
) -> "xarray.DataArray":
    r"""Compute the equivalent potential temperature from specific humidity.

    Parameters
    ----------
    t: number or array-like
        Temperature (K)
    q: number or array-like
        Specific humidity (kg/kg)
    p: number or array-like
        Pressure (Pa)
    method: str, optional
        Specifies the computation method. The possible values are: "ifs",
        "bolton35", "bolton39. See :func:`ept_from_dewpoint` for details.

    Returns
    -------
    number or array-like
        Equivalent potential temperature (K)


    The computations are the same as in :func:`ept_from_dewpoint`
    (the dewpoint is computed from q with :func:`dewpoint_from_specific_humidity`).


    Implementations
    ------------------------
    :func:`ept_from_specific_humidity` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.ept_from_specific_humidity` for xarray.DataArray

    """
    return dispatch(ept_from_specific_humidity, t, q, p, method=method)


@overload
def saturation_ept(
    t: "xarray.DataArray",
    p: "xarray.DataArray",
    method: str = "ifs",
) -> "xarray.DataArray": ...


def saturation_ept(
    t: "xarray.DataArray",
    p: "xarray.DataArray",
    method: str = "ifs",
) -> "xarray.DataArray":
    r"""Compute the saturation equivalent potential temperature.

    Parameters
    ----------
    t: number or array-like
        Temperature (K)
    p: number or array-like
        Pressure (Pa)
    method: str, optional
        Specifies the computation method. The possible values are: "ifs", "bolton35", "bolton39".

    Returns
    -------
    number or array-like
        Saturation equivalent potential temperature (K)


    The actual computation is based on the ``method``:

    * "ifs": The formula is based on the equivalent potential temperature definition used
       in the IFS model [IFS-CY47R3-PhysicalProcesses]_ (see Chapter 6.11) :

        .. math::

            \\Theta_{esat} = \\Theta\\; exp(\\frac{L_{v}\\; q_{sat}}{c_{pd}\\; t})

    * "bolton35": Eq (35) from [Bolton1980]_ is used:

        .. math::

            \\Theta_{e} = \\Theta (\\frac{10^{5}}{p})^{\\kappa 0.28 w_{sat}}\\; exp(\\frac{2675\\; w_{sat}}{t})

    * "bolton39": Eq (39) from [Bolton1980]_ is used:

        .. math::

            \\Theta_{e} =
            t (\\frac{10^{5}}{p-e_{sat}})^{\\kappa} exp[(\\frac{3036}{t} - 1.78)w_{sat}(1+0.448\\; w_{sat})]

    where:

        * :math:`\\Theta` is the :func:`potential_temperature`
        * :math:`e_{sat}` is the :func:`saturation_vapor_pressure`
        * :math:`q_{sat}` is the :func:`saturation_specific_humidity`
        * :math:`w_{sat}` is the :func:`saturation_mixing_ratio`
        * :math:`L_{v}` is the specific latent heat of vaporization (see :data:`earthkit.meteo.constants.Lv`)
        * :math:`c_{pd}` is the specific heat of dry air on constant pressure
          (see :data:`earthkit.meteo.constants.c_pd`)


    Implementations
    ------------------------
    :func:`saturation_ept` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.saturation_ept` for xarray.DataArray

    """
    return dispatch(saturation_ept, t, p, method=method)


@overload
def temperature_on_moist_adiabat(
    ept: "xarray.DataArray",
    p: "xarray.DataArray",
    ept_method: str = "ifs",
    t_method: str = "bisect",
) -> "xarray.DataArray": ...


def temperature_on_moist_adiabat(
    ept: "xarray.DataArray",
    p: "xarray.DataArray",
    ept_method: str = "ifs",
    t_method: str = "bisect",
) -> "xarray.DataArray":
    r"""Compute the temperature on a moist adiabat (pseudoadiabat)

    Parameters
    ----------
    ept: number or array-like
        Equivalent potential temperature defining the moist adiabat (K)
    p: number or array-like
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
    number or array-like
        Temperature on the moist adiabat (K). For values where the computation cannot
        be carried out nan is returned.


    Implementations
    ------------------------
    :func:`temperature_on_moist_adiabat` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.temperature_on_moist_adiabat` for xarray.DataArray

    """
    return dispatch(temperature_on_moist_adiabat, ept, p, ept_method=ept_method, t_method=t_method)


@overload
def wet_bulb_temperature_from_dewpoint(
    t: "xarray.DataArray",
    td: "xarray.DataArray",
    p: "xarray.DataArray",
    ept_method: str = "ifs",
    t_method: str = "bisect",
) -> "xarray.DataArray": ...


def wet_bulb_temperature_from_dewpoint(
    t: "xarray.DataArray",
    td: "xarray.DataArray",
    p: "xarray.DataArray",
    ept_method: str = "ifs",
    t_method: str = "bisect",
) -> "xarray.DataArray":
    r"""Compute the pseudo adiabatic wet bulb temperature from dewpoint.

    Parameters
    ----------
    t: number or array-like
        Temperature (K)
    td: number or array-like
        Dewpoint (K)
    p: number or array-like
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
    number or array-like
        Wet bulb temperature (K)


    The computation is based on Normand's rule [Wallace2006]_ (Chapter 3.5.6):

    * first the equivalent potential temperature is computed with the given
      ``ept_method`` (using :func:`ept_from_dewpoint`). This defines the moist adiabat.
    * then the wet bulb potential temperature is determined as the temperature at
      pressure ``p`` on the moist adiabat with the given ``t_method``.


    Implementations
    ------------------------
    :func:`wet_bulb_temperature_from_dewpoint` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.wet_bulb_temperature_from_dewpoint` for xarray.DataArray

    """
    return dispatch(wet_bulb_temperature_from_dewpoint, t, td, p, ept_method=ept_method, t_method=t_method)


@overload
def wet_bulb_temperature_from_specific_humidity(
    t: "xarray.DataArray",
    q: "xarray.DataArray",
    p: "xarray.DataArray",
    ept_method: str = "ifs",
    t_method: str = "bisect",
) -> "xarray.DataArray": ...


def wet_bulb_temperature_from_specific_humidity(
    t: "xarray.DataArray",
    q: "xarray.DataArray",
    p: "xarray.DataArray",
    ept_method: str = "ifs",
    t_method: str = "bisect",
) -> "xarray.DataArray":
    r"""Compute the pseudo adiabatic wet bulb temperature from specific humidity.

    Parameters
    ----------
    t: number or array-like
        Temperature (K)
    q: number or array-like
        Specific humidity (kg/kg)
    p: number or array-like
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
    number or array-like
        Wet bulb temperature (K)


    The computation is based on Normand's rule [Wallace2006]_ (Chapter 3.5.6):

    * first the equivalent potential temperature is computed with the given
      ``ept_method`` (using :func:`ept_from_dewpoint`). This defines the moist adiabat.
    * then the wet bulb potential temperature is determined as the temperature at
      pressure ``p`` on the moist adiabat with the given ``t_method``.


    Implementations
    ------------------------
    :func:`wet_bulb_temperature_from_specific_humidity` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.wet_bulb_temperature_from_specific_humidity` for xarray.DataArray

    """
    return dispatch(
        wet_bulb_temperature_from_specific_humidity,
        t,
        q,
        p,
        ept_method=ept_method,
        t_method=t_method,
    )


@overload
def wet_bulb_potential_temperature_from_dewpoint(
    t: "xarray.DataArray",
    td: "xarray.DataArray",
    p: "xarray.DataArray",
    ept_method: str = "ifs",
    t_method: str = "direct",
) -> "xarray.DataArray": ...


def wet_bulb_potential_temperature_from_dewpoint(
    t: "xarray.DataArray",
    td: "xarray.DataArray",
    p: "xarray.DataArray",
    ept_method: str = "ifs",
    t_method: str = "direct",
) -> "xarray.DataArray":
    r"""Compute the pseudo adiabatic wet bulb potential temperature from dewpoint.

    Parameters
    ----------
    t: number or array-like
        Temperature (K)
    td: number or array-like
        Dewpoint (K)
    p: number or array-like
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
    number or array-like
        Wet bulb potential temperature (K)


    The computation is based on Normand's rule [Wallace2006]_ (Chapter 3.5.6):

    * first the equivalent potential temperature is computed with the given
      ``ept_method`` (using :func:`ept_from_dewpoint`). This defines the moist adiabat.
    * then the wet bulb potential temperature is determined as the temperature at
      pressure :math:`10^{5}` Pa on the moist adiabat with the given ``t_method``.


    Implementations
    ------------------------
    :func:`wet_bulb_potential_temperature_from_dewpoint` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.wet_bulb_potential_temperature_from_dewpoint` for xarray.DataArray

    """
    return dispatch(
        wet_bulb_potential_temperature_from_dewpoint,
        t,
        td,
        p,
        ept_method=ept_method,
        t_method=t_method,
    )


@overload
def wet_bulb_potential_temperature_from_specific_humidity(
    t: "xarray.DataArray",
    q: "xarray.DataArray",
    p: "xarray.DataArray",
    ept_method: str = "ifs",
    t_method: str = "direct",
) -> "xarray.DataArray": ...


def wet_bulb_potential_temperature_from_specific_humidity(
    t: "xarray.DataArray",
    q: "xarray.DataArray",
    p: "xarray.DataArray",
    ept_method: str = "ifs",
    t_method: str = "direct",
) -> "xarray.DataArray":
    r"""Compute the pseudo adiabatic wet bulb potential temperature from specific humidity.

    Parameters
    ----------
    t: number or array-like
        Temperature (K)
    q: number or array-like
        Specific humidity (kg/kg)
    p: number or array-like
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
    number or array-like
        Wet bulb potential temperature (K)


    The computations are the same as in
    :func:`wet_bulb_potential_temperature_from_dewpoint`
    (the dewpoint is computed from q with :func:`dewpoint_from_specific_humidity`).


    Implementations
    ------------------------
    :func:`wet_bulb_potential_temperature_from_specific_humidity` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.wet_bulb_potential_temperature_from_specific_humidity` for xarray.DataArray

    """
    return dispatch(
        wet_bulb_potential_temperature_from_specific_humidity,
        t,
        q,
        p,
        ept_method=ept_method,
        t_method=t_method,
    )


@overload
def specific_gas_constant(q: "xarray.DataArray") -> "xarray.DataArray": ...


def specific_gas_constant(q: "xarray.DataArray") -> "xarray.DataArray":
    r"""Compute the specific gas constant of moist air.

    Specific content of cloud particles and hydrometeors are neglected.

    Parameters
    ----------
    q: number or array-like
        Specific humidity (kg/kg)

    Returns
    -------
    number or array-like
        Specific gas constant of moist air (J kg-1 K-1)


    The computation is based on the following formula:

    .. math::

        R = R_{d} + (R_{v} - R_{d}) q

    where:

        * :math:`R_{d}` is the gas constant for dry air (see :data:`earthkit.meteo.constants.Rd`)
        * :math:`R_{v}` is the gas constant for water vapour (see :data:`earthkit.meteo.constants.Rv`)


    Implementations
    ------------------------
    :func:`specific_gas_constant` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.thermo.xarray.specific_gas_constant` for xarray.DataArray

    """
    return dispatch(specific_gas_constant, q)
