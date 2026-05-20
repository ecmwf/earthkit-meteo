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

from earthkit.data import Field, FieldList  # type: ignore[import]

from earthkit.meteo.utils.decorators import fieldlist_ufunc
from earthkit.meteo.utils.fieldlist import pressure_from_metadata

from .. import array


def specific_humidity_from_mixing_ratio(w: FieldList | Field) -> FieldList | Field:
    r"""Compute the specific humidity from mixing ratio.

    Parameters
    ----------
    w : FieldList|Field
        Mixing ratio (kg/kg).

    Returns
    -------
    FieldList|Field
        Specific humidity (kg/kg). The result has the same type as the input ``w`` (FieldList or Field).


    The result is the specific humidity in kg/kg units. The computation is based on
    the following definition [Wallace2006]_:

    .. math::

        q = \frac {w}{1+w}

    """
    fieldlist_ufunc_kwargs = {"default": "q"}
    return fieldlist_ufunc(array.specific_humidity_from_mixing_ratio, w, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def mixing_ratio_from_specific_humidity(q: FieldList | Field) -> FieldList | Field:
    r"""Compute the mixing ratio from specific humidity.

    Parameters
    ----------
    q : FieldList|Field
        Specific humidity (kg/kg).

    Returns
    -------
    FieldList|Field
        Mixing ratio (kg/kg). The result has the same type as the input ``q`` (FieldList or Field).


    The result is the mixing ratio in kg/kg units. The computation is based on
    the following definition [Wallace2006]_:

    .. math::

        w = \frac {q}{1-q}

    """
    fieldlist_ufunc_kwargs = {"default": "w"}
    return fieldlist_ufunc(array.mixing_ratio_from_specific_humidity, q, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def vapour_pressure_from_specific_humidity(
    q: FieldList | Field, p: FieldList | Field | Iterable[float] | float | None = None
) -> FieldList | Field:
    r"""Compute the vapour pressure from specific humidity.

    Parameters
    ----------
    q : FieldList|Field
        Specific humidity (kg/kg).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``q``. Otherwise, if ``q``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``q``. If ``q``
        is a Field, ``p`` must be a single Field or a float.

    Returns
    -------
    FieldList|Field
        Vapour pressure (Pa). The result has the same type as the input ``q`` (FieldList or Field).


    The computation is based on the following formula [Wallace2006]_:

    .. math::

        e = \frac{pq}{\epsilon (1 + q(\frac{1}{\epsilon} -1 ))}

    with :math:`\epsilon =  R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).

    """
    fieldlist_ufunc_kwargs = {"default": "vapp", "param_unit": "Pa"}
    if p is None:
        p = pressure_from_metadata(q)  # convert to Pa

    return fieldlist_ufunc(
        array.vapour_pressure_from_specific_humidity, q, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def vapour_pressure_from_mixing_ratio(
    w: FieldList | Field, p: FieldList | Field | Iterable[float] | float | None = None
) -> FieldList | Field:
    r"""Compute the vapour pressure from mixing ratio.

    Parameters
    ----------
    w : FieldList|Field
        Mixing ratio (kg/kg).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``w``. Otherwise, if ``w``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``w``. If ``w``
        is a Field, ``p`` must be a single Field or a float.

    Returns
    -------
    FieldList|Field
        Vapour pressure (Pa). The result has the same type as the input ``w`` (FieldList or Field).


    The computation is based on the following formula:

    .. math::

        e = \frac{pw}{\epsilon + w}

    with :math:`\epsilon =  R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).

    """
    fieldlist_ufunc_kwargs = {"default": "vapp", "param_unit": "Pa"}
    if p is None:
        p = pressure_from_metadata(w)  # convert to Pa

    return fieldlist_ufunc(array.vapour_pressure_from_mixing_ratio, w, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def specific_humidity_from_vapour_pressure(
    e: FieldList | Field, p: FieldList | Field | Iterable[float] | float | None = None, eps: float = 1e-4
) -> FieldList | Field:
    r"""Compute the specific humidity from vapour pressure.

    Parameters
    ----------
    e : FieldList|Field
        Vapour pressure (Pa).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``e``. Otherwise, if ``e``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``e``. If ``e``
        is a Field, ``p`` must be a single Field or a float.
    eps : float, optional
        Where p - e < ``eps`` nan is returned.

    Returns
    -------
    FieldList|Field
        Specific humidity (kg/kg). The result has the same type as the input ``e`` (FieldList or Field).


    The computation is based on the following formula:

    .. math::

       q = \frac{\epsilon e}{p + e(\epsilon-1)}

    with :math:`\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).

    """
    fieldlist_ufunc_kwargs = {"default": "q", "param_unit": "kg/kg"}
    if p is None:
        p = pressure_from_metadata(e)  # convert to Pa

    return fieldlist_ufunc(
        array.specific_humidity_from_vapour_pressure,
        e,
        p,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        eps=eps,
    )


def mixing_ratio_from_vapour_pressure(
    e: FieldList | Field, p: FieldList | Field | Iterable[float] | float | None = None, eps: float = 1e-4
) -> FieldList | Field:
    r"""Compute the mixing ratio from vapour pressure.

    Parameters
    ----------
    e : FieldList|Field
        Vapour pressure (Pa).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``e``. Otherwise, if ``e``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``e``. If ``e``
        is a Field, ``p`` must be a single Field or a float.
    eps : float, optional
        Where p - e < ``eps`` nan is returned.

    Returns
    -------
    FieldList|Field
        Mixing ratio (kg/kg). The result has the same type as the input ``e`` (FieldList or Field).


    The computation is based on the following formula:

    .. math::

       w = \frac{\epsilon e}{p - e}

    with :math:`\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).

    """
    fieldlist_ufunc_kwargs = {"default": "w", "param_unit": "kg/kg"}
    if p is None:
        p = pressure_from_metadata(e)  # convert to Pa

    return fieldlist_ufunc(
        array.mixing_ratio_from_vapour_pressure, e, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, eps=eps
    )


def saturation_vapour_pressure(t: FieldList | Field, phase: str = "mixed") -> FieldList | Field:
    r"""Compute the saturation vapour pressure from temperature with respect to a phase.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    phase : str, optional
        Define the phase with respect to the saturation vapour pressure is computed.
        It is either "water", "ice" or "mixed".

    Returns
    -------
    FieldList|Field
        Saturation vapour pressure (Pa). The result has the same type as the input ``t`` (FieldList or Field).


    The algorithm was taken from the IFS model [IFS-CY47R3-PhysicalProcesses]_ (see Chapter 12).
    It uses the following formula when ``phase`` is "water" or "ice":

    .. math::

        e_{sat} = a_{1} exp \left(a_{3}\frac{t-273.16}{t-a_{4}}\right)

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
    fieldlist_ufunc_kwargs = {"default": "swvp", "param_unit": "Pa"}
    return fieldlist_ufunc(
        array.saturation_vapour_pressure, t, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, phase=phase
    )


def saturation_mixing_ratio(
    t: FieldList | Field, p: FieldList | Field | Iterable[float] | float | None = None, phase: str = "mixed"
) -> FieldList | Field:
    r"""Compute the saturation mixing ratio from temperature with respect to a phase.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``t``. Otherwise, if ``t``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t``. If ``t``
        is a Field, ``p`` must be a single Field or a float.
    phase : str, optional
        Define the phase with respect to the :func:`saturation_vapour_pressure` is computed.
        It is either "water", "ice" or "mixed".

    Returns
    -------
    FieldList|Field
        Saturation mixing ratio (kg/kg). The result has the same type as the input ``t`` (FieldList or Field).


    Equivalent to the following code:

    .. code-block:: python

        e = saturation_vapour_pressure(t, phase=phase)
        return mixing_ratio_from_vapour_pressure(e, p)

    """
    fieldlist_ufunc_kwargs = {"default": "ws", "param_unit": "kg/kg"}

    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa

    return fieldlist_ufunc(
        array.saturation_mixing_ratio, t, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, phase=phase
    )


def saturation_specific_humidity(
    t: FieldList | Field, p: FieldList | Field | Iterable[float] | float | None = None, phase: str = "mixed"
) -> FieldList | Field:
    r"""Compute the saturation specific humidity from temperature with respect to a phase.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``t``. Otherwise, if ``t``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t``. If ``t``
        is a Field, ``p`` must be a single Field or a float.
    phase : str, optional
        Define the phase with respect to the :func:`saturation_vapour_pressure` is computed.
        It is either "water", "ice" or "mixed".

    Returns
    -------
    FieldList|Field
        Saturation specific humidity (kg/kg). The result has the same type as the input ``t`` (FieldList or Field).


    Equivalent to the following code:

    .. code-block:: python

        e = saturation_vapour_pressure(t, phase=phase)
        return specific_humidity_from_vapour_pressure(e, p)

    """
    fieldlist_ufunc_kwargs = {"default": "sqw", "param_unit": "kg/kg"}
    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(
        array.saturation_specific_humidity, t, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, phase=phase
    )


def saturation_vapour_pressure_slope(t: FieldList | Field, phase: str = "mixed") -> FieldList | Field:
    r"""Compute the slope of saturation vapour pressure with respect to temperature.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K)
    phase : str, optional
        Define the phase with respect to the computation will be performed.
        It is either "water", "ice" or "mixed". See :func:`saturation_vapour_pressure`
        for details.

    Returns
    -------
    FieldList|Field
        Slope of saturation vapour pressure (Pa/K).
        The result has the same type as the input ``t`` (FieldList or Field).

    """
    fieldlist_ufunc_kwargs = {"default": "es_slope", "param_unit": "Pa/K"}
    return fieldlist_ufunc(
        array.saturation_vapour_pressure_slope, t, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, phase=phase
    )


def saturation_mixing_ratio_slope(
    t: FieldList | Field,
    p: FieldList | Field | Iterable[float] | float | None = None,
    es: FieldList | Field | None = None,
    es_slope: FieldList | Field | None = None,
    phase: str = "mixed",
    eps: float = 1e-4,
) -> FieldList | Field:
    r"""Compute the slope of saturation mixing ratio with respect to temperature.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``t``. Otherwise, if ``t``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t``. If ``t``
        is a Field, ``p`` must be a single Field or a float.
    es: FieldList|Field|None, optional
        :func:`saturation_vapour_pressure` pre-computed for the given ``phase`` (Pa).
        When specified, it is used in the computation instead of being computed from
        ``t`` and ``phase`` using :func:`saturation_vapour_pressure`.
    es_slope: FieldList|Field|None, optional
        :func:`saturation_vapour_pressure_slope` pre-computed for the given ``phase`` (Pa/K).
        When specified, it is used in the computation instead of being computed from
        ``t`` and ``phase`` using :func:`saturation_vapour_pressure_slope`.
    phase : str, optional
        Define the phase with respect to the computation will be performed.
        It is either "water", "ice" or "mixed".
    eps : float, optional
        Where p - es < ``eps`` nan is returned.

    Returns
    -------
    FieldList|Field
        Slope of saturation mixing ratio (kg kg-1 K-1).
        The result has the same type as the input ``t`` (FieldList or Field).


    The computation is based on the following formula:

    .. math::

        \frac{\partial w_{s}}{\partial t} = \frac{\epsilon  p}{(p-e_{s})^{2}} \frac{d e_{s}}{d t}

    where

        * :math:`\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).
        * :math:`e_{s}` is the :func:`saturation_vapour_pressure` for the given ``phase``

    """
    fieldlist_ufunc_kwargs = {"default": "ws_slope", "param_unit": "kg kg-1 K-1"}
    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(
        array.saturation_mixing_ratio_slope,
        t,
        p,
        es,
        es_slope,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        phase=phase,
        eps=eps,
    )


def saturation_specific_humidity_slope(
    t: FieldList | Field,
    p: FieldList | Field | Iterable[float] | float | None = None,
    es: FieldList | Field | None = None,
    es_slope: FieldList | Field | None = None,
    phase: str = "mixed",
    eps: float = 1e-4,
) -> FieldList | Field:
    r"""Compute the slope of saturation specific humidity with respect to temperature.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``t``. Otherwise, if ``t``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t``. If ``t``
        is a Field, ``p`` must be a single Field or a float.
    es: FieldList|Field|None, optional
        :func:`saturation_vapour_pressure` pre-computed for the given ``phase`` (Pa).
        When specified, it is used in the computation instead of being computed from
        ``t`` and ``phase`` using :func:`saturation_vapour_pressure`.
    es_slope: FieldList|Field|None, optional
        :func:`saturation_vapour_pressure_slope` pre-computed for the given ``phase`` (Pa/K).
        When specified, it is used in the computation instead of being computed from
        ``t`` and ``phase`` using :func:`saturation_vapour_pressure_slope`.
    phase : str, optional
        Define the phase with respect to the computation will be performed.
        It is either "water", "ice" or "mixed".
    eps : float, optional
        Where p - es < ``eps`` nan is returned.

    Returns
    -------
    FieldList|Field
        Slope of saturation specific humidity (kg kg-1 K-1).
        The result has the same type as the input ``t`` (FieldList or Field).


    The computation is based on the following formula:

    .. math::

        \frac{\partial q_{s}}{\partial t} =
        \frac{\epsilon  p}{(p+e_{s}(\epsilon - 1))^{2}} \frac{d e_{s}}{d t}

    where

        * :math:`\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).
        * :math:`e_{s}` is the :func:`saturation_vapour_pressure` for the given ``phase``

    """
    fieldlist_ufunc_kwargs = {"default": "sqw_slope", "param_unit": "kg kg-1 K-1"}
    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa

    return fieldlist_ufunc(
        array.saturation_specific_humidity_slope,
        t,
        p,
        es,
        es_slope,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        phase=phase,
        eps=eps,
    )


def temperature_from_saturation_vapour_pressure(es: FieldList | Field) -> FieldList | Field:
    r"""Compute the temperature from saturation vapour pressure.

    Parameters
    ----------
    es : FieldList|Field
        :func:`saturation_vapour_pressure` (Pa).

    Returns
    -------
    FieldList|Field
        Temperature (K). For zero ``es`` values returns nan.
        The result has the same type as the input ``es`` (FieldList or Field).


    The computation is always based on the "water" phase of
    the :func:`saturation_vapour_pressure` formulation irrespective of the
    phase ``es`` was computed to.

    """
    fieldlist_ufunc_kwargs = {"default": "t", "param_unit": "K"}
    return fieldlist_ufunc(
        array.temperature_from_saturation_vapour_pressure, es, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def relative_humidity_from_dewpoint(t: FieldList | Field, td: FieldList | Field) -> FieldList | Field:
    r"""Compute the relative humidity from dewpoint temperature.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    td : FieldList|Field
        Dewpoint (K).

    Returns
    -------
    FieldList|Field
        Relative humidity (%). The result has the same type as the input ``t`` and ``td`` (FieldList or Field).


    The computation is based on the following formula:

    .. math::

        r = 100 \frac {e_{wsat}(td)}{e_{wsat}(t)}

    where :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water.

    """
    fieldlist_ufunc_kwargs = {"default": "r", "param_unit": "%"}
    return fieldlist_ufunc(array.relative_humidity_from_dewpoint, t, td, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def relative_humidity_from_specific_humidity(
    t: FieldList | Field, q: FieldList | Field, p: FieldList | Field | Iterable[float] | float | None = None
) -> FieldList | Field:
    r"""Compute the relative humidity from specific humidity.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    q : FieldList|Field
        Specific humidity (kg/kg).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``t``. Otherwise, if ``t``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t``. If ``t``
        is a Field, ``p`` must be a single Field or a float.

    Returns
    -------
    FieldList|Field
        Relative humidity (%). The result has the same type as the input ``t`` and ``q`` (FieldList or Field).


    The computation is based on the following formula:

    .. math::

        r = 100 \frac {e(q, p)}{e_{msat}(t)}

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure_from_specific_humidity`)
        * :math:`e_{msat}` is the :func:`saturation_vapour_pressure` based on the "mixed" phase

    """
    fieldlist_ufunc_kwargs = {"default": "r", "param_unit": "%"}
    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(
        array.relative_humidity_from_specific_humidity, t, q, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def specific_humidity_from_dewpoint(
    td: FieldList | Field, p: FieldList | Field | Iterable[float] | float | None = None
) -> FieldList | Field:
    r"""Compute the specific humidity from dewpoint.

    Parameters
    ----------
    td : FieldList|Field
        Dewpoint (K).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``td``. Otherwise, if ``td``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``td``. If ``td``
        is a Field, ``p`` must be a single Field or a float.

    Returns
    -------
    FieldList|Field
        Specific humidity (kg/kg). The result has the same type as the input ``td`` (FieldList or Field).


    The computation starts with determining the vapour pressure:

    .. math::

        e(q, p) = e_{wsat}(td)

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure_from_specific_humidity`)
        * :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water
        * :math:`q` is the specific humidity

    Then `q` is computed from :math:`e` using :func:`specific_humidity_from_vapour_pressure`.

    """
    fieldlist_ufunc_kwargs = {"default": "q", "param_unit": "kg/kg"}
    if p is None:
        p = pressure_from_metadata(td)  # convert to Pa
    return fieldlist_ufunc(array.specific_humidity_from_dewpoint, td, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def mixing_ratio_from_dewpoint(
    td: FieldList | Field, p: FieldList | Field | Iterable[float] | float | None = None
) -> FieldList | Field:
    r"""Compute the mixing ratio from dewpoint.

    Parameters
    ----------
    td : FieldList|Field
        Dewpoint (K).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``td``. Otherwise, if ``td``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``td``. If ``td``
        is a Field, ``p`` must be a single Field or a float.

    Returns
    -------
    FieldList|Field
        Mixing ratio (kg/kg). The result has the same type as the input ``td`` (FieldList or Field).


    The computation starts with determining the vapour pressure:

    .. math::

        e(w, p) = e_{wsat}(td)

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure_from_mixing_ratio`)
        * :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water
        * :math:`w` is the mixing ratio

    Then `w` is computed from :math:`e` using :func:`mixing_ratio_from_vapour_pressure`.

    """
    fieldlist_ufunc_kwargs = {"default": "w", "param_unit": "kg/kg"}
    if p is None:
        p = pressure_from_metadata(td)  # convert to Pa
    return fieldlist_ufunc(array.mixing_ratio_from_dewpoint, td, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def specific_humidity_from_relative_humidity(
    t: FieldList | Field, r: FieldList | Field, p: FieldList | Field | Iterable[float] | float | None = None
) -> FieldList | Field:
    r"""Compute the specific humidity from relative humidity.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    r : FieldList|Field
        Relative humidity (%).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the field metadata.

    Returns
    -------
    FieldList|Field
        Specific humidity (kg/kg). The result has the same type as the input ``t`` and ``r`` (FieldList or Field).


    The computation starts with determining the vapour pressure:

    .. math::

        e(q, p) = r  \frac{e_{msat}(t)}{100}

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure`)
        * :math:`e_{msat}` is the :func:`saturation_vapour_pressure` based on the "mixed" phase
        * :math:`q` is the specific humidity

    Then :math:`q` is computed from :math:`e` using :func:`specific_humidity_from_vapour_pressure`.

    """
    fieldlist_ufunc_kwargs = {"default": "q", "param_unit": "kg/kg"}
    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(
        array.specific_humidity_from_relative_humidity, t, r, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def dewpoint_from_relative_humidity(t: FieldList | Field, r: FieldList | Field) -> FieldList | Field:
    r"""Compute the dewpoint temperature from relative humidity.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    r : FieldList|Field
        Relative humidity (%).

    Returns
    -------
    FieldList|Field
        Dewpoint temperature (K). For zero ``r`` values returns nan.
        The result has the same type as the input ``t`` and ``r`` (FieldList or Field).


    The computation starts with determining the saturation vapour pressure over
    water at the dewpoint temperature:

    .. math::

        e_{wsat}(td) = \frac{r  e_{wsat}(t)}{100}

    where:

    * :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water
    * :math:`td` is the dewpoint.

    Then :math:`td` is computed from :math:`e_{wsat}(td)` by inverting the
    equations used in :func:`saturation_vapour_pressure`.

    """
    param_ids = {
        # 130: "td",  # atmospheric dewpoint, paramId=?
        167: "2d",  # 2m dewpoint, paramId=168
    }

    variables = {
        # "t": "td",  # atmospheric dewpoint
        "2t": "2d",  # 2m dewpoint
    }

    fieldlist_ufunc_kwargs = {"param_ids": param_ids, "variables": variables, "default": "td"}
    return fieldlist_ufunc(array.dewpoint_from_relative_humidity, t, r, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def dewpoint_from_specific_humidity(
    q: FieldList | Field, p: FieldList | Field | Iterable[float] | float | None = None
) -> FieldList | Field:
    r"""Compute the dewpoint temperature from specific humidity.

    Parameters
    ----------
    q : FieldList|Field
        Specific humidity (kg/kg).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``q``. Otherwise, if ``q``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``q``. If ``q``
        is a Field, ``p`` must be a single Field or a float.

    Returns
    -------
    FieldList|Field
        Dewpoint temperature (K). For zero ``q`` values returns nan.
        The result has the same type as the input ``q`` (FieldList or Field).


    The computation starts with determining the saturation vapour pressure over
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
    param_ids = {
        174096: "2d",  # 2m dewpoint, paramId=168
    }

    variables = {
        "2sh": "2d",  # 2m dewpoint
    }

    fieldlist_ufunc_kwargs = {
        "param_ids": param_ids,
        "variables": variables,
        "default": "td",
        "param_unit": "K",
    }

    if p is None:
        p = pressure_from_metadata(q)  # convert to Pa
    return fieldlist_ufunc(array.dewpoint_from_specific_humidity, q, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def virtual_temperature(t: FieldList | Field, q: FieldList | Field) -> FieldList | Field:
    r"""Compute the virtual temperature from temperature and specific humidity.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    q : FieldList|Field
        Specific humidity (kg/kg)

    Returns
    -------
    FieldList|Field
        Virtual temperature (K). The result has the same type as the input ``t`` and ``q`` (FieldList or Field).


    The computation is based on the following formula [Wallace2006]_:

    .. math::

        t_{v} = t (1 + \frac{1 - \epsilon}{\epsilon} q)

    with :math:`\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).

    """
    fieldlist_ufunc_kwargs = {"default": "vtmp"}
    return fieldlist_ufunc(array.virtual_temperature, t, q, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def virtual_potential_temperature(
    t: FieldList | Field, q: FieldList | Field, p: FieldList | Field | Iterable[float] | float | None = None
) -> FieldList | Field:
    r"""Compute the virtual potential temperature from temperature and specific humidity.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    q : FieldList|Field
        Specific humidity (kg/kg).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``t``. Otherwise, if ``t``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t``. If ``t``
        is a Field, ``p`` must be a single Field or a float.

    Returns
    -------
    FieldList|Field
        Virtual potential temperature (K).
        The result has the same type as the input ``t`` and ``q`` (FieldList or Field).


    The computation is based on the following formula:

    .. math::

        \Theta_{v} = \theta (1 + \frac{1 - \epsilon}{\epsilon} q)

    where:

        * :math:`\Theta` is the :func:`potential_temperature`
        * :math:`\epsilon = R_{d}/R_{v}` (see :data:`earthkit.meteo.constants.epsilon`).

    """
    fieldlist_ufunc_kwargs = {"default": "vptmp"}

    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa

    return fieldlist_ufunc(array.virtual_potential_temperature, t, q, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def potential_temperature(
    t: FieldList | Field, p: FieldList | Field | Iterable[float] | float | None = None
) -> FieldList | Field:
    r"""Compute the potential temperature.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``t``. Otherwise, if ``t``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t``. If ``t``
        is a Field, ``p`` must be a single Field or a float.

    Returns
    -------
    FieldList|Field
        Potential temperature (K). The result has the same type as the input ``t`` (FieldList or Field).


    The computation is based on the following formula [Wallace2006]_:

    .. math::

       \theta = t \left(\frac{10^{5}}{p}\right)^{\kappa}

    with :math:`\kappa = R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`).

    """
    fieldlist_ufunc_kwargs = {"default": "pt"}

    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa

    return fieldlist_ufunc(array.potential_temperature, t, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def temperature_from_potential_temperature(
    th: FieldList | Field, p: FieldList | Field | Iterable[float] | float | None = None
) -> FieldList | Field:
    r"""Compute the temperature from potential temperature.

    Parameters
    ----------
    th : FieldList|Field
        Potential temperature (K).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``th``. Otherwise, if ``th``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``th``. If ``th``
        is a Field, ``p`` must be a single Field or a float.

    Returns
    -------
    FieldList|Field
        Temperature (K). The result has the same type as the input ``th`` (FieldList or Field).


    The computation is based on the following formula:

    .. math::

       t = \theta (\frac{p}{10^{5}})^{\kappa}

    with :math:`\kappa = R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`).

    """
    fieldlist_ufunc_kwargs = {"default": "t"}

    if p is None:
        p = pressure_from_metadata(th)

    return fieldlist_ufunc(
        array.temperature_from_potential_temperature, th, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def pressure_on_dry_adiabat(
    t: FieldList | Field,
    t_def: FieldList | Field,
    p_def: FieldList | Field | Iterable[float] | float | None = None,
) -> FieldList | Field:
    r"""Compute the pressure on a dry adiabat.

    Parameters
    ----------
    t : FieldList|Field
        Temperature on the dry adiabat (K).
    t_def : FieldList|Field
        Temperature defining the dry adiabat (K).
    p_def : FieldList|Field|Iterable[float]|float|None
        Pressure defining the dry adiabat (Pa). If None, inferred from the
        field metadata of ``t_def``. Otherwise, if ``t_def``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t_def``. If ``t_def``
        is a Field, ``p`` must be a single Field or a float.


    Returns
    -------
    FieldList|Field
        Pressure on the dry adiabat (Pa). The result has the same type as the input ``t`` (FieldList or Field).


    The computation is based on the following formula:

    .. math::

       p = p_{def} (\frac{t}{t_{def}})^{\frac{1}{\kappa}}

    with :math:`\kappa =  R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`).

    """
    fieldlist_ufunc_kwargs = {"default": "pres", "param_unit": "Pa"}

    if p_def is None:
        p_def = pressure_from_metadata(t_def)

    return fieldlist_ufunc(
        array.pressure_on_dry_adiabat, t, t_def, p_def, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def temperature_on_dry_adiabat(
    p: FieldList | Field | float,
    t_def: FieldList | Field,
    p_def: FieldList | Field | Iterable[float] | float | None = None,
) -> FieldList | Field:
    r"""Compute the temperature on a dry adiabat.

    Parameters
    ----------
    p : FieldList|Field|float
        Pressure on the dry adiabat (Pa).
    t_def : FieldList|Field
        Temperature defining the dry adiabat (K).
    p_def : FieldList|Field|Iterable[float]|float|None
        Pressure defining the dry adiabat (Pa). If None, inferred from the
        field metadata of ``t_def``. Otherwise, if ``t_def``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t_def``. If ``t_def``
        is a Field, ``p`` must be a single Field or a float.


    Returns
    -------
    FieldList|Field
        Temperature on the dry adiabat (K). The result has the same type as the input ``p`` (FieldList or Field).


    The computation is based on the following formula:

    .. math::

       t = t_{def} (\frac{p}{p_{def}})^{\kappa}

    with :math:`\kappa =  R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`).

    """
    fieldlist_ufunc_kwargs = {"default": "t", "param_unit": "K"}

    if p_def is None:
        p_def = pressure_from_metadata(t_def)

    return fieldlist_ufunc(
        array.temperature_on_dry_adiabat, p, t_def, p_def, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def lcl_temperature(t: FieldList | Field, td: FieldList | Field, method: str = "davies") -> FieldList | Field:
    r"""Compute the Lifting Condensation Level (LCL) temperature from dewpoint.

    Parameters
    ----------
    t : FieldList|Field
        Temperature at the start level (K).
    td : FieldList|Field
        Dewpoint at the start level (K).
    method : str, optional
        The computation method: "davies" or "bolton".

    Returns
    -------
    FieldList|Field
        Temperature of the LCL (K). The result has the same type as the input ``t`` and ``td`` (FieldList or Field).


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
    fieldlist_ufunc_kwargs = {"default": "t_lcl"}
    return fieldlist_ufunc(array.lcl_temperature, t, td, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, method=method)


def lcl(
    t: FieldList | Field,
    td: FieldList | Field,
    p: FieldList | Field | Iterable[float] | float | None = None,
    method: str = "davies",
) -> tuple[FieldList | Field, FieldList | Field]:
    r"""Compute the temperature and pressure of the Lifting Condensation Level (LCL) from dewpoint.

    Parameters
    ----------
    t : FieldList|Field
        Temperature at the start level (K).
    td : FieldList|Field
        Dewpoint at the start level (K).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure at the start level (Pa). If None, inferred from the
        field metadata of ``t``. Otherwise, if ``t``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t``. If ``t``
        is a Field, ``p`` must be a single Field or a float.
    method : str, optional
        The computation method: "davies" or "bolton".

    Returns
    -------
    FieldList|Field
        Temperature of the LCL (K). The result has the same type as the input ``t`` and ``td`` (FieldList or Field).
    FieldList|Field
        Pressure of the LCL (Pa). The result has the same type as the input ``t`` and ``td`` (FieldList or Field).


    The LCL temperature is determined by :func:`lcl_temperature` with the given ``method``
    and the pressure is computed with :math:`t_{LCL}` using :func:`pressure_on_dry_adiabat`.

    """
    import earthkit.data as ekd

    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa

    # Handle single Field input
    if isinstance(t, ekd.Field):
        t_lcl, p_lcl = array.lcl(t.values, td.values, p.values if isinstance(p, ekd.Field) else p, method=method)
        t_out = t.set({"values": t_lcl, "parameter.variable": "t_lcl", "parameter.units": "K"})
        p_out = t.set({"values": p_lcl, "parameter.variable": "p_lcl", "parameter.units": "Pa"})
        return t_out, p_out

    t_result = []
    p_result = []
    for t_f, td_f, p_f in zip(t, td, p):
        t_lcl, p_lcl = array.lcl(t_f.values, td_f.values, p_f.values, method=method)
        t_result.append(t_f.set({"values": t_lcl, "parameter.variable": "t_lcl", "parameter.units": "K"}))
        p_result.append(t_f.set({"values": p_lcl, "parameter.variable": "p_lcl", "parameter.units": "Pa"}))
    return ekd.FieldList.from_fields(t_result), ekd.FieldList.from_fields(p_result)


def ept_from_dewpoint(
    t: FieldList | Field,
    td: FieldList | Field,
    p: FieldList | Field | Iterable[float] | float | None = None,
    method: str = "ifs",
) -> FieldList | Field:
    r"""Compute the equivalent potential temperature from dewpoint.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    td : FieldList|Field
        Dewpoint (K).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``t``. Otherwise, if ``t``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t``. If ``t``
        is a Field, ``p`` must be a single Field or a float.
    method : str, optional
        Specify the computation method. The possible values are: "ifs", "bolton35", "bolton39", "bolton43".

    Returns
    -------
    FieldList|Field
        Equivalent potential temperature (K).
        The result has the same type as the input ``t`` and ``td`` (FieldList or Field).


    The actual computation is based on the value of ``method``:

    * "ifs": the formula from the IFS model [IFS-CY47R3-PhysicalProcesses]_ (Chapter 6.11) is used:

        .. math::

            \Theta_{e} = \Theta  exp(\frac{L_{v}  q}{c_{pd}  t_{LCL}})

    * "bolton35": Eq (35) from [Bolton1980]_ is used:


        .. math::

            \Theta_{e} = \Theta (\frac{10^{5}}{p})^{\kappa 0.28 w} exp(\frac{2675 w}{t_{LCL}})

    * "bolton39": Eq (39) from [Bolton1980]_ is used:

        .. math::

            \Theta_{e} =
            t (\frac{10^{5}}{p-e})^{\kappa} (\frac{t}{t_{LCL}})^{0.28 w} exp[(\frac{3036}{t_{LCL}} -
            1.78)w(1+0.448  w)]

    * "bolton43": Eq (43) from [Bolton1980]_ is used:

        .. math::

            \Theta_{e} =
            t (\frac{10^{5}}{p})^{\kappa (1-0.28\; 10^{-3}w)} exp[(\frac{3376}{t_{LCL}} -
            2.54)w(1+0.81w)]

    where:

        * :math:`\Theta` is the :func:`potential_temperature`
        * :math:`t` is the temperature at the start level
        * :math:`t_{LCL}` is the temperature at the Lifting Condensation Level computed
          with :func:`lcl_temperature` using option:

            * method="davis" when ``method`` is "ifs"
            * method="bolton" when ``method`` is "bolton35", "bolton39", or "bolton43"
        * :math:`q` is the specific humidity computed with :func:`specific_humidity_from_dewpoint`
        * :math:`w`: is the mixing ratio computed with :func:`mixing_ratio_from_dewpoint`
        * :math:`e` is the vapour pressure computed with :func:`vapour_pressure_from_mixing_ratio`
        * :math:`L_{v}`: is the latent heat of vaporisation
          (see :data:`earthkit.meteo.constants.Lv`)
        * :math:`c_{pd}` is the specific heat of dry air on constant pressure
          (see :data:`earthkit.meteo.constants.c_pd`)
        * :math:`\kappa = R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`)

    """
    fieldlist_ufunc_kwargs = {"default": "eqpt"}

    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa

    return fieldlist_ufunc(
        array.ept_from_dewpoint, t, td, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, method=method
    )


def ept_from_specific_humidity(
    t: FieldList | Field,
    q: FieldList | Field,
    p: FieldList | Field | Iterable[float] | float | None = None,
    method: str = "ifs",
) -> FieldList | Field:
    r"""Compute the equivalent potential temperature from specific humidity.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    q : FieldList|Field
        Specific humidity (kg/kg).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``t``. Otherwise, if ``t``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t``. If ``t``
        is a Field, ``p`` must be a single Field or a float.
    method : str, optional
        Specify the computation method. The possible values are: "ifs",
        "bolton35", "bolton39", "bolton43". See :func:`ept_from_dewpoint` for details.

    Returns
    -------
    FieldList|Field
        Equivalent potential temperature (K).
        The result has the same type as the input ``t`` and ``q`` (FieldList or Field).


    The computations are the same as in :func:`ept_from_dewpoint`
    (the dewpoint is computed from q with :func:`dewpoint_from_specific_humidity`).

    """
    fieldlist_ufunc_kwargs = {"default": "eqpt"}

    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa

    return fieldlist_ufunc(
        array.ept_from_specific_humidity,
        t,
        q,
        p,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        method=method,
    )


def saturation_ept(
    t: FieldList | Field, p: FieldList | Field | Iterable[float] | float | None = None, method: str = "ifs"
) -> FieldList | Field:
    r"""Compute the saturation equivalent potential temperature.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``t``. Otherwise, if ``t``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t``. If ``t``
        is a Field, ``p`` must be a single Field or a float.
    method : str, optional
        Specifies the computation method. The possible values are: "ifs", "bolton35", "bolton39".

    Returns
    -------
    FieldList|Field
        Saturation equivalent potential temperature (K).
        The result has the same type as the input ``t`` (FieldList or Field).


    The actual computation is based on the ``method``:

    * "ifs": The formula is based on the equivalent potential temperature definition used
       in the IFS model [IFS-CY47R3-PhysicalProcesses]_ (see Chapter 6.11) :

        .. math::

            \Theta_{esat} = \Theta  exp(\frac{L_{v}  q_{sat}}{c_{pd}  t})

    * "bolton35": Eq (35) from [Bolton1980]_ is used:

        .. math::

            \Theta_{e} = \Theta (\frac{10^{5}}{p})^{\kappa 0.28 w_{sat}}  exp(\frac{2675  w_{sat}}{t})

    * "bolton39": Eq (39) from [Bolton1980]_ is used:

        .. math::

            \Theta_{e} =
            t (\frac{10^{5}}{p-e_{sat}})^{\kappa} exp[(\frac{3036}{t} - 1.78)w_{sat}(1+0.448  w_{sat})]

    where:

        * :math:`\Theta` is the :func:`potential_temperature`
        * :math:`e_{sat}` is the :func:`saturation_vapor_pressure`
        * :math:`q_{sat}` is the :func:`saturation_specific_humidity`
        * :math:`w_{sat}` is the :func:`saturation_mixing_ratio`
        * :math:`L_{v}` is the specific latent heat of vaporization (see :data:`earthkit.meteo.constants.Lv`)
        * :math:`c_{pd}` is the specific heat of dry air on constant pressure
          (see :data:`earthkit.meteo.constants.c_pd`)

    """
    fieldlist_ufunc_kwargs = {"default": "sept"}
    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(array.saturation_ept, t, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, method=method)


def temperature_on_moist_adiabat(
    ept: FieldList | Field,
    p: FieldList | Field | Iterable[float] | float | None = None,
    ept_method: str = "ifs",
    t_method: str = "bisect",
) -> FieldList | Field:
    r"""Compute the temperature on a moist adiabat (pseudoadiabat).

    Parameters
    ----------
    ept : FieldList|Field
        Equivalent potential temperature defining the moist adiabat (K)
    p : FieldList|Field|Iterable[float]|float|None
        Pressure on the moist adiabat (Pa). If None, inferred from the
        field metadata of ``ept``. Otherwise, if ``ept``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``ept``. If ``ept``
        is a Field, ``p`` must be a single Field or a float.
    ept_method : str, optional
        Specifies the computation method that was used to compute ``ept``. The possible
        values are: "ifs", "bolton35", "bolton39".
        (See :func:`ept_from_dewpoint` for details.)
    t_method : str, optional
        Specifies the iteration method along the moist adiabat to find the temperature
        for the given ``p`` pressure. The possible values are as follows:

        * "bisect": a bisection method is used as defined in [Stipanuk1973]_
        * "newton": Newtons's method is used as defined by Eq (2.6) in [DaviesJones2008]_.
          For extremely hot and humid conditions (``ept`` > 800 K) depending on
          ``ept_method`` the computation might not be carried out
          and nan will be returned.


    Returns
    -------
    FieldList|Field
        Temperature on the moist adiabat (K). For values where the computation cannot
        be carried out nan is returned. The result has the same type as the input ``ept`` (FieldList or Field).

    """
    fieldlist_ufunc_kwargs = {"default": "t"}
    if p is None:
        p = pressure_from_metadata(ept)  # convert to Pa
    return fieldlist_ufunc(
        array.temperature_on_moist_adiabat,
        ept,
        p,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        ept_method=ept_method,
        t_method=t_method,
    )


def wet_bulb_temperature_from_dewpoint(
    t: FieldList | Field,
    td: FieldList | Field,
    p: FieldList | Field | Iterable[float] | float | None = None,
    ept_method: str = "ifs",
    t_method: str = "bisect",
) -> FieldList | Field:
    r"""Compute the pseudo adiabatic wet bulb temperature from dewpoint.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    td : FieldList|Field
        Dewpoint (K).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``t``. Otherwise, if ``t``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t``. If ``t``
        is a Field, ``p`` must be a single Field or a float.
    ept_method : str, optional
        Specifies the computation method for the equivalent potential temperature.
        The possible values are: "ifs", "bolton35", "bolton39".
        (See :func:`ept_from_dewpoint` for details.)
    t_method : str, optional
        Specifies the method to find the temperature along the moist adiabat defined
        by the equivalent potential temperature. The possible values are as follows:

        * "bisect": :func:`temperature_on_moist_adiabat` with ``t_method`` = "bisect" is used
        * "newton": :func:`temperature_on_moist_adiabat` with ``t_method`` = "newton" is used

    Returns
    -------
    FieldList|Field
        Wet bulb temperature (K). For values where the computation cannot be carried out nan is returned.
        The result has the same type as the input ``t`` and ``td`` (FieldList or Field).


    The computation is based on Normand's rule [Wallace2006]_ (Chapter 3.5.6):

    * first the equivalent potential temperature is computed with the given
      ``ept_method`` (using :func:`ept_from_dewpoint`). This defines the moist adiabat.
    * then the wet bulb potential temperature is determined as the temperature at
      pressure ``p`` on the moist adiabat with the given ``t_method``.

    """
    fieldlist_ufunc_kwargs = {"default": "wbgt"}
    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(
        array.wet_bulb_temperature_from_dewpoint,
        t,
        td,
        p,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        ept_method=ept_method,
        t_method=t_method,
    )


def wet_bulb_temperature_from_specific_humidity(
    t: FieldList | Field,
    q: FieldList | Field,
    p: FieldList | Field | Iterable[float] | float | None = None,
    ept_method: str = "ifs",
    t_method: str = "bisect",
) -> FieldList | Field:
    r"""Compute the pseudo adiabatic wet bulb temperature from specific humidity.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    q : FieldList|Field
        Specific humidity (kg/kg).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``t``. Otherwise, if ``t``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t``. If ``t``
        is a Field, ``p`` must be a single Field or a float.
    ept_method : str, optional
        Specifies the computation method for the equivalent potential temperature.
        The possible values are: "ifs", "bolton35", "bolton39".
        (See :func:`ept_from_dewpoint` for details.)
    t_method : str, optional
        Specifies the method to find the temperature along the moist adiabat
        defined by the equivalent potential temperature. The possible values are
        as follows:

        * "bisect": :func:`temperature_on_moist_adiabat` with ``t_method`` = "bisect" is used
        * "newton": :func:`temperature_on_moist_adiabat` with ``t_method`` = "newton" is used

    Returns
    -------
    FieldList|Field
        Wet bulb temperature (K). For values where the computation cannot be carried out nan is returned.
        The result has the same type as the input ``t`` and ``q`` (FieldList or Field).


    The computation is based on Normand's rule [Wallace2006]_ (Chapter 3.5.6):

    * first the equivalent potential temperature is computed with the given
      ``ept_method`` (using :func:`ept_from_dewpoint`). This defines the moist adiabat.
    * then the wet bulb potential temperature is determined as the temperature at
      pressure ``p`` on the moist adiabat with the given ``t_method``.

    """
    fieldlist_ufunc_kwargs = {"default": "wbgt"}
    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(
        array.wet_bulb_temperature_from_specific_humidity,
        t,
        q,
        p,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        ept_method=ept_method,
        t_method=t_method,
    )


def wet_bulb_potential_temperature_from_dewpoint(
    t: FieldList | Field,
    td: FieldList | Field,
    p: FieldList | Field | Iterable[float] | float | None = None,
    ept_method: str = "ifs",
    t_method: str = "direct",
) -> FieldList | Field:
    r"""Compute the pseudo adiabatic wet bulb potential temperature from dewpoint.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    td : FieldList|Field
        Dewpoint (K).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``t``. Otherwise, if ``t``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t``. If ``t``
        is a Field, ``p`` must be a single Field or a float.
    ept_method : str, optional
        Specifies the computation method for the equivalent potential temperature.
        The possible values are: "ifs", "bolton35", "bolton39".
        (See :func:`ept_from_dewpoint` for details.)
    t_method : str, optional
        Specifies the method to find the temperature along the moist adiabat defined
        by the equivalent potential temperature. The possible values are as follows:

        * "direct": the rational formula defined by Eq (3.8) in [DaviesJones2008]_ is used
        * "bisect": :func:`temperature_on_moist_adiabat` with ``t_method`` = "bisect" is used
        * "newton": :func:`temperature_on_moist_adiabat` with ``t_method`` = "newton" is used

    Returns
    -------
    FieldList|Field
        Wet bulb potential temperature (K). For values where the computation cannot be carried out nan is returned.
        The result has the same type as the input ``t`` and ``td`` (FieldList or Field).


    The computation is based on Normand's rule [Wallace2006]_ (Chapter 3.5.6):

    * first the equivalent potential temperature is computed with the given
      ``ept_method`` (using :func:`ept_from_dewpoint`). This defines the moist adiabat.
    * then the wet bulb potential temperature is determined as the temperature at
      pressure :math:`10^{5}` Pa on the moist adiabat with the given ``t_method``.

    """
    fieldlist_ufunc_kwargs = {"default": "wbgpt"}
    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(
        array.wet_bulb_potential_temperature_from_dewpoint,
        t,
        td,
        p,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        ept_method=ept_method,
        t_method=t_method,
    )


def wet_bulb_potential_temperature_from_specific_humidity(
    t: FieldList | Field,
    q: FieldList | Field,
    p: FieldList | Field | Iterable[float] | float | None = None,
    ept_method: str = "ifs",
    t_method: str = "direct",
) -> FieldList | Field:
    r"""Compute the pseudo adiabatic wet bulb potential temperature from specific humidity.

    Parameters
    ----------
    t : FieldList|Field
        Temperature (K).
    q : FieldList|Field
        Specific humidity (kg/kg).
    p : FieldList|Field|Iterable[float]|float|None
        Pressure (Pa). If None, inferred from the
        field metadata of ``t``. Otherwise, if ``t``
        is a FieldList ``p`` must be a FieldList or an
        array-like of the same length as ``t``. If ``t``
        is a Field, ``p`` must be a single Field or a float.
    ept_method : str, optional
        Specifies the computation method for the equivalent potential temperature.
        The possible values are: "ifs", "bolton35", "bolton39".
        (See :func:`ept_from_dewpoint` for details.)
    t_method : str, optional
        Specifies the method to find the temperature along the moist adiabat
        defined by the equivalent potential temperature. The possible values are as follows:

        * "direct": the rational formula defined by Eq (3.8) in [DaviesJones2008]_ is used
        * "bisect": :func:`temperature_on_moist_adiabat` with ``t_method`` = "bisect" is used
        * "newton": :func:`temperature_on_moist_adiabat` with ``t_method`` = "newton" is used

    Returns
    -------
    FieldList|Field
        Wet bulb potential temperature (K). For values where the computation cannot be carried out nan is returned.
        The result has the same type as the input ``t`` and ``q`` (FieldList or Field).


    The computations are the same as in
    :func:`wet_bulb_potential_temperature_from_dewpoint`
    (the dewpoint is computed from q with :func:`dewpoint_from_specific_humidity`).

    """
    fieldlist_ufunc_kwargs = {"default": "wbgpt"}
    if p is None:
        p = pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(
        array.wet_bulb_potential_temperature_from_specific_humidity,
        t,
        q,
        p,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        ept_method=ept_method,
        t_method=t_method,
    )


def specific_gas_constant(q: FieldList | Field) -> FieldList | Field:
    r"""Compute the specific gas constant of moist air.

    Specific content of cloud particles and hydrometeors are neglected.

    Parameters
    ----------
    q : FieldList|Field
        Specific humidity (kg/kg)

    Returns
    -------
    FieldList|Field
        Specific gas constant of moist air (J kg-1 K-1).
        The result has the same type as the input ``q`` (FieldList or Field).


    The computation is based on the following formula:

    .. math::

        R = R_{d} + (R_{v} - R_{d}) qss

    where:

        * :math:`R_{d}` is the gas constant for dry air (see :data:`earthkit.meteo.constants.Rd`)
        * :math:`R_{v}` is the gas constant for water vapour (see :data:`earthkit.meteo.constants.Rv`)

    """
    fieldlist_ufunc_kwargs = {"default": "R", "param_unit": "J kg-1 K-1"}
    return fieldlist_ufunc(array.specific_gas_constant, q, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)
