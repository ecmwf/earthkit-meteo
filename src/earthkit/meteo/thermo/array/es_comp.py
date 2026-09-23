# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

# Methods implementing the computations related to saturation vapour pressure and its slope

from earthkit.utils.array import array_namespace

from earthkit.meteo.constants import T0, T_C2K

# IFS coefficients
C1 = 611.21
C3W = 17.502
C4W = 32.19
C3I = 22.587
C4I = -0.7
TI = T0 - 23

# Huang (2018) coefficients (a, b, c, d, n) of es = exp(a - b / (tc + c)) / (tc + d)^n,
# where tc is the temperature in Celsius
HUANG_W = (34.494, 4924.99, 237.1, 105.0, 1.57)
HUANG_I = (43.494, 6545.8, 278.0, 868.0, 2.0)

PHASES = ["mixed", "water", "ice"]
METHODS = ["ifs", "huang"]


def check_phase(phase):
    if phase not in PHASES:
        raise ValueError(f"saturation_vapour_pressure(): invalid phase={phase}! Allowed values = {PHASES}")
    return True


def check_method(method):
    if method not in METHODS:
        raise ValueError(f"saturation_vapour_pressure(): invalid method={method}! Allowed values = {METHODS}")
    return True


def compute_es(t, phase, method="ifs"):
    r"""Compute the saturation vapour pressure from temperature with respect to a phase.

    Parameters
    ----------
    t: array-like
        Temperature (K)
    phase: str, optional
        Define the phase with respect to the saturation vapour pressure is computed.
        It is either “water”, “ice” or “mixed”.
    method: str, optional
        The computation method: "ifs" or "huang".

    Returns
    -------
    array-like
        Saturation vapour pressure (Pa)


    The actual computation is based on the ``method``:

    * "ifs": the algorithm taken from the IFS model [IFS-CY47R3-PhysicalProcesses]_ (see Chapter 12)
      is used. It uses the following formula when ``phase`` is "water" or "ice":

        .. math::

            e_{sat} = a_{1}exp \left(a_{3}\frac{t-273.16}{t-a_{4}}\right)

      where the parameters are set as follows:

      * ``phase`` = "water": :math:`a_{1}` =611.21 Pa, :math:`a_{3}` =17.502 and :math:`a_{4}` =32.19 K
      * ``phase`` = "ice": :math:`a_{1}` =611.21 Pa, :math:`a_{3}` =22.587 and :math:`a_{4}` =-0.7 K

    * "huang": the formulas by [Huang2018]_ are used when ``phase`` is "water" or "ice":

        .. math::

            e_{wsat} = \frac{exp \left(34.494 - \frac{4924.99}{t_{c} + 237.1}\right)}{(t_{c} + 105)^{1.57}}

            e_{isat} = \frac{exp \left(43.494 - \frac{6545.8}{t_{c} + 278}\right)}{(t_{c} + 868)^{2}}

      where :math:`t_{c} = t - 273.15` is the temperature in °C.

    When ``phase`` is "mixed" the formula is based on the value of ``t`` (for both methods):

    * if :math:`t <= t_{i}`: the formula for ``phase`` = "ice" is used (:math:`t_{i} = 250.16 K`)
    * if :math:`t >= t_{0}`: the formula for ``phase`` = "water" is used (:math:`t_{0} = 273.16 K`)
    * for the range :math:`t_{i} < t < t_{0}` an interpolation is used between the "ice" and "water" phases:

    .. math::

        \alpha(t) e_{wsat}(t) + (1 - \alpha(t)) e_{isat}(t)

    with :math:`\alpha(t) = (\frac{t-t_{i}}{t_{0}-t_{i}})^2`.

    """
    check_method(method)
    xp = array_namespace(t)
    es_water, es_ice = _ES_FUNCS[method]
    if phase == "mixed":
        return _es_mixed(t, xp, es_water, es_ice)
    elif phase == "water":
        return es_water(t, xp)
    elif phase == "ice":
        return es_ice(t, xp)


def compute_slope(t, phase, method="ifs"):
    r"""Computes the slope of saturation vapour pressure with respect to temperature.

    Parameters
    ----------
    t: array-like
        Temperature (K)
    phase: str, optional
        Defines the phase with respect to the computation will be performed.
        It is either “water”, “ice” or “mixed”. See :func:`saturation_vapour_pressure`
        for details.
    method: str, optional
        The computation method: "ifs" or "huang". See :func:`saturation_vapour_pressure`
        for details.

    Returns
    -------
    array-like
        Slope of saturation vapour pressure (Pa/K)

    """
    check_method(method)
    xp = array_namespace(t)
    es_water, es_ice = _ES_FUNCS[method]
    es_water_slope, es_ice_slope = _ES_SLOPE_FUNCS[method]
    if phase == "mixed":
        return _es_mixed_slope(t, xp, es_water, es_ice, es_water_slope, es_ice_slope)
    elif phase == "water":
        return es_water_slope(t, xp)
    elif phase == "ice":
        return es_ice_slope(t, xp)


def compute_t_from_es(es):
    r"""Compute the temperature from saturation vapour pressure.

    Parameters
    ----------
    es: array-like
        :func:`saturation_vapour_pressure` (Pa)

    Returns
    -------
    array-like
        Temperature (K). For zero ``es`` values returns nan.


    The computation is always based on the "water" phase of the "ifs" method of
    the :func:`saturation_vapour_pressure` formulation irrespective of the
    phase and method ``es`` was computed with.

    """
    xp = array_namespace(es)
    v = xp.log(es / C1)
    return (v * C4W - C3W * T0) / (v - C3W)


def _es_water(t, xp):
    return C1 * xp.exp(C3W * (t - T0) / (t - C4W))


def _es_ice(t, xp):
    return C1 * xp.exp(C3I * (t - T0) / (t - C4I))


def _es_huang(t, xp, a, b, c, d, n):
    tc = t - T_C2K
    return xp.exp(a - b / (tc + c)) / (tc + d) ** n


def _es_water_huang(t, xp):
    return _es_huang(t, xp, *HUANG_W)


def _es_ice_huang(t, xp):
    return _es_huang(t, xp, *HUANG_I)


def _es_mixed(t, xp, es_water, es_ice):
    # Fraction of liquid water (=alpha):
    #   t <= ti => alpha=0
    #   t > ti and t < t0 => alpha=(t-ti)/(t0-ti))^2
    #   t >= t0 => alpha=1
    #
    # svp is interpolated between the ice and water phases:
    #   svp = alpha * es_water + (1.0 - alpha) * es_ice

    t = xp.asarray(t)
    svp = xp.zeros_like(t, dtype=t.dtype)

    # ice range
    i_mask = t <= TI
    svp[i_mask] = es_ice(t[i_mask], xp)

    # water range
    w_mask = t >= T0
    svp[w_mask] = es_water(t[w_mask], xp)

    # mixed range
    m_mask = ~(i_mask | w_mask)

    alpha = xp.square((t[m_mask] - TI) / (T0 - TI))
    svp[m_mask] = alpha * es_water(t[m_mask], xp) + (1.0 - alpha) * es_ice(t[m_mask], xp)
    return svp


def _es_water_slope(t, xp):
    return _es_water(t, xp) * (C3W * (T0 - C4W)) / xp.square(t - C4W)


def _es_ice_slope(t, xp):
    return _es_ice(t, xp) * (C3I * (T0 - C4I)) / xp.square(t - C4I)


def _es_huang_slope(t, xp, a, b, c, d, n):
    tc = t - T_C2K
    return _es_huang(t, xp, a, b, c, d, n) * (b / xp.square(tc + c) - n / (tc + d))


def _es_water_huang_slope(t, xp):
    return _es_huang_slope(t, xp, *HUANG_W)


def _es_ice_huang_slope(t, xp):
    return _es_huang_slope(t, xp, *HUANG_I)


def _es_mixed_slope(t, xp, es_water, es_ice, es_water_slope, es_ice_slope):
    t = xp.asarray(t)
    d_svp = xp.zeros_like(t, dtype=t.dtype)

    # ice range
    i_mask = t <= TI
    d_svp[i_mask] = es_ice_slope(t[i_mask], xp)

    # water range
    w_mask = t >= T0
    d_svp[w_mask] = es_water_slope(t[w_mask], xp)

    # mixed range
    m_mask = ~(i_mask | w_mask)
    alpha = xp.square((t[m_mask] - TI) / (T0 - TI))
    d_alpha = (2.0 / (T0 - TI) ** 2) * (t[m_mask] - TI)
    t_m = t[m_mask]
    d_svp[m_mask] = (
        d_alpha * es_water(t_m, xp)
        + alpha * es_water_slope(t_m, xp)
        - d_alpha * es_ice(t_m, xp)
        + (1.0 - alpha) * es_ice_slope(t_m, xp)
    )
    return d_svp


# Water and ice saturation vapour pressure functions for each method
_ES_FUNCS = {
    "ifs": (_es_water, _es_ice),
    "huang": (_es_water_huang, _es_ice_huang),
}

# Water and ice saturation vapour pressure slope functions for each method
_ES_SLOPE_FUNCS = {
    "ifs": (_es_water_slope, _es_ice_slope),
    "huang": (_es_water_huang_slope, _es_ice_huang_slope),
}
