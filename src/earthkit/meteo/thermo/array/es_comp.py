# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

# Methods implementing the computations related to saturation vapour pressure and its slope

from earthkit.utils.array import array_namespace

C1 = 611.21
C3W = 17.502
C4W = 32.19
C3I = 22.587
C4I = -0.7
T0 = 273.16
TI = T0 - 23

PHASES = ["mixed", "water", "ice"]


def check_phase(phase):
    if phase not in PHASES:
        raise ValueError(f"saturation_vapour_pressure(): invalid phase={phase}! Allowed values = {PHASES}")
    return True


def compute_es(t, phase):
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
    xp = array_namespace(t)
    if phase == "mixed":
        return _es_mixed(t, xp)
    elif phase == "water":
        return _es_water(t, xp)
    elif phase == "ice":
        return _es_ice(t, xp)


def compute_slope(t, phase):
    r"""Computes the slope of saturation vapour pressure with respect to temperature.

    Parameters
    ----------
    t: array-like
        Temperature (K)
    phase: str, optional
        Defines the phase with respect to the computation will be performed.
        It is either “water”, “ice” or “mixed”. See :func:`saturation_vapour_pressure`
        for details.

    Returns
    -------
    array-like
        Slope of saturation vapour pressure (Pa/K)

    """
    xp = array_namespace(t)
    if phase == "mixed":
        return _es_mixed_slope(t, xp)
    elif phase == "water":
        return _es_water_slope(t, xp)
    elif phase == "ice":
        return _es_ice_slope(t, xp)


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


    The computation is always based on the "water" phase of
    the :func:`saturation_vapour_pressure` formulation irrespective of the
    phase ``es`` was computed to.

    """
    xp = array_namespace(es)
    v = xp.log(es / C1)
    return (v * C4W - C3W * T0) / (v - C3W)


def _es_water(t, xp):
    return C1 * xp.exp(C3W * (t - T0) / (t - C4W))


def _es_ice(t, xp):
    return C1 * xp.exp(C3I * (t - T0) / (t - C4I))


def _es_mixed(t, xp):
    # Fraction of liquid water (=alpha):
    #   t <= ti => alpha=0
    #   t > ti and t < t0 => alpha=(t-ti)/(t0-ti))^2
    #   t >= t0 => alpha=1
    #
    # svp is interpolated between the ice and water phases:
    #   svp = alpha * es_water + (1.0 - alpha) * es_ice

    t = xp.asarray(t)
    svp = xp.zeros(t.shape, dtype=t.dtype)

    # ice range
    i_mask = t <= TI
    svp[i_mask] = _es_ice(t[i_mask], xp)

    # water range
    w_mask = t >= T0
    svp[w_mask] = _es_water(t[w_mask], xp)

    # mixed range
    m_mask = ~(i_mask | w_mask)

    alpha = xp.square((t[m_mask] - TI) / (T0 - TI))
    svp[m_mask] = alpha * _es_water(t[m_mask], xp) + (1.0 - alpha) * _es_ice(t[m_mask], xp)
    return svp


def _es_water_slope(t, xp):
    return _es_water(t, xp) * (C3W * (T0 - C4W)) / xp.square(t - C4W)


def _es_ice_slope(t, xp):
    return _es_ice(t, xp) * (C3I * (T0 - C4I)) / xp.square(t - C4I)


def _es_mixed_slope(t, xp):
    t = xp.asarray(t)
    d_svp = xp.zeros(t.shape, dtype=t.dtype)

    # ice range
    i_mask = t <= TI
    d_svp[i_mask] = _es_ice_slope(t[i_mask], xp)

    # water range
    w_mask = t >= T0
    d_svp[w_mask] = _es_water_slope(t[w_mask], xp)

    # mixed range
    m_mask = ~(i_mask | w_mask)
    alpha = xp.square((t[m_mask] - TI) / (T0 - TI))
    d_alpha = (2.0 / (T0 - TI) ** 2) * (t[m_mask] - TI)
    t_m = t[m_mask]
    d_svp[m_mask] = (
        d_alpha * _es_water(t_m, xp)
        + alpha * _es_water_slope(t_m, xp)
        - d_alpha * _es_ice(t_m, xp)
        + (1.0 - alpha) * _es_ice_slope(t_m, xp)
    )
    return d_svp
