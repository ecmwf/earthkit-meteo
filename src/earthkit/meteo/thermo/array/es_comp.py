# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from earthkit.meteo.utils.array import array_namespace

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
    ns = array_namespace(t)
    if phase == "mixed":
        return _es_mixed(t, ns)
    elif phase == "water":
        return _es_water(t, ns)
    elif phase == "ice":
        return _es_ice(t, ns)


def compute_slope(t, phase):
    ns = array_namespace(t)
    if phase == "mixed":
        return _es_mixed_slope(t, ns)
    elif phase == "water":
        return _es_water_slope(t, ns)
    elif phase == "ice":
        return _es_ice_slope(t, ns)


def compute_t_from_es(es):
    ns = array_namespace(es)
    v = ns.log(es / C1)
    return (v * C4W - C3W * T0) / (v - C3W)


def _es_water(t, ns):
    return C1 * ns.exp(C3W * (t - T0) / (t - C4W))


def _es_ice(t, ns):
    return C1 * ns.exp(C3I * (t - T0) / (t - C4I))


def _es_mixed(t, ns):
    # Fraction of liquid water (=alpha):
    #   t <= ti => alpha=0
    #   t > ti and t < t0 => alpha=(t-ti)/(t0-ti))^2
    #   t >= t0 => alpha=1
    #
    # svp is interpolated between the ice and water phases:
    #   svp = alpha * es_water + (1.0 - alpha) * es_ice

    t = ns.asarray(t)
    svp = ns.zeros(t.shape)

    # ice range
    i_mask = t <= TI
    svp[i_mask] = _es_ice(t[i_mask], ns)

    # water range
    w_mask = t >= T0
    svp[w_mask] = _es_water(t[w_mask], ns)

    # mixed range
    m_mask = ~(i_mask | w_mask)
    alpha = ns.square(t[m_mask] - TI) / ns.square(T0 - TI)
    svp[m_mask] = alpha * _es_water(t[m_mask], ns) + (1.0 - alpha) * _es_ice(t[m_mask], ns)
    return svp


def _es_water_slope(t, ns):
    return _es_water(t, ns) * (C3W * (T0 - C4W)) / ns.square(t - C4W)


def _es_ice_slope(t, ns):
    return _es_ice(t, ns) * (C3I * (T0 - C4I)) / ns.square(t - C4I)


def _es_mixed_slope(t, ns):
    t = ns.asarray(t)
    d_svp = ns.zeros(t.shape)

    # ice range
    i_mask = t <= TI
    d_svp[i_mask] = _es_ice_slope(t[i_mask], ns)

    # water range
    w_mask = t >= T0
    d_svp[w_mask] = _es_water_slope(t[w_mask], ns)

    # mixed range
    m_mask = ~(i_mask | w_mask)
    alpha = ns.square(t[m_mask] - TI) / ns.square(T0 - TI)
    d_alpha = (2.0 / ns.square(T0 - TI)) * (t[m_mask] - TI)
    t_m = t[m_mask]
    d_svp[m_mask] = (
        d_alpha * _es_water(t_m, ns)
        + alpha * _es_water_slope(t_m, ns)
        - d_alpha * _es_ice(t_m, ns)
        + (1.0 - alpha) * _es_ice_slope(t_m, ns)
    )
    return d_svp
