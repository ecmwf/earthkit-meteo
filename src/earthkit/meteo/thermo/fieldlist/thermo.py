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

from earthkit.data import FieldList  # type: ignore[import]

from earthkit.meteo.utils.decorators import fieldlist_ufunc

from .. import array


def _pressure_from_metadata(fields: FieldList) -> list[float]:
    """Infer pressure in Pa from field metadata."""
    from earthkit.utils.units import Units

    return [
        (f.get("vertical.level") * ((f.get("vertical.units", Units.from_any("hPa"))).to_pint())).to("Pa").magnitude
        for f in fields
    ]


def specific_humidity_from_mixing_ratio(w: FieldList) -> FieldList:
    r"""Compute the specific humidity from mixing ratio.

    Parameters
    ----------
    w : FieldList
        Mixing ratio (kg/kg)

    Returns
    -------
    FieldList
        Specific humidity (kg/kg)

    """
    fieldlist_ufunc_kwargs = {"default": "q"}
    return fieldlist_ufunc(array.specific_humidity_from_mixing_ratio, w, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def mixing_ratio_from_specific_humidity(q: FieldList) -> FieldList:
    r"""Compute the mixing ratio from specific humidity.

    Parameters
    ----------
    q : FieldList
        Specific humidity (kg/kg)

    Returns
    -------
    FieldList
        Mixing ratio (kg/kg)

    """
    fieldlist_ufunc_kwargs = {"default": "w"}
    return fieldlist_ufunc(array.mixing_ratio_from_specific_humidity, q, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def vapour_pressure_from_specific_humidity(q: FieldList, p: FieldList) -> FieldList:
    r"""Compute the vapour pressure from specific humidity.

    Parameters
    ----------
    q : FieldList
        Specific humidity (kg/kg)
    p : FieldList
        Pressure (Pa)

    Returns
    -------
    FieldList
        Vapour pressure (Pa)

    """
    fieldlist_ufunc_kwargs = {"default": "e", "param_unit": "Pa"}
    if p is None:
        p = _pressure_from_metadata(q)  # convert to Pa

    return fieldlist_ufunc(
        array.vapour_pressure_from_specific_humidity, q, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def vapour_pressure_from_mixing_ratio(w: FieldList, p: FieldList) -> FieldList:
    r"""Compute the vapour pressure from mixing ratio.

    Parameters
    ----------
    w : FieldList
        Mixing ratio (kg/kg)
    p : FieldList
        Pressure (Pa)

    Returns
    -------
    FieldList
        Vapour pressure (Pa)

    """
    fieldlist_ufunc_kwargs = {"default": "e", "param_unit": "Pa"}
    if p is None:
        p = _pressure_from_metadata(w)  # convert to Pa

    return fieldlist_ufunc(array.vapour_pressure_from_mixing_ratio, w, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def specific_humidity_from_vapour_pressure(e: FieldList, p: FieldList, eps: float = 1e-4) -> FieldList:
    r"""Compute the specific humidity from vapour pressure.

    Parameters
    ----------
    e : FieldList
        Vapour pressure (Pa)
    p : FieldList
        Pressure (Pa)
    eps : float, optional
        Where p - e < ``eps`` nan is returned.

    Returns
    -------
    FieldList
        Specific humidity (kg/kg)

    """
    fieldlist_ufunc_kwargs = {"default": "q", "param_unit": "kg/kg"}
    if p is None:
        p = _pressure_from_metadata(e)  # convert to Pa

    return fieldlist_ufunc(
        array.specific_humidity_from_vapour_pressure,
        e,
        p,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        eps=eps,
    )


def mixing_ratio_from_vapour_pressure(e: FieldList, p: FieldList, eps: float = 1e-4) -> FieldList:
    r"""Compute the mixing ratio from vapour pressure.

    Parameters
    ----------
    e : FieldList
        Vapour pressure (Pa)
    p : FieldList
        Pressure (Pa)
    eps : float, optional
        Where p - e < ``eps`` nan is returned.

    Returns
    -------
    FieldList
        Mixing ratio (kg/kg)

    """
    fieldlist_ufunc_kwargs = {"default": "w", "param_unit": "kg/kg"}
    if p is None:
        p = _pressure_from_metadata(e)  # convert to Pa

    return fieldlist_ufunc(
        array.mixing_ratio_from_vapour_pressure, e, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, eps=eps
    )


def saturation_vapour_pressure(t: FieldList, phase: str = "mixed") -> FieldList:
    r"""Compute the saturation vapour pressure from temperature with respect to a phase.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    phase : str, optional
        Define the phase with respect to the saturation vapour pressure is computed.
        It is either "water", "ice" or "mixed".

    Returns
    -------
    FieldList
        Saturation vapour pressure (Pa)

    """
    fieldlist_ufunc_kwargs = {"default": "es", "param_unit": "Pa"}
    return fieldlist_ufunc(
        array.saturation_vapour_pressure, t, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, phase=phase
    )


def saturation_mixing_ratio(t: FieldList, p: FieldList, phase: str = "mixed") -> FieldList:
    r"""Compute the saturation mixing ratio from temperature with respect to a phase.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    p : FieldList
        Pressure (Pa)
    phase : str, optional
        Define the phase with respect to the saturation vapour pressure is computed.
        It is either "water", "ice" or "mixed".

    Returns
    -------
    FieldList
        Saturation mixing ratio (kg/kg)

    """
    fieldlist_ufunc_kwargs = {"default": "ws", "param_unit": "kg/kg"}

    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa

    return fieldlist_ufunc(
        array.saturation_mixing_ratio, t, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, phase=phase
    )


def saturation_specific_humidity(t: FieldList, p: FieldList, phase: str = "mixed") -> FieldList:
    r"""Compute the saturation specific humidity from temperature with respect to a phase.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    p : FieldList
        Pressure (Pa)
    phase : str, optional
        Define the phase with respect to the saturation vapour pressure is computed.
        It is either "water", "ice" or "mixed".

    Returns
    -------
    FieldList
        Saturation specific humidity (kg/kg)

    """
    fieldlist_ufunc_kwargs = {"default": "qs", "param_unit": "kg/kg"}
    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(
        array.saturation_specific_humidity, t, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, phase=phase
    )


def saturation_vapour_pressure_slope(t: FieldList, phase: str = "mixed") -> FieldList:
    r"""Compute the slope of saturation vapour pressure with respect to temperature.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    phase : str, optional
        Define the phase with respect to the computation will be performed.
        It is either "water", "ice" or "mixed".

    Returns
    -------
    FieldList
        Slope of saturation vapour pressure (Pa/K)

    """
    fieldlist_ufunc_kwargs = {"default": "es_slope", "param_unit": "Pa/K"}
    return fieldlist_ufunc(
        array.saturation_vapour_pressure_slope, t, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, phase=phase
    )


def saturation_mixing_ratio_slope(
    t: FieldList,
    p: FieldList,
    phase: str = "mixed",
    eps: float = 1e-4,
) -> FieldList:
    r"""Compute the slope of saturation mixing ratio with respect to temperature.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    p : FieldList
        Pressure (Pa)
    phase : str, optional
        Define the phase with respect to the computation will be performed.
        It is either "water", "ice" or "mixed".
    eps : float, optional
        Where p - es < ``eps`` nan is returned.

    Returns
    -------
    FieldList
        Slope of saturation mixing ratio (kg kg-1 K-1)

    """
    fieldlist_ufunc_kwargs = {"default": "ws_slope", "param_unit": "kg kg-1 K-1"}
    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(
        array.saturation_mixing_ratio_slope,
        t,
        p,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        phase=phase,
        eps=eps,
    )


def saturation_specific_humidity_slope(
    t: FieldList,
    p: FieldList,
    phase: str = "mixed",
    eps: float = 1e-4,
) -> FieldList:
    r"""Compute the slope of saturation specific humidity with respect to temperature.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    p : FieldList
        Pressure (Pa)
    phase : str, optional
        Define the phase with respect to the computation will be performed.
        It is either "water", "ice" or "mixed".
    eps : float, optional
        Where p - es < ``eps`` nan is returned.

    Returns
    -------
    FieldList
        Slope of saturation specific humidity (kg kg-1 K-1)

    """
    fieldlist_ufunc_kwargs = {"default": "qs_slope", "param_unit": "kg kg-1 K-1"}
    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(
        array.saturation_specific_humidity_slope,
        t,
        p,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        phase=phase,
        eps=eps,
    )


def temperature_from_saturation_vapour_pressure(es: FieldList) -> FieldList:
    r"""Compute the temperature from saturation vapour pressure.

    Parameters
    ----------
    es : FieldList
        Saturation vapour pressure (Pa)

    Returns
    -------
    FieldList
        Temperature (K)

    """
    fieldlist_ufunc_kwargs = {"default": "t", "param_unit": "K"}
    return fieldlist_ufunc(
        array.temperature_from_saturation_vapour_pressure, es, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def relative_humidity_from_dewpoint(t: FieldList, td: FieldList) -> FieldList:
    r"""Compute the relative humidity from dewpoint temperature.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    td : FieldList
        Dewpoint (K)

    Returns
    -------
    FieldList
        Relative humidity (%)

    """
    fieldlist_ufunc_kwargs = {"default": "r", "param_unit": "%"}
    return fieldlist_ufunc(array.relative_humidity_from_dewpoint, t, td, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def relative_humidity_from_specific_humidity(t: FieldList, q: FieldList, p: FieldList) -> FieldList:
    r"""Compute the relative humidity from specific humidity.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    q : FieldList
        Specific humidity (kg/kg)
    p : FieldList
        Pressure (Pa)

    Returns
    -------
    FieldList
        Relative humidity (%)

    """
    fieldlist_ufunc_kwargs = {"default": "r", "param_unit": "%"}
    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(
        array.relative_humidity_from_specific_humidity, t, q, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def specific_humidity_from_dewpoint(td: FieldList, p: FieldList) -> FieldList:
    r"""Compute the specific humidity from dewpoint.

    Parameters
    ----------
    td : FieldList
        Dewpoint (K)
    p : FieldList
        Pressure (Pa)

    Returns
    -------
    FieldList
        Specific humidity (kg/kg)

    """
    fieldlist_ufunc_kwargs = {"default": "q", "param_unit": "kg/kg"}
    if p is None:
        p = _pressure_from_metadata(td)  # convert to Pa
    return fieldlist_ufunc(array.specific_humidity_from_dewpoint, td, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def mixing_ratio_from_dewpoint(td: FieldList, p: FieldList) -> FieldList:
    r"""Compute the mixing ratio from dewpoint.

    Parameters
    ----------
    td : FieldList
        Dewpoint (K)
    p : FieldList
        Pressure (Pa)

    Returns
    -------
    FieldList
        Mixing ratio (kg/kg)

    """
    fieldlist_ufunc_kwargs = {"default": "w", "param_unit": "kg/kg"}
    if p is None:
        p = _pressure_from_metadata(td)  # convert to Pa
    return fieldlist_ufunc(array.mixing_ratio_from_dewpoint, td, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def specific_humidity_from_relative_humidity(t: FieldList, r: FieldList, p: FieldList) -> FieldList:
    r"""Compute the specific humidity from relative humidity.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    r : FieldList
        Relative humidity (%)
    p : FieldList
        Pressure (Pa)

    Returns
    -------
    FieldList
        Specific humidity (kg/kg)

    """
    fieldlist_ufunc_kwargs = {"default": "q", "param_unit": "kg/kg"}
    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(
        array.specific_humidity_from_relative_humidity, t, r, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def dewpoint_from_relative_humidity(t: FieldList, r: FieldList) -> FieldList:
    r"""Compute the dewpoint temperature from relative humidity.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    r : FieldList
        Relative humidity (%)

    Returns
    -------
    FieldList
        Dewpoint temperature (K)

    """
    fieldlist_ufunc_kwargs = {"default": "td"}
    return fieldlist_ufunc(array.dewpoint_from_relative_humidity, t, r, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def dewpoint_from_specific_humidity(q: FieldList, p: FieldList) -> FieldList:
    r"""Compute the dewpoint temperature from specific humidity.

    Parameters
    ----------
    q : FieldList
        Specific humidity (kg/kg)
    p : FieldList
        Pressure (Pa)

    Returns
    -------
    FieldList
        Dewpoint temperature (K)

    """
    fieldlist_ufunc_kwargs = {"default": "td", "param_unit": "K"}
    if p is None:
        p = _pressure_from_metadata(q)  # convert to Pa
    return fieldlist_ufunc(array.dewpoint_from_specific_humidity, q, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def virtual_temperature(t: FieldList, q: FieldList) -> FieldList:
    r"""Compute the virtual temperature from temperature and specific humidity.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    q : FieldList
        Specific humidity (kg/kg)

    Returns
    -------
    FieldList
        Virtual temperature (K)

    """
    fieldlist_ufunc_kwargs = {"default": "tv"}
    return fieldlist_ufunc(array.virtual_temperature, t, q, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def virtual_potential_temperature(t: FieldList, q: FieldList, p: FieldList) -> FieldList:
    r"""Compute the virtual potential temperature from temperature and specific humidity.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    q : FieldList
        Specific humidity (kg/kg)
    p : FieldList
        Pressure (Pa)

    Returns
    -------
    FieldList
        Virtual potential temperature (K)

    """
    fieldlist_ufunc_kwargs = {"default": "thv"}

    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa

    return fieldlist_ufunc(array.virtual_potential_temperature, t, q, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def potential_temperature(t: FieldList, p: FieldList | Iterable[float] | None = None) -> FieldList:
    r"""Compute the potential temperature.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    p : FieldList, Iterable[float], or None
        Pressure (Pa). If None, inferred from the field metadata.

    Returns
    -------
    FieldList
        Potential temperature (K)


    The computation is based on the following formula [Wallace2006]_:

    .. math::

       \theta = t \left(\frac{10^{5}}{p}\right)^{\kappa}

    with :math:`\kappa = R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`).

    """
    fieldlist_ufunc_kwargs = {"default": "pt"}

    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa

    return fieldlist_ufunc(array.potential_temperature, t, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


def temperature_from_potential_temperature(th: FieldList, p: FieldList | Iterable[float] | None = None) -> FieldList:
    r"""Compute the temperature from potential temperature.

    Parameters
    ----------
    th : FieldList
        Potential temperature (K)
    p : FieldList, Iterable[float], or None
        Pressure (Pa). If None, inferred from the field metadata.

    Returns
    -------
    FieldList
        Temperature (K)

    """
    fieldlist_ufunc_kwargs = {"default": "t"}

    if p is None:
        p = _pressure_from_metadata(th)

    return fieldlist_ufunc(
        array.temperature_from_potential_temperature, th, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def pressure_on_dry_adiabat(t: FieldList, t_def: FieldList, p_def: FieldList) -> FieldList:
    r"""Compute the pressure on a dry adiabat.

    Parameters
    ----------
    t : FieldList
        Temperature on the dry adiabat (K)
    t_def : FieldList
        Temperature defining the dry adiabat (K)
    p_def : FieldList
        Pressure defining the dry adiabat (Pa)

    Returns
    -------
    FieldList
        Pressure on the dry adiabat (Pa)

    """
    fieldlist_ufunc_kwargs = {"default": "p", "param_unit": "Pa"}
    return fieldlist_ufunc(
        array.pressure_on_dry_adiabat, t, t_def, p_def, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def temperature_on_dry_adiabat(p: FieldList, t_def: FieldList, p_def: FieldList) -> FieldList:
    r"""Compute the temperature on a dry adiabat.

    Parameters
    ----------
    p : FieldList
        Pressure on the dry adiabat (Pa)
    t_def : FieldList
        Temperature defining the dry adiabat (K)
    p_def : FieldList
        Pressure defining the dry adiabat (Pa)

    Returns
    -------
    FieldList
        Temperature on the dry adiabat (K)

    """
    fieldlist_ufunc_kwargs = {"default": "t", "param_unit": "K"}
    return fieldlist_ufunc(
        array.temperature_on_dry_adiabat, p, t_def, p_def, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def lcl_temperature(t: FieldList, td: FieldList, method: str = "davies") -> FieldList:
    r"""Compute the Lifting Condensation Level (LCL) temperature from dewpoint.

    Parameters
    ----------
    t : FieldList
        Temperature at the start level (K)
    td : FieldList
        Dewpoint at the start level (K)
    method : str, optional
        The computation method: "davies" or "bolton".

    Returns
    -------
    FieldList
        Temperature of the LCL (K)

    """
    fieldlist_ufunc_kwargs = {"default": "t_lcl"}
    return fieldlist_ufunc(array.lcl_temperature, t, td, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, method=method)


def lcl(t: FieldList, td: FieldList, p: FieldList, method: str = "davies") -> tuple[FieldList, FieldList]:
    r"""Compute the temperature and pressure of the Lifting Condensation Level (LCL) from dewpoint.

    Parameters
    ----------
    t : FieldList
        Temperature at the start level (K)
    td : FieldList
        Dewpoint at the start level (K)
    p : FieldList
        Pressure at the start level (Pa)
    method : str, optional
        The computation method: "davies" or "bolton".

    Returns
    -------
    FieldList
        Temperature of the LCL (K)
    FieldList
        Pressure of the LCL (Pa)

    """
    import earthkit.data as ekd

    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa

    t_result = []
    p_result = []
    for t_f, td_f, p_f in zip(t, td, p):
        t_lcl, p_lcl = array.lcl(t_f.values, td_f.values, p_f.values, method=method)
        t_result.append(t_f.set({"values": t_lcl, "parameter.variable": "t_lcl", "parameter.units": "K"}))
        p_result.append(t_f.set({"values": p_lcl, "parameter.variable": "p_lcl", "parameter.units": "Pa"}))
    return ekd.FieldList.from_fields(t_result), ekd.FieldList.from_fields(p_result)


def ept_from_dewpoint(t: FieldList, td: FieldList, p: FieldList, method: str = "ifs") -> FieldList:
    r"""Compute the equivalent potential temperature from dewpoint.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    td : FieldList
        Dewpoint (K)
    p : FieldList
        Pressure (Pa)
    method : str, optional
        Computation method: "ifs", "bolton35", "bolton39", "bolton43".

    Returns
    -------
    FieldList
        Equivalent potential temperature (K)

    """
    fieldlist_ufunc_kwargs = {"default": "ept"}

    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa

        return fieldlist_ufunc(
            array.ept_from_dewpoint, t, td, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, method=method
        )


def ept_from_specific_humidity(t: FieldList, q: FieldList, p: FieldList, method: str = "ifs") -> FieldList:
    r"""Compute the equivalent potential temperature from specific humidity.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    q : FieldList
        Specific humidity (kg/kg)
    p : FieldList
        Pressure (Pa)
    method : str, optional
        Computation method: "ifs", "bolton35", "bolton39", "bolton43".

    Returns
    -------
    FieldList
        Equivalent potential temperature (K)

    """
    fieldlist_ufunc_kwargs = {"default": "ept"}

    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa

    return fieldlist_ufunc(
        array.ept_from_specific_humidity,
        t,
        q,
        p,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        method=method,
    )


def saturation_ept(t: FieldList, p: FieldList, method: str = "ifs") -> FieldList:
    r"""Compute the saturation equivalent potential temperature.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    p : FieldList
        Pressure (Pa)
    method : str, optional
        Computation method: "ifs", "bolton35", "bolton39".

    Returns
    -------
    FieldList
        Saturation equivalent potential temperature (K)

    """
    fieldlist_ufunc_kwargs = {"default": "ept_sat"}
    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(array.saturation_ept, t, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs, method=method)


def temperature_on_moist_adiabat(
    ept: FieldList,
    p: FieldList,
    ept_method: str = "ifs",
    t_method: str = "bisect",
) -> FieldList:
    r"""Compute the temperature on a moist adiabat (pseudoadiabat).

    Parameters
    ----------
    ept : FieldList
        Equivalent potential temperature defining the moist adiabat (K)
    p : FieldList
        Pressure on the moist adiabat (Pa)
    ept_method : str, optional
        Computation method used to compute ``ept``: "ifs", "bolton35", "bolton39".
    t_method : str, optional
        Iteration method: "bisect" or "newton".

    Returns
    -------
    FieldList
        Temperature on the moist adiabat (K)

    """
    fieldlist_ufunc_kwargs = {"default": "t"}
    if p is None:
        p = _pressure_from_metadata(ept)  # convert to Pa
    return fieldlist_ufunc(
        array.temperature_on_moist_adiabat,
        ept,
        p,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        ept_method=ept_method,
        t_method=t_method,
    )


def wet_bulb_temperature_from_dewpoint(
    t: FieldList,
    td: FieldList,
    p: FieldList,
    ept_method: str = "ifs",
    t_method: str = "bisect",
) -> FieldList:
    r"""Compute the pseudo adiabatic wet bulb temperature from dewpoint.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    td : FieldList
        Dewpoint (K)
    p : FieldList
        Pressure (Pa)
    ept_method : str, optional
        Computation method for equivalent potential temperature: "ifs", "bolton35", "bolton39".
    t_method : str, optional
        Iteration method: "bisect" or "newton".

    Returns
    -------
    FieldList
        Wet bulb temperature (K)

    """
    fieldlist_ufunc_kwargs = {"default": "wbt"}
    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa
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
    t: FieldList,
    q: FieldList,
    p: FieldList,
    ept_method: str = "ifs",
    t_method: str = "bisect",
) -> FieldList:
    r"""Compute the pseudo adiabatic wet bulb temperature from specific humidity.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    q : FieldList
        Specific humidity (kg/kg)
    p : FieldList
        Pressure (Pa)
    ept_method : str, optional
        Computation method for equivalent potential temperature: "ifs", "bolton35", "bolton39".
    t_method : str, optional
        Iteration method: "bisect" or "newton".

    Returns
    -------
    FieldList
        Wet bulb temperature (K)

    """
    fieldlist_ufunc_kwargs = {"default": "wbt"}
    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa
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
    t: FieldList,
    td: FieldList,
    p: FieldList,
    ept_method: str = "ifs",
    t_method: str = "direct",
) -> FieldList:
    r"""Compute the pseudo adiabatic wet bulb potential temperature from dewpoint.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    td : FieldList
        Dewpoint (K)
    p : FieldList
        Pressure (Pa)
    ept_method : str, optional
        Computation method for equivalent potential temperature: "ifs", "bolton35", "bolton39".
    t_method : str, optional
        Iteration method: "direct", "bisect", or "newton".

    Returns
    -------
    FieldList
        Wet bulb potential temperature (K)

    """
    fieldlist_ufunc_kwargs = {"default": "wbpt"}
    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa
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
    t: FieldList,
    q: FieldList,
    p: FieldList,
    ept_method: str = "ifs",
    t_method: str = "direct",
) -> FieldList:
    r"""Compute the pseudo adiabatic wet bulb potential temperature from specific humidity.

    Parameters
    ----------
    t : FieldList
        Temperature (K)
    q : FieldList
        Specific humidity (kg/kg)
    p : FieldList
        Pressure (Pa)
    ept_method : str, optional
        Computation method for equivalent potential temperature: "ifs", "bolton35", "bolton39".
    t_method : str, optional
        Iteration method: "direct", "bisect", or "newton".

    Returns
    -------
    FieldList
        Wet bulb potential temperature (K)

    """
    fieldlist_ufunc_kwargs = {"default": "wbpt"}
    if p is None:
        p = _pressure_from_metadata(t)  # convert to Pa
    return fieldlist_ufunc(
        array.wet_bulb_potential_temperature_from_specific_humidity,
        t,
        q,
        p,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        ept_method=ept_method,
        t_method=t_method,
    )


def specific_gas_constant(q: FieldList) -> FieldList:
    r"""Compute the specific gas constant of moist air.

    Parameters
    ----------
    q : FieldList
        Specific humidity (kg/kg)

    Returns
    -------
    FieldList
        Specific gas constant of moist air (J kg-1 K-1)

    """
    fieldlist_ufunc_kwargs = {"default": "R", "param_unit": "J kg-1 K-1"}
    return fieldlist_ufunc(array.specific_gas_constant, q, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)
