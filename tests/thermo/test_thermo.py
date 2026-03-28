# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from pathlib import Path

import numpy as np
import pytest
from earthkit.utils.array import array_namespace
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo import thermo
from earthkit.meteo.utils.testing import NO_XARRAY


def _signature(obj):
    if isinstance(obj, tuple):
        return tuple(_signature(x) for x in obj)

    if hasattr(obj, "dims") and hasattr(obj, "shape"):
        return ("xarray", tuple(obj.dims), tuple(obj.shape))

    xp = array_namespace(obj)
    arr = xp.asarray(obj)
    return ("array", tuple(arr.shape))


def _da(values):
    import xarray as xr

    return xr.DataArray(np.asarray(values))


def _load_input():
    path = Path(__file__).resolve().parents[1] / "data" / "t_hum_p_data.csv"
    data = np.genfromtxt(path, delimiter=",", names=True)
    index = [250, 300]
    return {name: data[name][index] for name in data.dtype.names}


def _ops(impl, celsius, kelvin, t, td, r, q, p):
    w = impl.mixing_ratio_from_specific_humidity(q)
    e = impl.vapour_pressure_from_specific_humidity(q, p)
    es = impl.saturation_vapour_pressure(t)
    th = impl.potential_temperature(t, p)
    ept = impl.ept_from_dewpoint(t, td, p)

    return {
        "specific_humidity_from_mixing_ratio": ((w,), {}),
        "mixing_ratio_from_specific_humidity": ((q,), {}),
        "vapour_pressure_from_specific_humidity": ((q, p), {}),
        "vapour_pressure_from_mixing_ratio": ((w, p), {}),
        "specific_humidity_from_vapour_pressure": ((e, p), {}),
        "mixing_ratio_from_vapour_pressure": ((e, p), {}),
        "saturation_vapour_pressure": ((t,), {}),
        "saturation_mixing_ratio": ((t, p), {}),
        "saturation_specific_humidity": ((t, p), {}),
        "saturation_vapour_pressure_slope": ((t,), {}),
        "saturation_mixing_ratio_slope": ((t, p), {}),
        "saturation_specific_humidity_slope": ((t, p), {}),
        "temperature_from_saturation_vapour_pressure": ((es,), {}),
        "relative_humidity_from_dewpoint": ((t, td), {}),
        "relative_humidity_from_specific_humidity": ((t, q, p), {}),
        "specific_humidity_from_dewpoint": ((td, p), {}),
        "mixing_ratio_from_dewpoint": ((td, p), {}),
        "specific_humidity_from_relative_humidity": ((t, r, p), {}),
        "dewpoint_from_relative_humidity": ((t, r), {}),
        "dewpoint_from_specific_humidity": ((q, p), {}),
        "virtual_temperature": ((t, q), {}),
        "virtual_potential_temperature": ((t, q, p), {}),
        "potential_temperature": ((t, p), {}),
        "temperature_from_potential_temperature": ((th, p), {}),
        "pressure_on_dry_adiabat": ((t, t, p), {}),
        "temperature_on_dry_adiabat": ((p, t, p), {}),
        "lcl_temperature": ((t, td), {}),
        "lcl": ((t, td, p), {}),
        "ept_from_dewpoint": ((t, td, p), {}),
        "ept_from_specific_humidity": ((t, q, p), {}),
        "saturation_ept": ((t, p), {}),
        "temperature_on_moist_adiabat": ((ept, p), {}),
        "wet_bulb_temperature_from_dewpoint": ((t, td, p), {}),
        "wet_bulb_temperature_from_specific_humidity": ((t, q, p), {}),
        "wet_bulb_potential_temperature_from_dewpoint": ((t, td, p), {}),
        "wet_bulb_potential_temperature_from_specific_humidity": ((t, q, p), {}),
        "specific_gas_constant": ((q,), {}),
    }


def _case_array(xp, device):
    import earthkit.meteo.thermo.array as impl

    data = _load_input()
    celsius = xp.asarray([-10.0, 23.6], device=device)
    kelvin = xp.asarray([263.15, 296.75], device=device)
    t = xp.asarray(data["t"], device=device)
    td = xp.asarray(data["td"], device=device)
    r = xp.asarray(data["r"] / 100.0, device=device)
    q = xp.asarray(data["q"], device=device)
    p = xp.asarray(data["p"], device=device)

    return {"impl": impl, "ops": _ops(impl, celsius, kelvin, t, td, r, q, p)}


def _case_xarray():
    import earthkit.meteo.thermo.xarray as impl

    data = _load_input()
    celsius = _da([-10.0, 23.6])
    kelvin = _da([263.15, 296.75])
    t = _da(data["t"])
    td = _da(data["td"])
    r = _da(data["r"] / 100.0)
    q = _da(data["q"])
    p = _da(data["p"])

    return {"impl": impl, "ops": _ops(impl, celsius, kelvin, t, td, r, q, p)}


BACKEND_PARAMS = [
    pytest.param(("array", xp, device), id=f"array-{xp._earthkit_array_namespace_name}-{device}")
    for xp, device in NAMESPACE_DEVICES
]
if not NO_XARRAY:
    BACKEND_PARAMS.append(pytest.param(("xarray", None, None), id="xarray"))


@pytest.fixture(params=BACKEND_PARAMS)
def backend_case(request):
    kind, xp, device = request.param
    if kind == "xarray":
        return _case_xarray()
    return _case_array(xp, device)


@pytest.mark.parametrize(
    "op_name",
    [
        "specific_humidity_from_mixing_ratio",
        "mixing_ratio_from_specific_humidity",
        "vapour_pressure_from_specific_humidity",
        "vapour_pressure_from_mixing_ratio",
        "specific_humidity_from_vapour_pressure",
        "mixing_ratio_from_vapour_pressure",
        "saturation_vapour_pressure",
        "saturation_mixing_ratio",
        "saturation_specific_humidity",
        "saturation_vapour_pressure_slope",
        "saturation_mixing_ratio_slope",
        "saturation_specific_humidity_slope",
        "temperature_from_saturation_vapour_pressure",
        "relative_humidity_from_dewpoint",
        "relative_humidity_from_specific_humidity",
        "specific_humidity_from_dewpoint",
        "mixing_ratio_from_dewpoint",
        "specific_humidity_from_relative_humidity",
        "dewpoint_from_relative_humidity",
        "dewpoint_from_specific_humidity",
        "virtual_temperature",
        "virtual_potential_temperature",
        "potential_temperature",
        "temperature_from_potential_temperature",
        "pressure_on_dry_adiabat",
        "temperature_on_dry_adiabat",
        "lcl_temperature",
        "lcl",
        "ept_from_dewpoint",
        "ept_from_specific_humidity",
        "saturation_ept",
        "temperature_on_moist_adiabat",
        "wet_bulb_temperature_from_dewpoint",
        "wet_bulb_temperature_from_specific_humidity",
        "wet_bulb_potential_temperature_from_dewpoint",
        "wet_bulb_potential_temperature_from_specific_humidity",
        "specific_gas_constant",
    ],
)
def test_highlevel_compatible_with_backend_api(backend_case, op_name):
    impl = backend_case["impl"]
    args, kwargs = backend_case["ops"][op_name]

    got = getattr(thermo, op_name)(*args, **kwargs)
    ref = getattr(impl, op_name)(*args, **kwargs)

    assert _signature(got) == _signature(ref)
