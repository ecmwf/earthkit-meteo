# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Tests that the high level radiation API dispatches to the backend implementations."""

import numpy as np
import pytest
from earthkit.utils.array import array_namespace
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo import radiation
from earthkit.meteo.utils.testing import NO_EKD, NO_XARRAY

DIFFUSE = [0.0, 120.5, 310.0]  # W/m2
DIRECT = [0.0, 45.25, 590.0]  # W/m2
TOTAL = [0.0, 165.75, 900.0]  # W/m2

NET_LONGWAVE = [-60.0, -30.0, -105.5]  # W/m2
SURFACE_TEMPERATURE = [280.0, 300.0, 265.3]  # K

OP_NAMES = ["surface_downward_shortwave_radiation", "surface_downward_longwave_flux"]

# The output parameter.variable is the GRIB shortName from FIELD_PARAMS, not the op name.
OP_VARIABLES = {
    "surface_downward_shortwave_radiation": "avg_sdswrf",
    "surface_downward_longwave_flux": "avg_sdlwrf",
}


def _signature(obj):
    if hasattr(obj, "dims") and hasattr(obj, "shape"):
        return ("xarray", tuple(obj.dims), tuple(obj.shape))

    xp = array_namespace(obj)
    arr = xp.asarray(obj)
    return ("array", tuple(arr.shape))


def _ops(diffuse, direct, net_longwave, surface_temperature):
    return {
        "surface_downward_shortwave_radiation": ((diffuse, direct), {}),
        "surface_downward_longwave_flux": ((net_longwave, surface_temperature), {"emissivity": 0.98}),
    }


def _case_array(xp, device):
    import earthkit.meteo.radiation.array as impl

    def _a(values):
        return xp.asarray(values, device=device)

    return {
        "impl": impl,
        "ops": _ops(_a(DIFFUSE), _a(DIRECT), _a(NET_LONGWAVE), _a(SURFACE_TEMPERATURE)),
    }


def _case_xarray():
    import xarray as xr

    import earthkit.meteo.radiation.xarray as impl

    def _da(values):
        return xr.DataArray(np.asarray(values))

    return {
        "impl": impl,
        "ops": _ops(_da(DIFFUSE), _da(DIRECT), _da(NET_LONGWAVE), _da(SURFACE_TEMPERATURE)),
    }


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


@pytest.mark.parametrize("op_name", OP_NAMES)
def test_highlevel_compatible_with_backend_api(backend_case, op_name):
    args, kwargs = backend_case["ops"][op_name]

    got = getattr(radiation, op_name)(*args, **kwargs)
    ref = getattr(backend_case["impl"], op_name)(*args, **kwargs)

    assert _signature(got) == _signature(ref)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref))


def test_highlevel_shortwave_values(backend_case):
    args, kwargs = backend_case["ops"]["surface_downward_shortwave_radiation"]

    out = radiation.surface_downward_shortwave_radiation(*args, **kwargs)

    np.testing.assert_allclose(np.asarray(out), np.array(TOTAL))


@pytest.mark.skipif(NO_EKD, reason="EKD is not installed")
@pytest.mark.parametrize("op_name", OP_NAMES)
def test_highlevel_dispatches_to_fieldlist(op_name):
    from earthkit.data import Field, FieldList

    import earthkit.meteo.radiation.fieldlist as impl

    def _fieldlist(values, variable, units):
        return FieldList.from_fields([
            Field.from_components(
                values=np.array(values),
                parameter={"variable": variable, "units": units},
                vertical={"level": 0, "level_type": "surface"},
            )
        ])

    if op_name == "surface_downward_shortwave_radiation":
        args = (_fieldlist(DIFFUSE, "ssrd_diffuse", "W/m2"), _fieldlist(DIRECT, "fdir", "W/m2"))
        kwargs = {}
    else:
        args = (_fieldlist(NET_LONGWAVE, "athb_s", "W/m2"), _fieldlist(SURFACE_TEMPERATURE, "t", "K"))
        kwargs = {"emissivity": 0.98}

    out = getattr(radiation, op_name)(*args, **kwargs)
    ref = getattr(impl, op_name)(*args, **kwargs)

    assert len(out) == 1
    assert out.get("parameter.variable") == [OP_VARIABLES[op_name]]
    np.testing.assert_allclose(out[0].values, ref[0].values)
