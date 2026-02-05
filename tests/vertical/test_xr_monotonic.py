# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest

from earthkit.meteo.utils.testing import NO_XARRAY

# These tests reuse the same test cases as in test_array_monotonic.py

# The type of the input data per level is encoded in the test name as three letters with:
#   s: scalar
#   a: array
#
# So, e.g. "s_a_s" means the following input data on a level:
#  - data is scalar
#  - coord is array
#  - target_coord is scalar


def _make_xr(xp, data, coord):
    """Make xarray Dataset from data and coord arrays"""
    import xarray as xr

    data_is_scalar = xp.ndim(data[0]) == 0
    coord_is_scalar = xp.ndim(coord[0]) == 0

    if data_is_scalar and coord_is_scalar:
        data_da = xr.DataArray(
            data,
            dims=("z",),
            coords={"z": coord},
        )

        ds = xr.Dataset({"data": data_da})

    elif not data_is_scalar and coord_is_scalar:
        x_num = data.shape[1]
        level_num = data.shape[0]

        x_dim = xp.arange(x_num)
        level_dim = xp.arange(level_num)

        data_da = xr.DataArray(
            data,
            dims=("z", "x"),
            coords={"z": level_dim, "x": x_dim},
        )

        ds = xr.Dataset({"data": data_da})

    elif not data_is_scalar and not coord_is_scalar:
        assert data.shape == coord.shape

        x_num = data.shape[1]
        level_num = data.shape[0]

        x_dim = xp.arange(x_num)
        level_dim = xp.arange(level_num)

        data_da = xr.DataArray(
            data,
            dims=("z", "x"),
            coords={"z": level_dim, "x": x_dim},
        )

        level_da = xr.DataArray(coord, dims=("z", "x"), coords={"z": level_dim, "x": x_dim})

        ds = xr.Dataset({"data": data_da, "level_coord": level_da})

    return ds


def _get_data():
    import os
    import sys

    here = os.path.dirname(__file__)
    sys.path.insert(0, here)

    from _monotonic_cases import cases

    return cases


DATA = _get_data()


def make_input(conf_id):
    if NO_XARRAY:
        return None

    for d in DATA[conf_id]:
        data, coord, target_coord, mode, expected_data = d

        yield *_make_input_xr(data, coord, target_coord, expected_data), mode


def _make_input_xr(data, coord, target_coord, expected_data, xp=np, device="cpu"):
    data = xp.asarray(data, device=device)
    coord = xp.asarray(coord, device=device)
    target_coord = xp.asarray(target_coord, device=device)
    expected_data = xp.asarray(expected_data, device=device)

    ds_input = _make_xr(xp, data, coord)
    ds_expected = _make_xr(xp, expected_data, target_coord)

    return ds_input, ds_expected


@pytest.mark.skipif(NO_XARRAY, reason="Xarray tests disabled")
@pytest.mark.parametrize("ds_input,ds_expected,mode", make_input("pressure_s_s_s"))
def test_xr_interpolate_monotonic_s_s_s(ds_input, ds_expected, mode):
    from earthkit.meteo.vertical.interpolation import interpolate_monotonic

    observed = interpolate_monotonic(ds_input.data, ds_input.z, ds_expected.z, mode)
    assert observed == ds_expected


@pytest.mark.skipif(NO_XARRAY, reason="Xarray tests disabled")
@pytest.mark.parametrize("ds_input,ds_expected,mode", make_input("pressure_a_a_s"))
def test_xr_interpolate_monotonic_a_a_s(ds_input, ds_expected, mode):
    from earthkit.meteo.vertical.interpolation import interpolate_monotonic

    observed = interpolate_monotonic(ds_input.data, ds_input.z, ds_expected.z, mode)
    assert observed == ds_expected


@pytest.mark.skipif(NO_XARRAY, reason="Xarray tests disabled")
@pytest.mark.parametrize("ds_input,ds_expected,mode", make_input("pressure_a_s_s"))
def test_xr_interpolate_monotonic_a_s_s(ds_input, ds_expected, mode):
    from earthkit.meteo.vertical.interpolation import interpolate_monotonic

    observed = interpolate_monotonic(ds_input.data, ds_input.z, ds_expected.z, mode)
    assert observed == ds_expected


@pytest.mark.skipif(NO_XARRAY, reason="Xarray tests disabled")
@pytest.mark.parametrize("ds_input,ds_expected,mode", make_input("pressure_a_s_a"))
def test_xr_interpolate_monotonic_a_s_a(ds_input, ds_expected, mode):
    from earthkit.meteo.vertical.interpolation import interpolate_monotonic

    observed = interpolate_monotonic(ds_input.data, ds_input.z, ds_expected.z, mode)
    assert observed == ds_expected


@pytest.mark.skipif(NO_XARRAY, reason="Xarray tests disabled")
@pytest.mark.parametrize("ds_input,ds_expected,mode", make_input("pressure_a_a_a"))
def test_xr_interpolate_monotonic_a_a_a(ds_input, ds_expected, mode):
    from earthkit.meteo.vertical.interpolation import interpolate_monotonic

    observed = interpolate_monotonic(ds_input.data, ds_input.z, ds_expected.z, mode)
    assert observed == ds_expected
