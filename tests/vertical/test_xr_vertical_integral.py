# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest

from earthkit.meteo import constants
from earthkit.meteo.utils.testing import NO_XARRAY

pytestmark = pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")

# Pressure (Pa) on the 3 data levels and on the 4 layer interfaces.
P = np.array([10000.0, 20000.0, 40000.0])
P_INTF = np.array([5000.0, 15000.0, 30000.0, 70000.0])

# Geometric height (m) on the 3 data levels and on the 4 layer interfaces.
H = np.array([0.0, 1000.0, 2000.0])
H_INTF = np.array([100.0, 600.0, 1500.0, 3000.0])

# Air density (kg m-3) on the 3 data levels.
RHO = np.array([1.2, 0.9, 0.6])

# Data with three levels and two columns. With the trapezoidal weights
# w = [5000, 15000, 10000] Pa the first column is (1 * 5000 + 2 * 15000 + 3 * 10000) / g.
F = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
EXPECTED_P = np.array([65000.0, 130000.0]) / constants.g

# Three-dimensional data: (x, y, z).
F3 = np.arange(1.0, 19.0).reshape(2, 3, 3)

# Two columns labelled "A" and "B" with different vertical coordinates, to check that the
# coordinate arguments of the integrals are matched with the data by the labels of their
# horizontal dimensions. The rows below are always in the order (A, B).
FX = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (x, z)
PX = np.array([[10000.0, 20000.0, 40000.0], [20000.0, 40000.0, 80000.0]])  # (x, z)
PX_INTF = np.array([[5000.0, 15000.0, 30000.0, 70000.0], [10000.0, 30000.0, 60000.0, 140000.0]])  # (x, level_interface)
HX = np.array([[0.0, 1000.0, 2000.0], [0.0, 100.0, 200.0]])  # (x, z)
RHOX = np.array([[1.2, 0.9, 0.6], [0.4, 0.3, 0.2]])  # (x, z)

INTF = "level_interface"


def _array():
    import earthkit.meteo.vertical.array as vertical_array

    return vertical_array


def _data(
    values=F,
    dims=("z", "cell"),
    coord=None,
    coord_dim="z",
    coord_units=None,
    attrs=None,
    name="q",
):
    """Create a DataArray, optionally with a coordinate along the vertical dimension."""
    import xarray as xr

    data = xr.DataArray(
        np.asarray(values, dtype="float64"),
        dims=dims,
        name=name,
        attrs=dict(attrs or {}),
    )
    if coord is not None:
        data = data.assign_coords(**{coord_dim: (coord_dim, np.asarray(coord, dtype="float64"))})
        if coord_units is not None:
            data = data.assign_coords(**{coord_dim: data[coord_dim].assign_attrs(units=coord_units)})
    return data


def _coord(values, dim, units=None, coord=None):
    """Create a 1D vertical coordinate DataArray defined on the ``dim`` dimension.

    ``units`` is the value of the "units" attribute and ``coord`` the values of the
    coordinate defined along ``dim``.
    """
    return _data(
        values=values, dims=(dim,), name=None, coord=coord, coord_dim=dim, attrs={"units": units} if units else None
    )


def _labelled(values, labels, dims=("x", "z")):
    """Create a DataArray whose first dimension ``dims[0]`` is indexed by ``labels``."""
    import xarray as xr

    return xr.DataArray(np.asarray(values, dtype="float64"), dims=dims, coords={dims[0]: list(labels)}, name=None)


# -----------------------------------------------------------------------------
# pressure
# -----------------------------------------------------------------------------


def test_pressure_levels():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    res = vertical_integral_pressure(_data(), _coord(P, "z"))

    assert res.dims == ("cell",)
    np.testing.assert_allclose(res, EXPECTED_P, rtol=1e-12)
    np.testing.assert_allclose(res, _array().vertical_integral_pressure(F, P), rtol=1e-12)


def test_pressure_levels_descending():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    # the direction of the coordinate is irrelevant
    res = vertical_integral_pressure(_data(values=F[::-1]), _coord(P[::-1], "z"))

    np.testing.assert_allclose(res, EXPECTED_P, rtol=1e-12)


def test_pressure_from_data_coord():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    res = vertical_integral_pressure(_data(coord=P))

    np.testing.assert_allclose(res, EXPECTED_P, rtol=1e-12)


def test_pressure_from_data_coord_hpa():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    # the "hPa" units of the vertical coordinate are converted to Pa
    res = vertical_integral_pressure(_data(coord=P / 100.0, coord_units="hPa"))

    np.testing.assert_allclose(res, EXPECTED_P, rtol=1e-12)


def test_pressure_from_data_coord_pa():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    res = vertical_integral_pressure(_data(coord=P, coord_units="Pa"))

    np.testing.assert_allclose(res, EXPECTED_P, rtol=1e-12)


def test_pressure_units_of_p_are_converted():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    # "p" is in hPa and its own "z" coordinate (in Pa) does not match the one of the
    # data (in hPa): the coordinates are not used, only the converted values
    data = _data(coord=P / 100.0, coord_units="hPa")
    p = _coord(P / 100.0, "z", units="hPa", coord=P)

    res = vertical_integral_pressure(data, p)

    np.testing.assert_allclose(res, EXPECTED_P, rtol=1e-12)


def test_pressure_interfaces():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    res = vertical_integral_pressure(_data(), _coord(P_INTF, INTF))

    assert res.dims == ("cell",)
    assert INTF not in res.dims and INTF not in res.coords
    np.testing.assert_allclose(res, _array().vertical_integral_pressure(F, P_INTF), rtol=1e-12)


def test_pressure_interfaces_own_coordinate():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    # the interface dimension has its own coordinate, which is dropped from the result
    p = _coord(P_INTF, INTF, coord=P_INTF)

    res = vertical_integral_pressure(_data(), p)

    np.testing.assert_allclose(res, _array().vertical_integral_pressure(F, P_INTF), rtol=1e-12)


def test_pressure_one_level_with_two_interfaces():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    # a single level needs the layer interfaces
    res = vertical_integral_pressure(_data(values=F[:1]), _coord(P_INTF[:2], INTF))

    np.testing.assert_allclose(res, _array().vertical_integral_pressure(F[:1], P_INTF[:2]), rtol=1e-12)


# -----------------------------------------------------------------------------
# broadcasting and position of the vertical dimension
# -----------------------------------------------------------------------------


def test_pressure_1d_coordinate_with_3d_data():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    data = _data(values=F3, dims=("x", "y", "z"))

    res = vertical_integral_pressure(data, _coord(P, "z"))

    assert res.dims == ("x", "y")
    np.testing.assert_allclose(res, _array().vertical_integral_pressure(F3, P, vertical_dim=2), rtol=1e-10)


def test_pressure_partial_rank_coordinate():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    # "p" is defined on (y, z): it is broadcast over the "x" dimension of the data
    p = _data(values=np.tile(P, (3, 1)), dims=("y", "z"), name=None)
    data = _data(values=F3, dims=("x", "y", "z"))

    res = vertical_integral_pressure(data, p)

    assert res.dims == ("x", "y")
    np.testing.assert_allclose(
        res, _array().vertical_integral_pressure(F3, p.values[np.newaxis], vertical_dim=2), rtol=1e-10
    )


def test_pressure_partial_rank_interfaces():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    # the interfaces are defined on (y, level_interface)
    p = _data(values=np.tile(P_INTF, (3, 1)), dims=("y", INTF), name=None)
    data = _data(values=F3, dims=("x", "y", "z"))

    res = vertical_integral_pressure(data, p)

    assert res.dims == ("x", "y")
    np.testing.assert_allclose(
        res, _array().vertical_integral_pressure(F3, p.values[np.newaxis], vertical_dim=2), rtol=1e-10
    )


@pytest.mark.parametrize(
    "dims,axis",
    [(("level", "x", "y"), 0), (("x", "level", "y"), 1), (("x", "y", "level"), 2)],
)
def test_pressure_vertical_dim(dims, axis):
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    # the same data with the vertical dimension first, in the middle and last
    values = np.moveaxis(F3, 2, axis)
    data = _data(values=values, dims=dims)

    res = vertical_integral_pressure(data, _coord(P, "level"), vertical_dim="level")

    assert res.dims == tuple(d for d in dims if d != "level")
    np.testing.assert_allclose(
        res,
        _array().vertical_integral_pressure(np.ascontiguousarray(values), P, vertical_dim=axis),
        rtol=1e-10,
    )


def test_pressure_1d_data():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    data = _data(values=P * 2.0, dims=("z",))

    res = vertical_integral_pressure(data, _coord(P, "z"))

    assert res.dims == ()
    np.testing.assert_allclose(res, _array().vertical_integral_pressure(P * 2.0, P), rtol=1e-10)


# -----------------------------------------------------------------------------
# matching the coordinate arguments with the data by horizontal labels
# -----------------------------------------------------------------------------


def test_pressure_labels_reversed():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    # "p" lists the labels of the shared "x" dimension in the reverse order of the data:
    # its values are matched with the data by label, not by position
    data = _labelled(FX, ["A", "B"]).rename("q")
    p = _labelled(PX[::-1], ["B", "A"])

    res = vertical_integral_pressure(data, p)

    expected = np.array([_array().vertical_integral_pressure(FX[i], PX[i]) for i in range(2)])
    np.testing.assert_allclose(res, expected, rtol=1e-12)
    # the result keeps the labels and the order of the data
    assert list(res.coords["x"].values) == ["A", "B"]


def test_pressure_interfaces_labels_reversed():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    # the interfaces are defined on (x, level_interface) with the labels in reverse order
    data = _labelled(FX, ["A", "B"]).rename("q")
    p = _labelled(PX_INTF[::-1], ["B", "A"], dims=("x", INTF))

    res = vertical_integral_pressure(data, p)

    expected = np.array([_array().vertical_integral_pressure(FX[i], PX_INTF[i]) for i in range(2)])
    np.testing.assert_allclose(res, expected, rtol=1e-12)
    assert list(res.coords["x"].values) == ["A", "B"]
    assert INTF not in res.dims and INTF not in res.coords


def test_height_labels_reversed():
    from earthkit.meteo.vertical.xarray import vertical_integral_height

    data = _labelled(FX, ["A", "B"]).rename("q")
    h = _labelled(HX[::-1], ["B", "A"])

    res = vertical_integral_height(data, h)

    expected = np.array([_array().vertical_integral_height(FX[i], HX[i]) for i in range(2)])
    np.testing.assert_allclose(res, expected, rtol=1e-12)
    assert list(res.coords["x"].values) == ["A", "B"]


def test_height_rho_labels_reversed():
    from earthkit.meteo.vertical.xarray import vertical_integral_height

    # "h" is common to both columns while "rho" is labelled in the reverse order
    data = _labelled(FX, ["A", "B"]).rename("q")
    rho = _labelled(RHOX[::-1], ["B", "A"])

    res = vertical_integral_height(data, _coord(H, "z"), rho=rho)

    expected = np.array([_array().vertical_integral_height(FX[i], H, rho=RHOX[i]) for i in range(2)])
    np.testing.assert_allclose(res, expected, rtol=1e-12)
    assert list(res.coords["x"].values) == ["A", "B"]


def test_coordinate_without_horizontal_index_matched_by_position():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    # "p" has no coordinate along "x": its values keep being matched with the data by
    # position, while the result keeps the coordinate of the data
    data = _labelled(FX, ["A", "B"]).rename("q")
    p = _labelled(PX[::-1], ["B", "A"]).drop_vars("x")

    res = vertical_integral_pressure(data, p)

    expected = np.array([_array().vertical_integral_pressure(FX[i], PX[::-1][i]) for i in range(2)])
    np.testing.assert_allclose(res, expected, rtol=1e-12)
    assert list(res.coords["x"].values) == ["A", "B"]


def test_mismatched_horizontal_labels():
    from earthkit.meteo.vertical.xarray import vertical_integral_height, vertical_integral_pressure

    data = _labelled(FX, ["A", "B"]).rename("q")
    match = 'different coordinate labels along the dimension "x"'

    with pytest.raises(ValueError, match=match):
        vertical_integral_pressure(data, _labelled(PX, ["A", "C"]))

    with pytest.raises(ValueError, match=match):
        vertical_integral_pressure(data, _labelled(PX_INTF, ["A", "C"], dims=("x", INTF)))

    with pytest.raises(ValueError, match=match):
        vertical_integral_height(data, _labelled(HX, ["B", "C"]))

    with pytest.raises(ValueError, match=match):
        vertical_integral_height(data, _coord(H, "z"), rho=_labelled(RHOX, ["C", "A"]))


# -----------------------------------------------------------------------------
# height
# -----------------------------------------------------------------------------


def test_height_levels():
    from earthkit.meteo.vertical.xarray import vertical_integral_height

    res = vertical_integral_height(_data(), _coord(H, "z"))

    assert res.dims == ("cell",)
    np.testing.assert_allclose(res, _array().vertical_integral_height(F, H), rtol=1e-12)


def test_height_from_data_coord():
    from earthkit.meteo.vertical.xarray import vertical_integral_height

    res = vertical_integral_height(_data(coord=H))

    np.testing.assert_allclose(res, _array().vertical_integral_height(F, H), rtol=1e-12)


def test_height_from_data_coord_km():
    from earthkit.meteo.vertical.xarray import vertical_integral_height

    res = vertical_integral_height(_data(coord=H / 1000.0, coord_units="km"))

    np.testing.assert_allclose(res, _array().vertical_integral_height(F, H), rtol=1e-10)


def test_height_interfaces():
    from earthkit.meteo.vertical.xarray import vertical_integral_height

    res = vertical_integral_height(_data(), _coord(H_INTF, INTF))

    np.testing.assert_allclose(res, _array().vertical_integral_height(F, H_INTF), rtol=1e-12)


def test_height_rho():
    from earthkit.meteo.vertical.xarray import vertical_integral_height

    res = vertical_integral_height(_data(), _coord(H, "z"), rho=_coord(RHO, "z"))

    assert res.dims == ("cell",)
    np.testing.assert_allclose(res, _array().vertical_integral_height(F, H, rho=RHO), rtol=1e-12)


def test_height_rho_partial_rank():
    from earthkit.meteo.vertical.xarray import vertical_integral_height

    # "rho" is defined on (y, z) and is broadcast over the "x" dimension of the data
    rho = _data(values=np.tile(RHO, (3, 1)), dims=("y", "z"), name=None)
    data = _data(values=F3, dims=("x", "y", "z"))

    res = vertical_integral_height(data, _coord(H, "z"), rho=rho)

    assert res.dims == ("x", "y")
    expected = _array().vertical_integral_height(F3, H, rho=rho.values[np.newaxis], vertical_dim=2)
    np.testing.assert_allclose(res, expected, rtol=1e-10)


def test_height_rho_data_shaped():
    from earthkit.meteo.vertical.xarray import vertical_integral_height

    # a different density profile in each column
    rho = _data(values=np.column_stack([RHO, RHO[::-1]]), dims=("z", "cell"))

    res = vertical_integral_height(_data(), _coord(H, "z"), rho=rho)

    assert res.dims == ("cell",)
    np.testing.assert_allclose(res, _array().vertical_integral_height(F, H, rho=np.asarray(rho)), rtol=1e-12)


def test_height_interfaces_with_rho():
    from earthkit.meteo.vertical.xarray import vertical_integral_height

    res = vertical_integral_height(_data(), _coord(H_INTF, INTF), rho=_coord(RHO, "z"))

    np.testing.assert_allclose(res, _array().vertical_integral_height(F, H_INTF, rho=RHO), rtol=1e-12)


# -----------------------------------------------------------------------------
# metadata of the result
# -----------------------------------------------------------------------------


def test_metadata_pressure():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    data = _data(
        coord=P,
        name="t",
        attrs={"units": "K", "standard_name": "air_temperature", "long_name": "temperature", "foo": "bar"},
    ).assign_coords(cell=("cell", [1, 2]), lat=("cell", [10.0, 20.0]))

    res = vertical_integral_pressure(data)

    assert res.name == "t"
    assert res.dims == ("cell",)
    assert "z" not in res.dims and "z" not in res.coords
    assert set(res.coords) == {"cell", "lat"}
    assert res.attrs == {
        "long_name": "vertical integral of temperature",
        "units": "K kg m-2",
        "foo": "bar",
    }
    # the attributes of the input are not modified
    assert data.attrs["standard_name"] == "air_temperature"
    assert data.attrs["long_name"] == "temperature"


def test_metadata_long_name_falls_back_to_the_name():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    res = vertical_integral_pressure(_data(coord=P, name="q", attrs={"units": "kg kg-1"}))

    assert res.attrs["long_name"] == "vertical integral of q"
    assert res.attrs["units"] == "kg kg-1 kg m-2"
    assert "standard_name" not in res.attrs


def test_metadata_without_units():
    from earthkit.meteo.vertical.xarray import vertical_integral_height, vertical_integral_pressure

    assert vertical_integral_pressure(_data(coord=P, name=None)).attrs == {}
    assert vertical_integral_height(_data(coord=H, name=None)).attrs == {}


def test_metadata_height_units():
    from earthkit.meteo.vertical.xarray import vertical_integral_height

    data = _data(coord=H, attrs={"units": "K"})

    assert vertical_integral_height(data).attrs["units"] == "K m"
    assert vertical_integral_height(data, rho=_coord(RHO, "z")).attrs["units"] == "K kg m-2"


# -----------------------------------------------------------------------------
# dask
# -----------------------------------------------------------------------------


def _dask():
    return pytest.importorskip("dask.array")


def test_dask_levels():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    da = _dask()

    # several chunks along the vertical dimension
    res = vertical_integral_pressure(_data(coord=P).chunk({"z": 1}), _coord(P, "z"))

    assert isinstance(res.data, da.Array)
    assert sum(res.chunks[0]) == 2
    np.testing.assert_allclose(res, EXPECTED_P, rtol=1e-12)


def test_dask_interfaces():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    da = _dask()

    res = vertical_integral_pressure(_data().chunk({"z": 2, "cell": 1}), _coord(P_INTF, INTF))

    assert isinstance(res.data, da.Array)
    np.testing.assert_allclose(res, _array().vertical_integral_pressure(F, P_INTF), rtol=1e-12)


def test_dask_chunked_coordinate():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    da = _dask()

    # the coordinate is also chunked along the vertical dimension
    data = _data().chunk({"z": 1, "cell": 1})
    p = _data(values=np.tile(P_INTF[:, None], (1, 2)), dims=(INTF, "cell")).chunk({INTF: 2, "cell": 1})

    res = vertical_integral_pressure(data, p)

    assert isinstance(res.data, da.Array)
    np.testing.assert_allclose(res, _array().vertical_integral_pressure(F, np.asarray(p)), rtol=1e-12)


def test_dask_height_with_rho():
    from earthkit.meteo.vertical.xarray import vertical_integral_height

    da = _dask()

    res = vertical_integral_height(_data().chunk({"z": 1}), _coord(H, "z"), rho=_coord(RHO, "z"))

    assert isinstance(res.data, da.Array)
    np.testing.assert_allclose(res, _array().vertical_integral_height(F, H, rho=RHO), rtol=1e-12)


def test_dask_labels_reversed():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    da = _dask()

    # the coordinate is chunked and its horizontal labels are in the reverse order of the data
    data = _labelled(FX, ["A", "B"]).rename("q").chunk({"z": 1})
    p = _labelled(PX[::-1], ["B", "A"]).chunk({"x": 1})

    res = vertical_integral_pressure(data, p)

    assert isinstance(res.data, da.Array)
    expected = np.array([_array().vertical_integral_pressure(FX[i], PX[i]) for i in range(2)])
    np.testing.assert_allclose(res, expected, rtol=1e-12)
    assert list(res.coords["x"].values) == ["A", "B"]


def test_dask_3d_with_partial_rank_coordinate():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    _dask()

    # the vertical dimension of the data is not chunked, the other ones are
    data = _data(values=F3, dims=("x", "y", "z")).chunk({"x": 1, "y": 2})
    p = _data(values=np.tile(P, (3, 1)), dims=("y", "z"), name=None)

    res = vertical_integral_pressure(data, p)

    np.testing.assert_allclose(res, _array().vertical_integral_pressure(F3, P, vertical_dim=2), rtol=1e-10)


# -----------------------------------------------------------------------------
# errors
# -----------------------------------------------------------------------------


def _error_cases():
    from earthkit.meteo.vertical.xarray import vertical_integral_height, vertical_integral_pressure

    data = _data(coord=P)
    data3 = _data(values=F3, dims=("x", "y", "z"))

    return [
        # unsupported units of the coordinate given in "p"
        ("unsupported units", lambda: vertical_integral_pressure(data, _coord(P, "z", units="psi"))),
        # unsupported units of the vertical coordinate of the data
        ("unsupported units", lambda: vertical_integral_pressure(_data(coord=P, coord_units="bar"))),
        # unsupported units of the coordinate given in "h"
        ("unsupported units", lambda: vertical_integral_height(_data(coord=H), _coord(H, "z", units="cm"))),
        # "p" is on the vertical dimension of the data, with the values of the interfaces
        ('"data" has 3 level', lambda: vertical_integral_pressure(data, _coord(P_INTF, "z"))),
        # the dimension of the interfaces has the wrong number of values
        ("must be 4", lambda: vertical_integral_pressure(data, _coord([1.0, 2.0, 3.0, 4.0, 5.0], INTF))),
        # the dimension of the interfaces is ambiguous
        (
            "exactly one dimension that is not in",
            lambda: vertical_integral_pressure(data, _data(values=np.ones((2, 2)), dims=("u", "v"))),
        ),
        # the interface dimension is mixed with the vertical dimension of the data
        (
            "mixes the vertical dimension",
            lambda: vertical_integral_pressure(data3, _data(values=np.ones((2, 3, 4)), dims=("x", "z", INTF))),
        ),
        # "rho" is not defined on the vertical dimension of the data
        (
            "must be defined on the vertical dimension",
            lambda: vertical_integral_height(data, _coord(H, "z"), rho=_coord(RHO, "lev")),
        ),
        # "rho" has the wrong number of levels
        (
            "defined on the 3 data levels",
            lambda: vertical_integral_height(data, _coord(H, "z"), rho=_coord([1.0, 2.0], "z")),
        ),
        # the vertical dimension is not in the data
        ("is not in the dimensions", lambda: vertical_integral_pressure(data, vertical_dim="level")),
        # the data has no vertical coordinate and "p" is not given
        ("was not specified", lambda: vertical_integral_pressure(_data())),
    ]


@pytest.mark.parametrize("match,call", _error_cases())
def test_errors(match, call):
    with pytest.raises(ValueError, match=match):
        call()


def test_coordinate_must_be_a_dataarray():
    from earthkit.meteo.vertical.xarray import vertical_integral_pressure

    with pytest.raises(TypeError, match="must be an xarray.DataArray"):
        vertical_integral_pressure(_data(coord=P), P)


# -----------------------------------------------------------------------------
# the top-level, dispatching function
# -----------------------------------------------------------------------------


def test_high_level_vertical_integral_pressure():
    import xarray as xr

    import earthkit.meteo.vertical as vertical_high_level
    import earthkit.meteo.vertical.xarray as vertical_xarray

    # the pressure is taken from the coordinate of the data and "vertical_dim" defaults to "z"
    data = _data(coord=P / 100.0, coord_units="hPa", name="t", attrs={"units": "K"})

    res = vertical_high_level.vertical_integral_pressure(data)

    assert isinstance(res, xr.DataArray)
    xr.testing.assert_identical(res, vertical_xarray.vertical_integral_pressure(data))
    np.testing.assert_allclose(res, EXPECTED_P, rtol=1e-12)

    # "p" and "vertical_dim" are forwarded when they are given
    data = _data(values=np.moveaxis(F3, 2, 0), dims=("level", "x", "y"))

    res = vertical_high_level.vertical_integral_pressure(data, _coord(P, "level"), vertical_dim="level")

    assert res.dims == ("x", "y")
    np.testing.assert_allclose(res, _array().vertical_integral_pressure(np.moveaxis(F3, 2, 0), P, vertical_dim=0))


def test_high_level_vertical_integral_height():
    import xarray as xr

    import earthkit.meteo.vertical as vertical_high_level
    import earthkit.meteo.vertical.xarray as vertical_xarray

    data = _data(coord=H, coord_units="m", name="t", attrs={"units": "K"})
    rho = _coord(RHO, "z")

    res = vertical_high_level.vertical_integral_height(data, rho=rho)

    assert isinstance(res, xr.DataArray)
    xr.testing.assert_identical(res, vertical_xarray.vertical_integral_height(data, rho=rho))
    np.testing.assert_allclose(res, _array().vertical_integral_height(F, H, rho=RHO), rtol=1e-12)

    # "h" is forwarded when it is given
    res = vertical_high_level.vertical_integral_height(_data(), _coord(H, "z"))

    xr.testing.assert_identical(res, vertical_xarray.vertical_integral_height(_data(), _coord(H, "z")))
