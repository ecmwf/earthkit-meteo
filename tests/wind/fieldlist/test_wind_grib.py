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

from earthkit.meteo.utils.testing import NO_EKD

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})

import numpy as np
from earthkit.data.core.temporary import temp_file

from earthkit.meteo.utils import testing

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})
pytestmark = pytest.mark.skipif(NO_EKD, reason="EKD is not installed")

LEVELS = [1000, 850, 700, 500, 400, 300]
WIND_GRIB_FILE = "tuv_pl.grib"
OMEGA_GRIB_FILE = "omega_pl.grib"


def _make_input_fieldlist(filename, param, input_type="fieldlist", level_type="pressure"):
    fl = _get_fieldlist(filename)
    if level_type == "pressure":
        param_fl = fl.sel({"parameter.variable": param, "vertical.level_type": level_type}).order_by("level")
    else:
        param_fl = fl.sel({"parameter.variable": param})

    return param_fl if input_type == "fieldlist" else param_fl[0]


def _make_pres_fieldlist(md_fl, pres_type="fl"):
    from earthkit.data import Field, FieldList

    if isinstance(md_fl, FieldList):
        if pres_type == "fl":
            p_fl = []
            for md_field in md_fl:
                p_fl.append(
                    md_field.set(
                        values=md_field.values * 0.0 + md_field.get("vertical.level") * 100.0,
                        **{"parameter.variable": "pres"},
                    ).sync()
                )
            return FieldList.from_fields(p_fl)
        elif pres_type == "value":
            pres = np.array(md_fl.get("vertical.level")) * 100.0  # convert to Pa
            return pres
    elif isinstance(md_fl, Field):
        if pres_type == "fl":
            return md_fl.set(
                values=md_fl.values * 0.0 + md_fl.get("vertical.level") * 100.0,
                **{"parameter.variable": "pres"},
            ).sync()
        elif pres_type == "value":
            pres = md_fl.get("vertical.level") * 100.0  # convert to Pa
            return pres
    else:
        raise ValueError(f"Unsupported md_fl type: {type(md_fl)}")

    if pres_type is None:
        return None

    raise ValueError(f"Unsupported pres_type: {pres_type}")


def _get_fieldlist(name, sample=False):
    import earthkit.data as ekd

    if sample:
        return ekd.from_source("sample", name).to_fieldlist()
    else:
        path = testing.get_test_data(name, "test-data")
        fl = ekd.from_source("file", path).to_fieldlist()
        return fl


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_wind_speed(input_type):
    import earthkit.data as ekd

    import earthkit.meteo.wind.fieldlist as wind

    u = _make_input_fieldlist(WIND_GRIB_FILE, "u", input_type=input_type)
    v = _make_input_fieldlist(WIND_GRIB_FILE, "v", input_type=input_type)
    out = wind.speed(u, v)

    assert isinstance(out, type(u))

    if input_type == "fieldlist":
        assert len(out) == len(u)
        ref_metadata = {
            "parameter.variable": ["ws"] * len(out),
            "vertical.level": LEVELS,
            "vertical.level_type": ["pressure"] * len(out),
        }
        field = out[0]
    elif input_type == "field":
        ref_metadata = {
            "parameter.variable": "ws",
            "vertical.level": 1000,
            "vertical.level_type": "pressure",
        }
        field = out

    for k, v in ref_metadata.items():
        assert out.get(k) == v

    # values
    ref_vals = np.array([[10.0443162200, 10.0443162200], [11.9568410174, 4.1988514807], [7.3634531062, 4.5049595334]])
    assert field.shape == (7, 12)
    np.testing.assert_allclose(field.to_numpy()[:3, :2], ref_vals)

    # GRIB metadata
    field = field.sync()
    assert field.get("metadata.shortName") == "ws"

    # write back to GRIB
    with temp_file() as tmp:
        field.to_target("file", tmp)
        field_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

        ref_metadata = {
            "parameter.variable": "ws",
            "metadata.shortName": "ws",
            "vertical.level": 1000,
            "metadata.levelist": 1000,
            "vertical.level_type": "pressure",
            "metadata.typeOfLevel": "isobaricInhPa",
        }

        for k, v in ref_metadata.items():
            assert field_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_wind_direction(input_type):
    import earthkit.data as ekd

    import earthkit.meteo.wind.fieldlist as wind

    u = _make_input_fieldlist(WIND_GRIB_FILE, "u", input_type=input_type)
    v = _make_input_fieldlist(WIND_GRIB_FILE, "v", input_type=input_type)
    out = wind.direction(u, v)

    assert isinstance(out, type(u))

    if input_type == "fieldlist":
        assert len(out) == len(u)
        ref_metadata = {
            "parameter.variable": ["wdir"] * len(out),
            "vertical.level": LEVELS,
            "vertical.level_type": ["pressure"] * len(out),
        }
        field = out[0]
    elif input_type == "field":
        ref_metadata = {
            "parameter.variable": "wdir",
            "vertical.level": 1000,
            "vertical.level_type": "pressure",
        }
        field = out

    for k, v in ref_metadata.items():
        assert out.get(k) == v

    # values
    ref_vals = np.array([
        [141.2506786965, 141.2506786965],
        [188.2374160532, 204.0790818835],
        [121.3731188739, 337.6493368993],
    ])
    assert field.shape == (7, 12)
    np.testing.assert_allclose(field.to_numpy()[:3, :2], ref_vals)

    # GRIB metadata
    field = field.sync()
    assert field.get("metadata.shortName") == "wdir"

    # write back to GRIB
    with temp_file() as tmp:
        field.to_target("file", tmp)
        field_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

        ref_metadata = {
            "parameter.variable": "wdir",
            "metadata.shortName": "wdir",
            "vertical.level": 1000,
            "metadata.levelist": 1000,
            "vertical.level_type": "pressure",
            "metadata.typeOfLevel": "isobaricInhPa",
        }

        for k, v in ref_metadata.items():
            assert field_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_w_from_omega(input_type, pres_type):
    import earthkit.data as ekd

    import earthkit.meteo.wind.fieldlist as wind

    omega = _make_input_fieldlist(OMEGA_GRIB_FILE, "w", input_type=input_type)
    t = _make_input_fieldlist(OMEGA_GRIB_FILE, "t", input_type=input_type)
    p = _make_pres_fieldlist(omega, pres_type=pres_type)
    out = wind.w_from_omega(omega, t, p=p)

    assert isinstance(out, type(omega))

    if input_type == "fieldlist":
        assert len(out) == len(omega)
        ref_metadata = {
            "parameter.variable": ["wz"] * len(out),
            "vertical.level": [1000, 500],
            "vertical.level_type": ["pressure"] * len(out),
        }
        field = out[0]
    elif input_type == "field":
        ref_metadata = {
            "parameter.variable": "wz",
            "vertical.level": 1000,
            "vertical.level_type": "pressure",
        }
        field = out

    for k, v in ref_metadata.items():
        assert out.get(k) == v

    # values
    ref_vals = np.array([[0.0006601101, 0.0006601101], [-0.0021794176, -0.0002340644], [0.0002355885, 0.0022681146]])
    assert field.shape == (19, 36)
    np.testing.assert_allclose(field.to_numpy()[:3, :2], ref_vals, rtol=1e-5)

    # TODO: the code below only works for GRIB2
    return

    # GRIB metadata
    field = field.sync()
    assert field.get("metadata.shortName") == "wz"

    # write back to GRIB
    with temp_file() as tmp:
        field.to_target("file", tmp)
        field_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

        ref_metadata = {
            "parameter.variable": "wz",
            "metadata.shortName": "wz",
            "vertical.level": 1000,
            "metadata.levelist": 1000,
            "vertical.level_type": "pressure",
            "metadata.typeOfLevel": "isobaricInhPa",
        }

        for k, v in ref_metadata.items():
            assert field_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_coriolis(input_type):
    import earthkit.meteo.wind.fieldlist as wind

    data = _make_input_fieldlist(WIND_GRIB_FILE, "t", input_type=input_type)
    out = wind.coriolis(data)

    assert isinstance(out, type(data))

    if input_type == "fieldlist":
        assert len(out) == len(data)
        ref_metadata = {
            "parameter.variable": ["fc"] * len(out),
            "vertical.level": LEVELS,
            "vertical.level_type": ["pressure"] * len(out),
        }
        field = out[0]
    elif input_type == "field":
        ref_metadata = {
            "parameter.variable": "fc",
            "vertical.level": 1000,
            "vertical.level_type": "pressure",
        }
        field = out

    for k, v in ref_metadata.items():
        assert out.get(k) == v

    # values
    ref_vals = np.array([[0.0001458423, 0.0001458423], [0.0001263031, 0.0001263031], [0.0000729212, 0.0000729212]])
    assert field.shape == (7, 12)
    np.testing.assert_allclose(field.to_numpy()[:3, :2], ref_vals, rtol=1e-5)

    # TODO: coriolis parameter is not yet supported in GRIB, so we cannot test the metadata and
    #  write back to GRIB
