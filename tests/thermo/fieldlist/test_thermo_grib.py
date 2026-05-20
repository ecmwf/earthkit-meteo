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
from earthkit.data.core.temporary import temp_file

from earthkit.meteo.utils import testing
from earthkit.meteo.utils.testing import NO_EKD

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})
pytestmark = pytest.mark.skipif(NO_EKD, reason="EKD is not installed")

THERMO_PL_FILE = "thermo_850_pl.grib1"
THERMO_2M_FILE = "thermo_2m.grib1"


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


def _get_fieldlist(name):
    import earthkit.data as ekd

    path = testing.get_test_data(name, "test-data")
    fl = ekd.from_source("file", path).to_fieldlist()
    return fl


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_grib_mixing_ratio_from_specific_humidity_pl(input_type):
    from earthkit.meteo.thermo.fieldlist import mixing_ratio_from_specific_humidity

    q = _make_input_fieldlist(THERMO_PL_FILE, "q", input_type=input_type)
    out = mixing_ratio_from_specific_humidity(q)

    assert isinstance(out, type(q))

    if input_type == "fieldlist":
        assert len(out) == len(q)
        assert (np.array(out.get("parameter.variable")) == "w").all()

    if input_type == "field":
        f = out
    else:
        f = out[0]

    # field metadata
    ref_metadata = {
        "parameter.variable": "w",
        # "metadata.shortName": "w",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([[0.0041623926, 0.0056681556], [0.0115226539, 0.0095335452], [0.0009134935, 0.0020191762]])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    # TODO: mixing ratio is not yet supported in GRIB, so we cannot test the metadata and write back to GRIB


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_vapour_pressure_from_specific_humidity_pl(input_type, pres_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import vapour_pressure_from_specific_humidity

    q = _make_input_fieldlist(THERMO_PL_FILE, "q", input_type=input_type)
    p = _make_pres_fieldlist(q, pres_type=pres_type)
    out = vapour_pressure_from_specific_humidity(q, p)

    assert isinstance(out, type(q))
    if input_type == "fieldlist":
        assert len(out) == len(q)
        assert (np.array(out.get("parameter.variable")) == "vapp").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "vapp",
        # "metadata.shortName": "pt",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [565.0516711398, 767.6155087265],
        [1546.0456668635, 1283.1871325047],
        [124.6550566566, 275.0479580490],
    ])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    # TODO: the code below only works for GRIB 2
    return

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "vapp"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "vapp",
                "metadata.shortName": "vapp",
                "vertical.level": 850,
                "metadata.levelist": 850,
                "vertical.level_type": "pressure",
                "metadata.typeOfLevel": "isobaricInhPa",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_grib_saturation_vapour_pressure_pl(input_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import saturation_vapour_pressure

    t = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    out = saturation_vapour_pressure(t)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "swvp").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "swvp",
        # "metadata.shortName": "w",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [1776.6368916801, 1776.6368916801],
        [1226.0465217611, 1226.0465217611],
        [831.7174825476, 831.7174825476],
    ])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    # TODO: the code below only works for GRIB 2
    return

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "swvp"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "swvp",
                "metadata.shortName": "swvp",
                "vertical.level": 850,
                "metadata.levelist": 850,
                "vertical.level_type": "pressure",
                "metadata.typeOfLevel": "isobaricInhPa",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_saturation_mixing_ratio_pl(input_type, pres_type):
    from earthkit.meteo.thermo.fieldlist import saturation_mixing_ratio

    t = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = saturation_mixing_ratio(t, p)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "ws").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "ws",
        # "metadata.shortName": "ws",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([[0.0132779348, 0.0132779348], [0.0091028012, 0.0091028012], [0.0061461688, 0.0061461688]])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    # TODO: saturation mixing ratio is not yet supported in GRIB, so we cannot test the metadata and write back to GRIB


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_saturation_specific_humidity_pl(input_type, pres_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import saturation_specific_humidity

    t = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = saturation_specific_humidity(t, p)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "sqw").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "sqw",
        # "metadata.shortName": "sqw",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([[0.0131039415, 0.0131039415], [0.0090206876, 0.0090206876], [0.0061086242, 0.0061086242]])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    # TODO: the code below only works for GRIB 2
    return

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "sqw"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "sqw",
                "metadata.shortName": "sqw",
                "vertical.level": 850,
                "metadata.levelist": 850,
                "vertical.level_type": "pressure",
                "metadata.typeOfLevel": "isobaricInhPa",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_grib_saturation_vapour_pressure_slope_pl(input_type):
    from earthkit.meteo.thermo.fieldlist import saturation_vapour_pressure_slope

    t = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    out = saturation_vapour_pressure_slope(t)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "es_slope").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "es_slope",
        # "metadata.shortName": "es_slope",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [113.7851239702, 113.7851239702],
        [82.1068429179, 82.1068429179],
        [58.3010504773, 58.3010504773],
    ])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    # TODO: saturation vapour pressure slope is not yet supported in GRIB, so we cannot test the metadata and
    #  write back to GRIB


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_saturation_mixing_ratio_slope_pl(input_type, pres_type):
    from earthkit.meteo.thermo.fieldlist import saturation_mixing_ratio_slope

    t = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = saturation_mixing_ratio_slope(t, p)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "ws_slope").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "ws_slope",
        # "metadata.shortName": "ws_slope",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([[0.0008685424, 0.0008685424], [0.0006185252, 0.0006185252], [0.0004350864, 0.0004350864]])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    # TODO: saturation mixing ratio slope is not yet supported in GRIB, so we cannot test the metadata and
    #  write back to GRIB


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_saturation_specific_humidity_slope_pl(input_type, pres_type):
    from earthkit.meteo.thermo.fieldlist import saturation_specific_humidity_slope

    t = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = saturation_specific_humidity_slope(t, p)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "sqw_slope").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "sqw_slope",
        # "metadata.shortName": "sqw_slope",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([[0.0008459288, 0.0008459288], [0.0006074165, 0.0006074165], [0.0004297871, 0.0004297871]])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    # TODO: saturation mixing ratio slope is not yet supported in GRIB, so we cannot test the metadata and
    #  write back to GRIB


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_relative_humidity_from_specific_humidity_pl(input_type, pres_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import relative_humidity_from_specific_humidity

    t = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    q = _make_input_fieldlist(THERMO_PL_FILE, "q", input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = relative_humidity_from_specific_humidity(t, q, p)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(q)
        assert (np.array(out.get("parameter.variable")) == "r").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "r",
        # "metadata.shortName": "pt",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [31.8045670326, 43.2060998126],
        [126.1000818014, 104.6605581215],
        [14.9876682013, 33.0698781522],
    ])

    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "r"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "r",
                "metadata.shortName": "r",
                "vertical.level": 850,
                "metadata.levelist": 850,
                "vertical.level_type": "pressure",
                "metadata.typeOfLevel": "isobaricInhPa",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_specific_humidity_from_relative_humidity_pl(input_type, pres_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import specific_humidity_from_relative_humidity

    t = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    r = _make_input_fieldlist(THERMO_PL_FILE, "r", input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = specific_humidity_from_relative_humidity(t, r, p)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "q").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "q",
        # "metadata.shortName": "pt",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([[0.0082534342, 0.0082534342], [0.0059874332, 0.0059874332], [0.0042604840, 0.0042604840]])

    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "q"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "q",
                "metadata.shortName": "q",
                "vertical.level": 850,
                "metadata.levelist": 850,
                "vertical.level_type": "pressure",
                "metadata.typeOfLevel": "isobaricInhPa",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_grib_dewpoint_from_relative_humidity_pl(input_type):
    from earthkit.meteo.thermo.fieldlist import dewpoint_from_relative_humidity

    t = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    r = _make_input_fieldlist(THERMO_PL_FILE, "r", input_type=input_type)
    out = dewpoint_from_relative_humidity(t, r)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "td").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "td",
        # "metadata.shortName": td",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [281.8275987199, 281.8275987199],
        [277.1928010072, 277.1928010072],
        [272.4578545089, 272.4578545089],
    ])

    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    # TODO: dewpoint is not yet supported in GRIB, so we cannot test the metadata and
    #  write back to GRIB


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_grib_dewpoint_from_relative_humidity_2m(input_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import dewpoint_from_relative_humidity

    t = _make_input_fieldlist(THERMO_2M_FILE, param="2t", input_type=input_type, level_type=None)
    r = _make_input_fieldlist(THERMO_2M_FILE, param="2r", input_type=input_type, level_type=None)
    out = dewpoint_from_relative_humidity(t, r)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "2d").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "2d",
        # "metadata.shortName": "2d",
        "vertical.level": 0,
        # "metadata.levelist": 2,
        "vertical.level_type": "surface",
        # "metadata.typeOfLevel": "surface",
    }

    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [279.3868804650, 272.7390417302],
        [297.1772547593, 282.9556785267],
        [273.8545975889, 273.5849340251],
    ])

    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "2d"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "2d",
                "metadata.shortName": "2d",
                "vertical.level": 0,
                "metadata.levelist": None,
                "vertical.level_type": "surface",
                "metadata.typeOfLevel": "surface",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_dewpoint_from_specific_humidity_pl(input_type, pres_type):
    from earthkit.meteo.thermo.fieldlist import dewpoint_from_specific_humidity

    q = _make_input_fieldlist(THERMO_PL_FILE, "q", input_type=input_type)
    p = _make_pres_fieldlist(q, pres_type=pres_type)
    out = dewpoint_from_specific_humidity(q, p)

    assert isinstance(out, type(q))
    if input_type == "fieldlist":
        assert len(out) == len(q)
        assert (np.array(out.get("parameter.variable")) == "td").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "td",
        # "metadata.shortName": "td",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [272.0837073468, 276.3384265116],
        [286.6524592617, 283.8231616922],
        [253.0930583531, 262.6458926692],
    ])

    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    # TODO: dewpoint is not yet supported in GRIB, so we cannot test the metadata and
    #  write back to GRIB


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_grib_virtual_temperature_pl(input_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import virtual_temperature

    t = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    q = _make_input_fieldlist(THERMO_PL_FILE, "q", input_type=input_type)
    out = virtual_temperature(t, q)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "vtmp").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "vtmp",
        # "metadata.shortName": "pt",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [289.5325107205, 289.7942320320],
        [285.1013853357, 284.7661871974],
        [277.6312070648, 277.8171252551],
    ])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "vtmp"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "vtmp",
                "metadata.shortName": "vtmp",
                "vertical.level": 850,
                "metadata.levelist": 850,
                "vertical.level_type": "pressure",
                "metadata.typeOfLevel": "isobaricInhPa",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_virtual_potential_temperature_pl(input_type, pres_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import virtual_potential_temperature

    t = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    q = _make_input_fieldlist(THERMO_PL_FILE, "q", input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = virtual_potential_temperature(t, q, p)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "vptmp").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "vptmp",
        # "metadata.shortName": "pt",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [303.2925301324, 303.5666897374],
        [298.6508157151, 298.2996873005],
        [290.8256175614, 291.0203715053],
    ])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    # TODO: the code below only works for GRIB2
    return

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "vptmp"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "vptmp",
                "metadata.shortName": "vptmp",
                "vertical.level": 850,
                "metadata.levelist": 850,
                "vertical.level_type": "pressure",
                "metadata.typeOfLevel": "isobaricInhPa",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_potential_temperature_pl(input_type, pres_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import potential_temperature

    t = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = potential_temperature(t, p)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "pt").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "pt",
        # "metadata.shortName": "pt",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [302.5303728899, 302.5303728899],
        [296.5973818204, 296.5973818204],
        [290.6643907509, 290.6643907509],
    ])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "pt"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "pt",
                "metadata.shortName": "pt",
                "vertical.level": 850,
                "metadata.levelist": 850,
                "vertical.level_type": "pressure",
                "metadata.typeOfLevel": "isobaricInhPa",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_temperature_from_potential_temperature_pl(input_type, pres_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import temperature_from_potential_temperature

    pt = _make_input_fieldlist(THERMO_PL_FILE, "pt", input_type=input_type)
    p = _make_pres_fieldlist(pt, pres_type=pres_type)
    out = temperature_from_potential_temperature(pt, p)

    assert isinstance(out, type(pt))
    if input_type == "fieldlist":
        assert len(out) == len(pt)
        assert (np.array(out.get("parameter.variable")) == "t").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "t",
        # "metadata.shortName": "t",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [288.7955141300, 288.7955141300],
        [283.1318196258, 283.1318196258],
        [277.4683581858, 277.4683581858],
    ])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "t"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "t",
                "metadata.shortName": "t",
                "vertical.level": 850,
                "metadata.levelist": 850,
                "vertical.level_type": "pressure",
                "metadata.typeOfLevel": "isobaricInhPa",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_pressure_on_dry_adiabat_pl(input_type, pres_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import pressure_on_dry_adiabat

    t_def = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    p_def = _make_pres_fieldlist(t_def, pres_type=pres_type)

    if input_type == "field":
        t = t_def.set(values=t_def.values - 10.0).sync()
    else:
        t = []
        for f in t_def:
            t.append(f.set(values=f.values - 10.0).sync())
        t = ekd.FieldList.from_fields(t)

    out = pressure_on_dry_adiabat(
        t,
        t_def,
        p_def,
    )

    assert isinstance(out, type(t_def))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "pres").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "pres",
        # "metadata.shortName": "pres",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [75136.3376750559, 75136.3376750559],
        [74947.8122968675, 74947.8122968675],
        [74751.9497390965, 74751.9497390965],
    ])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "pres"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "pres",
                "metadata.shortName": "pres",
                "vertical.level": 850,
                "metadata.levelist": 850,
                "vertical.level_type": "pressure",
                "metadata.typeOfLevel": "isobaricInhPa",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_temperature_on_dry_adiabat_pl(input_type, pres_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import temperature_on_dry_adiabat

    t_def = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    p_def = _make_pres_fieldlist(t_def, pres_type=pres_type)

    # create target pressure fields for 500 hPa
    if input_type == "field":
        p = t_def.set(values=t_def.values * 0 + 50000.0, **{"parameter.variable": "pres"}).sync()
    else:
        p = []
        for f in t_def:
            p.append(f.set(values=f.values * 0 + 50000.0, **{"parameter.variable": "pres"}).sync())
        p = ekd.FieldList.from_fields(p)

    out = temperature_on_dry_adiabat(
        p,
        t_def,
        p_def,
    )

    assert isinstance(out, type(t_def))
    if input_type == "fieldlist":
        assert len(out) == len(p)
        assert (np.array(out.get("parameter.variable")) == "t").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "t",
        # "metadata.shortName": "t",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [248.1803668525, 248.1803668525],
        [243.3132459546, 243.3132459546],
        [238.4461250567, 238.4461250567],
    ])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "t"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "t",
                "metadata.shortName": "t",
                "vertical.level": 850,
                "metadata.levelist": 850,
                "vertical.level_type": "pressure",
                "metadata.typeOfLevel": "isobaricInhPa",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_ept_from_specific_humidity_pl(input_type, pres_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import ept_from_specific_humidity

    t = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    q = _make_input_fieldlist(THERMO_PL_FILE, "q", input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = ept_from_specific_humidity(t, q, p)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "eqpt").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "eqpt",
        # "metadata.shortName": "eqpt",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [314.3727581626, 318.4389739178],
        [327.3420444094, 322.1902630949],
        [293.3309203494, 296.3306469411],
    ])

    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "eqpt"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "eqpt",
                "metadata.shortName": "eqpt",
                "vertical.level": 850,
                "metadata.levelist": 850,
                "vertical.level_type": "pressure",
                "metadata.typeOfLevel": "isobaricInhPa",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_saturation_ept_pl(input_type, pres_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import saturation_ept

    t = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = saturation_ept(t, p)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "sept").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "sept",
        # "metadata.shortName": "sept",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [338.6982227942, 338.6982227942],
        [321.0733525313, 321.0733525313],
        [307.0349593414, 307.0349593414],
    ])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "sept"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "sept",
                "metadata.shortName": "sept",
                "vertical.level": 850,
                "metadata.levelist": 850,
                "vertical.level_type": "pressure",
                "metadata.typeOfLevel": "isobaricInhPa",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_grib_temperature_on_moist_adiabat_pl(input_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import temperature_on_moist_adiabat

    ept = _make_input_fieldlist(THERMO_PL_FILE, "eqpt", input_type=input_type)

    # create target pressure fields for 700 and 500 hPa
    if input_type == "field":
        f = ept
        p = f.set(values=f.values * 0 + 50000.0, **{"parameter.variable": "pres"}).sync()
    else:
        p = []
        f = ept[0]
        for p_val in [50000.0]:
            p.append(f.set(values=f.values * 0 + p_val, **{"parameter.variable": "pres"}).sync())
        p = ekd.FieldList.from_fields(p)

    out = temperature_on_moist_adiabat(ept, p)

    assert isinstance(out, type(ept))
    if input_type == "fieldlist":
        assert len(out) == len(p)
        assert (np.array(out.get("parameter.variable")) == "t").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "t",
        # "metadata.shortName": "t",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [261.5682031250, 261.5682031250],
        [254.4197656250, 254.4197656250],
        [247.0369531250, 247.0369531250],
    ])
    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "t"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "t",
                "metadata.shortName": "t",
                "vertical.level": 850,
                "metadata.levelist": 850,
                "vertical.level_type": "pressure",
                "metadata.typeOfLevel": "isobaricInhPa",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_grib_wet_bulb_temperature_from_specific_humidity_pl(input_type, pres_type):
    import earthkit.data as ekd

    from earthkit.meteo.thermo.fieldlist import wet_bulb_temperature_from_specific_humidity

    t = _make_input_fieldlist(THERMO_PL_FILE, "t", input_type=input_type)
    q = _make_input_fieldlist(THERMO_PL_FILE, "q", input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = wet_bulb_temperature_from_specific_humidity(t, q, p)

    assert isinstance(out, type(t))
    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert (np.array(out.get("parameter.variable")) == "wbgt").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "wbgt",
        # "metadata.shortName": "wbgt",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [280.6111718750, 282.1932031250],
        [285.2986718750, 283.5408593750],
        [270.6502343750, 272.2322656250],
    ])

    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    # TODO: the code below only works for GRIB 2
    return

    if input_type == "fieldlist":
        # GRIB metadata
        f = out[0].sync()
        assert f.get("metadata.shortName") == "wbgt"

        # write back to GRIB
        with temp_file() as tmp:
            f.to_target("file", tmp)
            f_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

            ref_metadata = {
                "parameter.variable": "wbgt",
                "metadata.shortName": "wbgt",
                "vertical.level": 850,
                "metadata.levelist": 850,
                "vertical.level_type": "pressure",
                "metadata.typeOfLevel": "isobaricInhPa",
            }

            for k, v in ref_metadata.items():
                assert f_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_grib_specific_gas_constant_pl(input_type):
    from earthkit.meteo.thermo.fieldlist import specific_gas_constant

    q = _make_input_fieldlist(THERMO_PL_FILE, "q", input_type=input_type)
    out = specific_gas_constant(q)

    assert isinstance(out, type(q))
    if input_type == "fieldlist":
        assert len(out) == len(q)
        assert (np.array(out.get("parameter.variable")) == "R").all()

    # field metadata
    if input_type == "field":
        f = out
    else:
        f = out[0]
    ref_metadata = {
        "parameter.variable": "R",
        # "metadata.shortName": "R",
        "vertical.level": 850,
        # "metadata.levelist": 850,
        "vertical.level_type": "pressure",
        # "metadata.typeOfLevel": "isobaricInhPa",
    }
    for k, v in ref_metadata.items():
        assert f.get(k) == v

    # values
    ref_vals = np.array([
        [287.7828207241, 288.0429382970],
        [289.0469322353, 288.7071240214],
        [287.2189137762, 287.4112360750],
    ])

    assert f.shape == (3, 12)
    np.testing.assert_allclose(f.to_numpy()[:, :2], ref_vals)

    # TODO: this parameter is not yet supported in GRIB, so we cannot test the metadata and write back to GRIB
