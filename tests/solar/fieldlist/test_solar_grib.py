# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime

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
GRIB_FILE = "tuv_pl.grib"


def _make_input_fieldlist(filename, param, input_type="fieldlist", level_type="pressure"):
    fl = _get_fieldlist(filename)
    if level_type == "pressure":
        param_fl = fl.sel({"parameter.variable": param, "vertical.level_type": level_type}).order_by("level")
    else:
        param_fl = fl.sel({"parameter.variable": param})

    return param_fl if input_type == "fieldlist" else param_fl[0]


def _get_fieldlist(name, sample=False):
    import earthkit.data as ekd

    if sample:
        return ekd.from_source("sample", name).to_fieldlist()
    else:
        path = testing.get_test_data(name, "test-data")
        fl = ekd.from_source("file", path).to_fieldlist()
        return fl


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_grib_cos_solar_zenith_angle(input_type):
    import earthkit.data as ekd

    import earthkit.meteo.solar.fieldlist as solar

    dt = datetime.datetime(2021, 6, 21, 12, 0)  # summer solstice at noon
    data = _make_input_fieldlist(GRIB_FILE, "u", input_type=input_type)
    out = solar.cos_solar_zenith_angle(dt, data)

    assert isinstance(out, type(data))

    if input_type == "fieldlist":
        assert len(out) == len(data)
        ref_metadata = {
            "parameter.variable": ["cossza"] * len(out),
            "vertical.level": LEVELS,
            "vertical.level_type": ["pressure"] * len(out),
        }
        field = out[0]
    elif input_type == "field":
        ref_metadata = {
            "parameter.variable": "cossza",
            "vertical.level": 1000,
            "vertical.level_type": "pressure",
        }
        field = out

    for k, v in ref_metadata.items():
        assert out.get(k) == v

    # values
    ref_vals = np.array([[0.3980160529, 0.3980160529], [0.8033725310, 0.7433341190], [0.9934659883, 0.8894764082]])
    assert field.shape == (7, 12)
    np.testing.assert_allclose(field.to_numpy()[:3, :2], ref_vals)

    # GRIB metadata
    field = field.sync()
    assert field.get("metadata.shortName") == "cossza"

    # write back to GRIB
    with temp_file() as tmp:
        field.to_target("file", tmp)
        field_saved = ekd.from_source("file", tmp).to_fieldlist()[0]

        ref_metadata = {
            "parameter.variable": "cossza",
            "metadata.shortName": "cossza",
            "vertical.level": 1000,
            "metadata.levelist": 1000,
            "vertical.level_type": "pressure",
            "metadata.typeOfLevel": "isobaricInhPa",
        }

        for k, v in ref_metadata.items():
            assert field_saved.get(k) == v


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_grib_cos_solar_zenith_angle_integrated(input_type):

    import earthkit.meteo.solar.fieldlist as solar

    begin_dt = datetime.datetime(2021, 6, 21, 12, 0)  # summer solstice at noon
    end_dt = datetime.datetime(2021, 6, 21, 15, 0)  # 3 hours later
    data = _make_input_fieldlist(GRIB_FILE, "u", input_type=input_type)
    out = solar.cos_solar_zenith_angle_integrated(begin_dt, end_dt, data)

    assert isinstance(out, type(data))

    if input_type == "fieldlist":
        assert len(out) == len(data)
        ref_metadata = {
            "parameter.variable": ["cossza_integrated"] * len(out),
            "vertical.level": LEVELS,
            "vertical.level_type": ["pressure"] * len(out),
        }
        field = out[0]
    elif input_type == "field":
        ref_metadata = {
            "parameter.variable": "cossza_integrated",
            "vertical.level": 1000,
            "vertical.level_type": "pressure",
        }
        field = out

    for k, v in ref_metadata.items():
        assert out.get(k) == v

    # values
    ref_vals = np.array([[0.3980200995, 0.3980200995], [0.7784063554, 0.6636375477], [0.9502192569, 0.7514338509]])
    assert field.shape == (7, 12)
    np.testing.assert_allclose(field.to_numpy()[:3, :2], ref_vals)

    # TODO: cos_solar_zenith_angle_integrated is not yet supported in GRIB, so we cannot test
    # the metadata and write back to GRIB


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_grib_toa_incident_solar_radiation(input_type):

    import earthkit.meteo.solar.fieldlist as solar

    begin_dt = datetime.datetime(2021, 6, 21, 12, 0)  # summer solstice at noon
    end_dt = datetime.datetime(2021, 6, 21, 15, 0)  # 3 hours later
    data = _make_input_fieldlist(GRIB_FILE, "u", input_type=input_type)
    out = solar.toa_incident_solar_radiation(begin_dt, end_dt, data)

    assert isinstance(out, type(data))

    if input_type == "fieldlist":
        assert len(out) == len(data)
        ref_metadata = {
            "parameter.variable": ["toa_incident_solar_radiation"] * len(out),
            "vertical.level": LEVELS,
            "vertical.level_type": ["pressure"] * len(out),
        }
        field = out[0]
    elif input_type == "field":
        ref_metadata = {
            "parameter.variable": "toa_incident_solar_radiation",
            "vertical.level": 1000,
            "vertical.level_type": "pressure",
        }
        field = out

    for k, v in ref_metadata.items():
        assert out.get(k) == v

    # values
    ref_vals = np.array([
        [1882745.2876005317, 1882745.2876005317],
        [3682078.0591244297, 3139190.6148891458],
        [4494800.9882375803, 3554492.3520308542],
    ])
    assert field.shape == (7, 12)
    np.testing.assert_allclose(field.to_numpy()[:3, :2], ref_vals)

    # TODO: toa_incident_solar_radiation is not yet supported in GRIB, so we cannot test
    #  the metadata and write back to GRIB
