# (C) Copyright 2026 ECMWF.
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
pytestmark = pytest.mark.skipif(NO_EKD, reason="EKD is not installed")

# Latitude/longitude pairs for test fields (degrees)
LATITUDES = [[40.0, 50.0], [30.0, 60.0], [-20.0, 10.0]]
LONGITUDES = [[18.0, 5.0], [10.0, -30.0], [120.0, 80.0]]

# Dummy values to initialise input fields (the solar functions ignore them)
DUMMY_VALUES = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

PARAMETERS = {
    "test": {"variable": "2t", "units": "K"},
}


def _make_input_fieldlist(values, latitudes, longitudes, input_type="fieldlist"):
    from earthkit.data import Field, FieldList

    param_def = PARAMETERS["test"]

    if input_type == "field":
        return Field.from_components(
            values=np.array(values[0]),
            parameter={"variable": param_def["variable"], "units": param_def["units"]},
            geography={"latitudes": np.array(latitudes[0]), "longitudes": np.array(longitudes[0])},
        )
    elif input_type == "fieldlist":
        fl = []
        for v, lat, lon in zip(values, latitudes, longitudes):
            fl.append(
                Field.from_components(
                    values=np.array(v),
                    parameter={"variable": param_def["variable"], "units": param_def["units"]},
                    geography={"latitudes": np.array(lat), "longitudes": np.array(lon)},
                )
            )
        return FieldList.from_fields(fl)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_cos_solar_zenith_angle(input_type):
    import earthkit.meteo.solar.array as array_solar
    import earthkit.meteo.solar.fieldlist as solar

    date = datetime.datetime(2024, 4, 22, 12, 0, 0)
    data = _make_input_fieldlist(DUMMY_VALUES, LATITUDES, LONGITUDES, input_type=input_type)
    out = solar.cos_solar_zenith_angle(date, data)

    assert isinstance(out, type(data))

    if input_type == "fieldlist":
        assert len(out) == len(data)
        assert out.get("parameter.variable") == ["cossza"] * len(data)
        for f, lat_vals, lon_vals in zip(out, LATITUDES, LONGITUDES):
            ref = array_solar.cos_solar_zenith_angle(date, np.array(lat_vals), np.array(lon_vals))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "cossza"
        ref = array_solar.cos_solar_zenith_angle(date, np.array(LATITUDES[0]), np.array(LONGITUDES[0]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_cos_solar_zenith_angle_integrated(input_type):
    import earthkit.meteo.solar.array as array_solar
    import earthkit.meteo.solar.fieldlist as solar

    begin_date = datetime.datetime(2024, 4, 22)
    end_date = datetime.datetime(2024, 4, 23)
    data = _make_input_fieldlist(DUMMY_VALUES, LATITUDES, LONGITUDES, input_type=input_type)
    out = solar.cos_solar_zenith_angle_integrated(begin_date, end_date, data)

    assert isinstance(out, type(data))

    if input_type == "fieldlist":
        assert len(out) == len(data)
        assert out.get("parameter.variable") == ["cossza_integrated"] * len(data)
        for f, lat_vals, lon_vals in zip(out, LATITUDES, LONGITUDES):
            ref = array_solar.cos_solar_zenith_angle_integrated(
                begin_date, end_date, np.array(lat_vals), np.array(lon_vals)
            )
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "cossza_integrated"
        ref = array_solar.cos_solar_zenith_angle_integrated(
            begin_date, end_date, np.array(LATITUDES[0]), np.array(LONGITUDES[0])
        )
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("integration_order", [1, 2, 3, 4])
def test_fieldlist_cos_solar_zenith_angle_integrated_orders(input_type, integration_order):
    import earthkit.meteo.solar.array as array_solar
    import earthkit.meteo.solar.fieldlist as solar

    begin_date = datetime.datetime(2024, 4, 22)
    end_date = datetime.datetime(2024, 4, 23)
    data = _make_input_fieldlist(DUMMY_VALUES, LATITUDES, LONGITUDES, input_type=input_type)
    out = solar.cos_solar_zenith_angle_integrated(begin_date, end_date, data, integration_order=integration_order)

    assert isinstance(out, type(data))

    if input_type == "fieldlist":
        for f, lat_vals, lon_vals in zip(out, LATITUDES, LONGITUDES):
            ref = array_solar.cos_solar_zenith_angle_integrated(
                begin_date,
                end_date,
                np.array(lat_vals),
                np.array(lon_vals),
                integration_order=integration_order,
            )
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        ref = array_solar.cos_solar_zenith_angle_integrated(
            begin_date,
            end_date,
            np.array(LATITUDES[0]),
            np.array(LONGITUDES[0]),
            integration_order=integration_order,
        )
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_toa_incident_solar_radiation(input_type):
    import earthkit.meteo.solar.array as array_solar
    import earthkit.meteo.solar.fieldlist as solar

    begin_date = datetime.datetime(2024, 4, 22)
    end_date = datetime.datetime(2024, 4, 23)
    data = _make_input_fieldlist(DUMMY_VALUES, LATITUDES, LONGITUDES, input_type=input_type)
    out = solar.toa_incident_solar_radiation(begin_date, end_date, data)

    assert isinstance(out, type(data))

    if input_type == "fieldlist":
        assert len(out) == len(data)
        assert out.get("parameter.variable") == ["toa_incident_solar_radiation"] * len(data)
        for f, lat_vals, lon_vals in zip(out, LATITUDES, LONGITUDES):
            ref = array_solar.toa_incident_solar_radiation(begin_date, end_date, np.array(lat_vals), np.array(lon_vals))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "toa_incident_solar_radiation"
        ref = array_solar.toa_incident_solar_radiation(
            begin_date, end_date, np.array(LATITUDES[0]), np.array(LONGITUDES[0])
        )
        np.testing.assert_allclose(out.values, ref)
