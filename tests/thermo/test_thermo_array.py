# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import os

import numpy as np
import pytest

from earthkit.meteo import thermo

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})


def data_file(name):
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", name)


def save_test_reference(file_name, data):
    """Helper function to save test reference data into csv"""
    np.savetxt(
        data_file(file_name),
        np.column_stack(tuple(data.values())),
        delimiter=",",
        header=",".join(list(data.keys())),
    )


class ThermoInputData:
    """Helper class to load thermo input data."""

    def __init__(self, file_name="t_hum_p_data.csv"):
        self.file_name = file_name

        d = np.genfromtxt(
            data_file(self.file_name),
            delimiter=",",
            names=True,
        )
        self.t = d["t"]
        self.td = d["td"]
        self.r = d["r"]
        self.q = d["q"]
        self.p = d["p"]


def test_celsius_to_kelvin():
    t = np.array([-10, 23.6])
    v = thermo.array.celsius_to_kelvin(t)
    v_ref = np.array([263.16, 296.76])
    np.testing.assert_allclose(v, v_ref)


def test_kelvin_to_celsius():
    t = np.array([263.16, 296.76])
    v = thermo.array.kelvin_to_celsius(t)
    v_ref = np.array([-10, 23.6])
    np.testing.assert_allclose(v, v_ref)


def test_mixing_ratio_from_specific_humidity():
    q = np.array([0.008, 0.018])
    mr = thermo.array.mixing_ratio_from_specific_humidity(q)
    v_ref = np.array([0.0080645161, 0.0183299389])
    np.testing.assert_allclose(mr, v_ref, rtol=1e-7)


def test_specific_humidity_from_mixing_ratio():
    mr = np.array([0.0080645161, 0.0183299389])
    q = thermo.array.specific_humidity_from_mixing_ratio(mr)
    v_ref = np.array([0.008, 0.018])
    np.testing.assert_allclose(q, v_ref, rtol=1e-07)


def test_vapour_pressure_from_specific_humidity():
    q = np.array([0.008, 0.018])
    p = np.array([700, 1000]) * 100
    v_ref = np.array([895.992614, 2862.662152])
    vp = thermo.array.vapour_pressure_from_specific_humidity(q, p)
    np.testing.assert_allclose(vp, v_ref)


def test_vapour_pressure_from_mixing_ratio():
    mr = np.array([0.0080645161, 0.0183299389])
    p = np.array([700, 1000]) * 100
    v_ref = np.array([895.992614, 2862.662152])
    vp = thermo.array.vapour_pressure_from_mixing_ratio(mr, p)
    np.testing.assert_allclose(vp, v_ref)


def test_specific_humidity_from_vapour_pressure():
    vp = np.array([895.992614, 2862.662152, 10000])
    p = np.array([700, 1000, 50]) * 100
    v_ref = np.array([0.008, 0.018, np.nan])
    q = thermo.array.specific_humidity_from_vapour_pressure(vp, p)
    np.testing.assert_allclose(q, v_ref)

    # single pressure value
    vp = np.array([895.992614, 2862.662152, 100000])
    p = 700 * 100.0
    v_ref = np.array([0.008, 0.0258354146, np.nan])
    q = thermo.array.specific_humidity_from_vapour_pressure(vp, p)
    np.testing.assert_allclose(q, v_ref)

    # numbers
    vp = 895.992614
    p = 700 * 100.0
    v_ref = 0.008
    q = thermo.array.specific_humidity_from_vapour_pressure(vp, p)
    np.testing.assert_allclose(q, v_ref)

    vp = 100000
    p = 700 * 100.0
    v_ref = np.nan
    q = thermo.array.specific_humidity_from_vapour_pressure(vp, p)
    np.testing.assert_allclose(q, v_ref)


def test_mixing_ratio_from_vapour_pressure():
    vp = np.array([895.992614, 2862.662152, 100000])
    p = np.array([700, 1000, 50]) * 100
    v_ref = np.array([0.0080645161, 0.0183299389, np.nan])
    mr = thermo.array.mixing_ratio_from_vapour_pressure(vp, p)
    np.testing.assert_allclose(mr, v_ref, rtol=1e-07)

    # numbers
    vp = 895.992614
    p = 700 * 100.0
    v_ref = 0.0080645161
    q = thermo.array.mixing_ratio_from_vapour_pressure(vp, p)
    np.testing.assert_allclose(q, v_ref)

    vp = 100000.0
    p = 50 * 100.0
    v_ref = np.nan
    q = thermo.array.mixing_ratio_from_vapour_pressure(vp, p)
    np.testing.assert_allclose(q, v_ref)


def test_saturation_vapour_pressure():
    ref_file = "sat_vp.csv"
    phases = ["mixed", "water", "ice"]

    # o = {"t": thermo.array.celsius_to_kelvin(np.linspace(-40.0, 56.0, 49))}
    # for phase in ["mixed", "water", "ice"]:
    #     o[phase] = thermo.array.saturation_vapour_pressure(o["t"], phase=phase)
    # save_test_reference(ref_file, o)

    d = np.genfromtxt(
        data_file(ref_file),
        delimiter=",",
        names=True,
    )

    for phase in phases:
        svp = thermo.array.saturation_vapour_pressure(d["t"], phase=phase)
        np.testing.assert_allclose(svp, d[phase])

    # scalar value
    t = 267.16
    v_ref = {
        "mixed": 3.807622914202970037e02,
        "water": 3.909282234208898785e02,
        "ice": 3.685208149695831139e02,
    }
    for phase, v in v_ref.items():
        svp = thermo.array.saturation_vapour_pressure(t, phase=phase)
        np.testing.assert_allclose(svp, v)


def test_saturation_mixing_ratio():
    ref_file = "sat_mr.csv"
    phases = ["mixed", "water", "ice"]

    # t = thermo.array.celsius_to_kelvin(np.linspace(-40.0, 56.0, 25))
    # p = [1000, 950, 850, 700]
    # t_num = len(t)
    # t = np.repeat(t, repeats=len(p))
    # p = np.array(p * t_num) * 100.0
    # o = {"t": t, "p": p}
    # for phase in ["mixed", "water", "ice"]:
    #     o[phase] = thermo.array.saturation_mixing_ratio(t, p, phase=phase)
    # save_test_reference(ref_file, o)

    d = np.genfromtxt(
        data_file(ref_file),
        delimiter=",",
        names=True,
    )

    for phase in phases:
        mr = thermo.array.saturation_mixing_ratio(d["t"], d["p"], phase=phase)
        np.testing.assert_allclose(mr, d[phase])


def test_saturation_specific_humidity():
    ref_file = "sat_q.csv"
    phases = ["mixed", "water", "ice"]

    # t = thermo.array.celsius_to_kelvin(np.linspace(-40.0, 56.0, 25))
    # p = [1000, 950, 850, 700]
    # t_num = len(t)
    # t = np.repeat(t, repeats=len(p))
    # p = np.array(p * t_num) * 100.0
    # o = {"t": t, "p": p}
    # for phase in ["mixed", "water", "ice"]:
    #     o[phase] = thermo.array.saturation_specific_humidity(t, p, phase=phase)
    # save_test_reference(ref_file, o)

    d = np.genfromtxt(
        data_file(ref_file),
        delimiter=",",
        names=True,
    )

    for phase in phases:
        mr = thermo.array.saturation_specific_humidity(d["t"], d["p"], phase=phase)
        np.testing.assert_allclose(mr, d[phase])


def test_saturation_vapour_pressure_slope():
    ref_file = "sat_vp_slope.csv"
    phases = ["mixed", "water", "ice"]

    # o = {"t": thermo.array.celsius_to_kelvin(np.linspace(-40.0, 56.0, 49))}
    # for phase in ["mixed", "water", "ice"]:
    #     o[phase] = thermo.array.saturation_vapour_pressure_slope(o["t"], phase=phase)
    # save_test_reference(ref_file, o)

    d = np.genfromtxt(
        data_file(ref_file),
        delimiter=",",
        names=True,
    )

    for phase in phases:
        svp = thermo.array.saturation_vapour_pressure_slope(d["t"], phase=phase)
        np.testing.assert_allclose(svp, d[phase])


def test_saturation_mixing_ratio_slope():
    ref_file = "sat_mr_slope.csv"
    phases = ["mixed", "water", "ice"]

    # t = thermo.array.celsius_to_kelvin(np.linspace(-40.0, 56.0, 25))
    # p = [1000, 950, 850, 700]
    # t_num = len(t)
    # t = np.repeat(t, repeats=len(p))
    # p = np.array(p * t_num) * 100.0
    # o = {"t": t, "p": p}
    # for phase in ["mixed", "water", "ice"]:
    #     o[phase] = thermo.array.saturation_mixing_ratio_slope(t, p, phase=phase)
    # save_test_reference(ref_file, o)

    d = np.genfromtxt(
        data_file(ref_file),
        delimiter=",",
        names=True,
    )

    for phase in phases:
        svp = thermo.array.saturation_mixing_ratio_slope(d["t"], d["p"], phase=phase)
        np.testing.assert_allclose(svp, d[phase])

    v_ref = np.array([np.nan])
    t = thermo.array.celsius_to_kelvin(np.array([200]))
    p = np.array([1000])
    svp = thermo.array.saturation_mixing_ratio_slope(t, p, phase="mixed")
    np.testing.assert_allclose(svp, v_ref)

    # numbers
    t = [283.0, 600.0]
    p = [1e5, 1e5]
    v_ref = [0.0005189819, np.nan]
    for i in range(len(t)):
        svp = thermo.array.saturation_mixing_ratio_slope(t[i], p[i], phase="mixed")
        np.testing.assert_allclose(svp, v_ref[i])


def test_saturation_specific_humidity_slope():
    ref_file = "sat_q_slope.csv"
    phases = ["mixed", "water", "ice"]

    # t = thermo.array.celsius_to_kelvin(np.linspace(-40.0, 56.0, 25))
    # p = [1000, 950, 850, 700]
    # t_num = len(t)
    # t = np.repeat(t, repeats=len(p))
    # p = np.array(p * t_num) * 100.0
    # o = {"t": t, "p": p}
    # for phase in ["mixed", "water", "ice"]:
    #     o[phase] = thermo.array.saturation_specific_humidity_slope(t, p, phase=phase)
    # save_test_reference(ref_file, o)

    d = np.genfromtxt(
        data_file(ref_file),
        delimiter=",",
        names=True,
    )

    for phase in phases:
        svp = thermo.array.saturation_specific_humidity_slope(d["t"], d["p"], phase=phase)
        np.testing.assert_allclose(svp, d[phase])

    v_ref = np.array([np.nan])
    t = thermo.array.celsius_to_kelvin(np.array([200]))
    p = np.array([1000])
    svp = thermo.array.saturation_specific_humidity_slope(t, p, phase="mixed")
    np.testing.assert_allclose(svp, v_ref)

    # numbers
    t = [283.0, 600.0]
    p = [1e5, 1e5]
    v_ref = [0.0005111349, np.nan]
    for i in range(len(t)):
        svp = thermo.array.saturation_specific_humidity_slope(t[i], p[i], phase="mixed")
        np.testing.assert_allclose(svp, v_ref[i])


def test_temperature_from_saturation_vapour_pressure_1():
    ref_file = "sat_vp.csv"
    d = np.genfromtxt(
        data_file(ref_file),
        delimiter=",",
        names=True,
    )

    t = thermo.array.temperature_from_saturation_vapour_pressure(d["water"])
    assert np.allclose(t, d["t"], equal_nan=True)


@pytest.mark.parametrize(
    "es,expected_values",
    [
        (4.2, 219.7796336743947),
        (0, np.nan),
    ],
)
def test_temperature_from_saturation_vapour_pressure_2(es, expected_values):

    multi = isinstance(es, list)
    if multi:
        es = np.array(es)
        expected_values = np.array(expected_values)

    t = thermo.array.temperature_from_saturation_vapour_pressure(es)
    if multi:
        np.testing.assert_allclose(t, expected_values, equal_nan=True)
    else:
        assert np.isclose(t, expected_values, equal_nan=True)


def test_relative_humidity_from_dewpoint():
    t = thermo.array.celsius_to_kelvin(np.array([20.0, 20, 0, 35, 5, -15, 25]))
    td = thermo.array.celsius_to_kelvin(np.array([20.0, 10, -10, 32, -15, -24, -3]))
    # reference was tested with an online relhum calculator at:
    # https://bmcnoldy.rsmas.miami.edu/Humidity.html
    v_ref = np.array(
        [
            100.0000000000,
            52.5224541378,
            46.8714823296,
            84.5391163313,
            21.9244774232,
            46.1081101229,
            15.4779832381,
        ]
    )
    r = thermo.array.relative_humidity_from_dewpoint(t, td)
    np.testing.assert_allclose(r, v_ref, rtol=1e-05)


def test_relative_humidity_from_specific_humidity():
    t = thermo.array.celsius_to_kelvin(np.array([-29.2884, -14.4118, -5.9235, 9.72339, 18.4514]))
    p = np.array([300, 400, 500, 700, 850]) * 100.0
    q = np.array(
        [
            0.000845416891024797,
            0.00277950354211498,
            0.00464489207661245,
            0.0076785187585422,
            0.0114808182580539,
        ]
    )
    v_ref = np.array(
        [
            99.70488530734642,
            100.25885732613531,
            97.15956159465799,
            71.37937968160273,
            73.41420898756694,
        ]
    )
    r = thermo.array.relative_humidity_from_specific_humidity(t, q, p)
    np.testing.assert_allclose(r, v_ref)


def test_specific_humidity_from_dewpoint():
    td = thermo.array.celsius_to_kelvin(
        np.array([21.78907, 19.90885, 16.50236, 7.104064, -0.3548709, -16.37916])
    )
    p = np.array([967.5085, 936.3775, 872.248, 756.1647, 649.157, 422.4207]) * 100
    v_ref = np.array(
        [
            0.0169461501,
            0.0155840075,
            0.0134912382,
            0.0083409720,
            0.0057268584,
            0.0025150791,
        ]
    )
    q = thermo.array.specific_humidity_from_dewpoint(td, p)
    np.testing.assert_allclose(q, v_ref, rtol=1e-05)


def test_specific_humidity_from_relative_humidity():
    t = thermo.array.celsius_to_kelvin(np.array([-29.2884, -14.4118, -5.9235, 9.72339, 18.4514]))
    p = np.array([300, 400, 500, 700, 850]) * 100.0
    r = np.array(
        [
            99.70488530734642,
            100.25885732613531,
            97.15956159465799,
            71.37937968160273,
            73.41420898756694,
        ]
    )
    v_ref = np.array(
        [
            0.000845416891024797,
            0.00277950354211498,
            0.00464489207661245,
            0.0076785187585422,
            0.0114808182580539,
        ]
    )
    q = thermo.array.specific_humidity_from_relative_humidity(t, r, p)
    np.testing.assert_allclose(q, v_ref)


@pytest.mark.parametrize(
    "t,r,expected_values",
    [
        (
            [20.0, 20, 0, 35, 5, -15, 25, 25],
            [
                100.0000000000,
                52.5224541378,
                46.8714823296,
                84.5391163313,
                21.9244774232,
                46.1081101229,
                15.4779832381,
                0,
            ],
            [20.0, 10, -10, 32, -15, -24, -3, np.nan],
        ),
        (20.0, 52.5224541378, 10.0),
    ],
)
def test_dewpoint_from_relative_humidity(t, r, expected_values):
    # reference was tested with an online relhum calculator at:
    # https://bmcnoldy.rsmas.miami.edu/Humidity.html

    multi = isinstance(t, list)
    if multi:
        t = np.array(t)
        r = np.array(r)
        expected_values = np.array(expected_values)

    t = thermo.array.celsius_to_kelvin(t)
    v_ref = thermo.array.celsius_to_kelvin(expected_values)

    td = thermo.array.dewpoint_from_relative_humidity(t, r)
    if multi:
        assert np.allclose(td, v_ref, equal_nan=True)
    else:
        assert np.isclose(td, v_ref, equal_nan=True)


@pytest.mark.parametrize(
    "q,p,expected_values",
    [
        (
            [0.0169461501, 0.0155840075, 0.0134912382, 0.0083409720, 0.0057268584, 0.0025150791, 0],
            [967.5085, 936.3775, 872.248, 756.1647, 649.157, 422.4207, 422.4207],
            [21.78907, 19.90885, 16.50236, 7.104064, -0.3548709, -16.37916, np.nan],
        ),
        (0.0169461501, 967.508, 21.78907),
    ],
)
def test_dewpoint_from_specific_humidity(q, p, expected_values):
    multi = isinstance(q, list)
    if multi:
        q = np.array(q)
        p = np.array(p)
        expected_values = np.array(expected_values)

    p = p * 100.0
    v_ref = thermo.array.celsius_to_kelvin(expected_values)

    td = thermo.array.dewpoint_from_specific_humidity(q, p)
    if multi:
        assert np.allclose(td, v_ref, equal_nan=True)
    else:
        assert np.isclose(td, v_ref, equal_nan=True)


def test_virtual_temperature():
    t = np.array([286.4, 293.4])
    # w = np.array([0.02, 0.03])
    q = np.array([0.0196078431, 0.0291262136])
    v_ref = np.array([289.8130240470, 298.5937453245])
    tv = thermo.array.virtual_temperature(t, q)
    np.testing.assert_allclose(tv, v_ref)


def test_virtual_potential_temperature_temperature():
    t = np.array([286.4, 293.4])
    # w = np.array([0.02, 0.03])
    q = np.array([0.0196078431, 0.0291262136])
    p = np.array([100300.0, 95000.0])
    v_ref = np.array([289.5651110613, 303.0015650834])
    tv = thermo.array.virtual_potential_temperature(t, q, p)
    np.testing.assert_allclose(tv, v_ref)


def test_potential_temperature():
    t = np.array([252.16, 298.16])
    p = np.array([72350, 100500])
    v_ref = np.array([276.588026, 297.735455])
    th = thermo.array.potential_temperature(t, p)
    np.testing.assert_allclose(th, v_ref)


def test_temperature_from_potential_temperature():
    p = np.array([72350, 100500])
    th = np.array([276.588026, 297.735455])
    v_ref = np.array([252.16, 298.16])
    t = thermo.array.temperature_from_potential_temperature(th, p)
    np.testing.assert_allclose(t, v_ref)


def test_temperature_on_dry_adibat():
    t_def = np.array([252.16, 298.16])
    p_def = np.array([72350, 100500])
    p = np.array([700, 500]) * 100
    v_ref = np.array([249.792414, 244.246863])
    t = thermo.array.temperature_on_dry_adiabat(p, t_def, p_def)
    np.testing.assert_allclose(t, v_ref)

    # cross checking
    th1 = thermo.array.potential_temperature(t_def, p_def)
    th2 = thermo.array.potential_temperature(t, p)
    np.testing.assert_allclose(th1, th2)

    # multiple values along a single adiabat
    v_ref = np.array([249.792414, 226.898581])
    t = thermo.array.temperature_on_dry_adiabat(p, t_def[0], p_def[0])
    np.testing.assert_allclose(t, v_ref)


def test_pressure_on_dry_adibat():
    t_def = np.array([252.16, 298.16])
    p_def = np.array([72350, 100500])
    t = np.array([249.792414, 244.246863])
    v_ref = np.array([700, 500]) * 100
    p = thermo.array.pressure_on_dry_adiabat(t, t_def, p_def)
    np.testing.assert_allclose(p, v_ref)

    # cross checking
    th1 = thermo.array.potential_temperature(t_def, p_def)
    th2 = thermo.array.potential_temperature(t, p)
    np.testing.assert_allclose(th1, th2)

    # multiple values along a single adiabat
    v_ref = np.array([70000, 64709.699161])
    p = thermo.array.pressure_on_dry_adiabat(t, t_def[0], p_def[0])
    np.testing.assert_allclose(p, v_ref)


def test_lcl():
    t = np.array([10, 30, 43, 20]) + 273.16
    td = np.array([-2, 26, 5, 28]) + 273.16
    p = np.array([95000, 100500, 100100, 100200])

    # davies
    t_ref = np.array([268.706024, 298.200936, 270.517934, 303.138144])
    p_ref = np.array([79081.766347, 94862.350635, 57999.83367, 112654.210439])
    t_lcl = thermo.array.lcl_temperature(t, td, method="davies")
    np.testing.assert_allclose(t_lcl, t_ref)
    t_lcl, p_lcl = thermo.array.lcl(t, td, p, method="davies")
    np.testing.assert_allclose(t_lcl, t_ref)
    np.testing.assert_allclose(p_lcl, p_ref)

    # bolton
    t_ref = np.array([268.683018, 298.182282, 270.531264, 303.199544])
    p_ref = np.array([79058.068785, 94841.581142, 58009.838027, 112734.100243])
    t_lcl = thermo.array.lcl_temperature(t, td, method="bolton")
    np.testing.assert_allclose(t_lcl, t_ref)
    t_lcl, p_lcl = thermo.array.lcl(t, td, p, method="bolton")
    np.testing.assert_allclose(t_lcl, t_ref)
    np.testing.assert_allclose(p_lcl, p_ref)


def test_ept():
    data = ThermoInputData()

    ref_file = "eqpt.csv"
    methods = ["ifs", "bolton35", "bolton39"]

    # o = {}
    # for m in methods:
    #     o[f"{m}_td"] = thermo.array.ept_from_dewpoint(data.t, data.td, data.p, method=m)
    #     o[f"{m}_q"] = thermo.array.ept_from_specific_humidity(
    #         data.t, data.q, data.p, method=m
    #     )
    # save_test_reference(ref_file, o)

    ref = np.genfromtxt(
        data_file(ref_file),
        delimiter=",",
        names=True,
    )

    for m in methods:
        pt = thermo.array.ept_from_dewpoint(data.t, data.td, data.p, method=m)
        np.testing.assert_allclose(pt, ref[m + "_td"], err_msg=f"method={m}")
        pt = thermo.array.ept_from_specific_humidity(data.t, data.q, data.p, method=m)
        np.testing.assert_allclose(pt, ref[m + "_q"], err_msg=f"method={m}")


def test_saturation_ept():
    data = ThermoInputData()

    ref_file = "seqpt.csv"
    methods = ["ifs", "bolton35", "bolton39"]

    # o = {}
    # for m in methods:
    #     o[m] = thermo.array.saturation_ept(data.t, data.p, method=m)
    # save_test_reference(ref_file, o)

    ref = np.genfromtxt(
        data_file(ref_file),
        delimiter=",",
        names=True,
    )

    for m in methods:
        pt = thermo.array.saturation_ept(data.t, data.p, method=m)
        np.testing.assert_allclose(pt, ref[m], err_msg=f"method={m}")


def test_temperature_on_moist_adiabat():
    ref_file = "t_on_most_adiabat.csv"
    ept_methods = ["ifs", "bolton35", "bolton39"]
    t_methods = ["bisect", "newton"]

    # ept = np.array([220, 250, 273.16, 300, 330, 360, 400, 500, 600, 700, 800, 900])
    # p = np.array([1010, 1000, 925, 850, 700, 500, 300]) * 100
    # o = {"ept": np.repeat(ept, repeats=len(p)), "p": np.array(p.tolist() * len(ept))}
    # for m_ept in ept_methods:
    #     for m_t in t_methods:
    #         o[f"{m_ept}_{m_t}"] = thermo.array.temperature_on_moist_adiabat(
    #             o["ept"], o["p"], ept_method=m_ept, t_method=m_t
    #         )
    # save_test_reference(ref_file, o)

    ref = np.genfromtxt(
        data_file(ref_file),
        delimiter=",",
        names=True,
    )

    for m_ept in ept_methods:
        for m_t in t_methods:
            pt = thermo.array.temperature_on_moist_adiabat(
                ref["ept"], ref["p"], ept_method=m_ept, t_method=m_t
            )
            np.testing.assert_allclose(pt, ref[f"{m_ept}_{m_t}"], err_msg=f"method={m_ept}_{m_t}")


def test_wet_bulb_temperature():
    data = ThermoInputData()

    ref_file = "t_wet.csv"
    ept_methods = ["ifs", "bolton35", "bolton39"]
    t_methods = ["bisect", "newton"]

    # o = {}
    # for m_ept in ept_methods:
    #     for m_t in t_methods:
    #         o[f"{m_ept}_{m_t}_td"] = thermo.array.wet_bulb_temperature_from_dewpoint(
    #             data.t, data.td, data.p, ept_method=m_ept, t_method=m_t
    #         )
    #         o[f"{m_ept}_{m_t}_q"] = thermo.array.wet_bulb_temperature_from_specific_humidity(
    #             data.t, data.q, data.p, ept_method=m_ept, t_method=m_t
    #         )
    # save_test_reference(ref_file, o)

    ref = np.genfromtxt(
        data_file(ref_file),
        delimiter=",",
        names=True,
    )

    for m_ept in ept_methods:
        for m_t in t_methods:
            pt = thermo.array.wet_bulb_temperature_from_dewpoint(
                data.t, data.td, data.p, ept_method=m_ept, t_method=m_t
            )
            np.testing.assert_allclose(
                pt,
                ref[f"{m_ept}_{m_t}_td"],
                rtol=1e-03,
                atol=0,
                err_msg=f"method={m_ept}_{m_t}_td",
            )
            pt = thermo.array.wet_bulb_temperature_from_specific_humidity(
                data.t, data.q, data.p, ept_method=m_ept, t_method=m_t
            )
            np.testing.assert_allclose(
                pt,
                ref[f"{m_ept}_{m_t}_q"],
                rtol=1e-03,
                atol=0,
                err_msg=f"method={m_ept}_{m_t}_q",
            )


def test_wet_bulb_potential_temperature():
    data = ThermoInputData()

    ref_file = "t_wetpt.csv"
    ept_methods = ["ifs", "bolton35", "bolton39"]
    t_methods = ["bisect", "newton", "direct"]

    # o = {}
    # for m_ept in ept_methods:
    #     for m_t in t_methods:
    #         o[
    #             f"{m_ept}_{m_t}_td"
    #         ] = thermo.array.wet_bulb_potential_temperature_from_dewpoint(
    #             data.t, data.td, data.p, ept_method=m_ept, t_method=m_t
    #         )
    #         o[
    #             f"{m_ept}_{m_t}_q"
    #         ] = thermo.array.wet_bulb_potential_temperature_from_specific_humidity(
    #             data.t, data.q, data.p, ept_method=m_ept, t_method=m_t
    #         )
    # save_test_reference(ref_file, o)

    ref = np.genfromtxt(
        data_file(ref_file),
        delimiter=",",
        names=True,
    )

    for m_ept in ept_methods:
        for m_t in t_methods:
            pt = thermo.array.wet_bulb_potential_temperature_from_dewpoint(
                data.t, data.td, data.p, ept_method=m_ept, t_method=m_t
            )
            np.testing.assert_allclose(
                pt,
                ref[f"{m_ept}_{m_t}_td"],
                rtol=1e-03,
                atol=0,
                err_msg=f"method={m_ept}_{m_t}_td",
            )
            pt = thermo.array.wet_bulb_potential_temperature_from_specific_humidity(
                data.t, data.q, data.p, ept_method=m_ept, t_method=m_t
            )
            np.testing.assert_allclose(
                pt,
                ref[f"{m_ept}_{m_t}_q"],
                rtol=1e-03,
                atol=0,
                err_msg=f"method={m_ept}_{m_t}_q",
            )
