# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""
Tests for the array level thermo functions
"""

import os

import numpy as np
import pytest

from earthkit.meteo import thermo
from earthkit.meteo.utils.testing import ARRAY_BACKENDS

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})


def data_file(name):
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", name)


def read_data_file(path):
    import numpy as np

    d = np.genfromtxt(
        data_file(path),
        delimiter=",",
        names=True,
    )
    return d


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

    def __init__(self, array_backend, file_name="t_hum_p_data.csv"):
        self.file_name = file_name

        d = read_data_file(self.file_name)

        self.t = array_backend.asarray(d["t"])
        self.td = array_backend.asarray(d["td"])
        self.r = array_backend.asarray(d["r"])
        self.q = array_backend.asarray(d["q"])
        self.p = array_backend.asarray(d["p"])


# high level method call
def test_high_level_celsius_to_kelvin():
    t = np.array([-10, 23.6])
    v = thermo.celsius_to_kelvin(t)
    v_ref = np.array([263.16, 296.76])
    np.testing.assert_allclose(v, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("t, v_ref", [([-10, 23.6], [263.16, 296.76])])
def test_celsius_to_kelvin(t, v_ref, array_backend):
    t, v_ref = array_backend.asarray(t, v_ref)
    tk = thermo.array.celsius_to_kelvin(t)
    assert array_backend.allclose(tk, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("t, v_ref", [([263.16, 296.76], [-10, 23.6])])
def test_kelvin_to_celsius(t, v_ref, array_backend):
    t, v_ref = array_backend.asarray(t, v_ref)
    tc = thermo.array.kelvin_to_celsius(t)
    assert array_backend.allclose(tc, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("q, v_ref", [([0.008, 0.018], [0.0080645161, 0.0183299389])])
def test_mixing_ratio_from_specific_humidity(q, v_ref, array_backend):
    q, v_ref = array_backend.asarray(q, v_ref)
    mr = thermo.array.mixing_ratio_from_specific_humidity(q)
    assert array_backend.allclose(mr, v_ref, equal_nan=True, rtol=1e-07)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("mr, v_ref", [([0.0080645161, 0.0183299389], [0.008, 0.018])])
def test_specific_humidity_from_mixing_ratio(mr, v_ref, array_backend):
    mr, v_ref = array_backend.asarray(mr, v_ref)
    q = thermo.array.specific_humidity_from_mixing_ratio(mr)
    assert array_backend.allclose(q, v_ref, equal_nan=True, rtol=1e-07)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("q, p, v_ref", [([0.008, 0.018], [700, 1000], [895.992614, 2862.662152])])
def test_vapour_pressure_from_specific_humidity(q, p, v_ref, array_backend):
    q, p, v_ref = array_backend.asarray(q, p, v_ref)
    p = p * 100
    vp = thermo.array.vapour_pressure_from_specific_humidity(q, p)
    assert array_backend.allclose(vp, v_ref, equal_nan=True)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "mr, p, v_ref", [([0.0080645161, 0.0183299389], [700, 1000], [895.992614, 2862.662152])]
)
def test_vapour_pressure_from_mixing_ratio(mr, p, v_ref, array_backend):
    mr, p, v_ref = array_backend.asarray(mr, p, v_ref)
    p = p * 100
    vp = thermo.array.vapour_pressure_from_mixing_ratio(mr, p)
    assert array_backend.allclose(vp, v_ref, equal_nan=True)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "vp, p, v_ref",
    [
        ([895.992614, 2862.662152, 10000], [700, 1000, 50], [0.008, 0.018, np.nan]),
        ([895.992614, 2862.662152, 100000], 700, [0.008, 0.0258354146, np.nan]),
        (895.992614, 700, 0.008),
        (100000, 700, np.nan),
    ],
)
def test_specific_humidity_from_vapour_pressure(vp, p, v_ref, array_backend):
    vp, p, v_ref = array_backend.asarray(vp, p, v_ref)
    p = p * 100
    q = thermo.array.specific_humidity_from_vapour_pressure(vp, p)

    assert array_backend.allclose(q, v_ref, equal_nan=True)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "vp, p, v_ref",
    [
        ([895.992614, 2862.662152, 10000], [700, 1000, 50], [0.0080645161, 0.0183299389, np.nan]),
        ([895.992614, 2862.662152, 100000], 700, [0.0080645161, 0.0265205849, np.nan]),
        (895.992614, 700, 0.0080645161),
        (100000.0, 700.0, np.nan),
    ],
)
def test_mixing_ratio_from_vapour_pressure(vp, p, v_ref, array_backend):

    vp, p, v_ref = array_backend.asarray(vp, p, v_ref)
    p = p * 100
    mr = thermo.array.mixing_ratio_from_vapour_pressure(vp, p)

    assert array_backend.allclose(mr, v_ref, equal_nan=True)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("phase", ["mixed", "water", "ice"])
def test_saturation_vapour_pressure_1(phase, array_backend):
    ref_file = "sat_vp.csv"
    # phases = ["mixed", "water", "ice"]

    # o = {"t": thermo.array.celsius_to_kelvin(np.linspace(-40.0, 56.0, 49))}
    # for phase in ["mixed", "water", "ice"]:
    #     o[phase] = thermo.array.saturation_vapour_pressure(o["t"], phase=phase)
    # save_test_reference(ref_file, o)

    d = read_data_file(ref_file)

    t = array_backend.asarray(d["t"])
    v_ref = array_backend.asarray(d[phase])

    svp = thermo.array.saturation_vapour_pressure(t, phase=phase)
    assert array_backend.allclose(svp, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "t,v_ref,phase",
    [
        (267.16, 3.807622914202970037e02, "mixed"),
        (267.16, 3.909282234208898785e02, "water"),
        (267.16, 3.685208149695831139e02, "ice"),
    ],
)
def test_saturation_vapour_pressure_2(t, v_ref, phase, array_backend):
    t, v_ref = array_backend.asarray(t, v_ref)

    svp = thermo.array.saturation_vapour_pressure(t, phase=phase)
    assert array_backend.allclose(svp, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("phase", ["mixed", "water", "ice"])
def test_saturation_mixing_ratio(phase, array_backend):
    ref_file = "sat_mr.csv"

    # t = thermo.array.celsius_to_kelvin(np.linspace(-40.0, 56.0, 25))
    # p = [1000, 950, 850, 700]
    # t_num = len(t)
    # t = np.repeat(t, repeats=len(p))
    # p = np.array(p * t_num) * 100.0
    # o = {"t": t, "p": p}
    # for phase in ["mixed", "water", "ice"]:
    #     o[phase] = thermo.array.saturation_mixing_ratio(t, p, phase=phase)
    # save_test_reference(ref_file, o)

    d = read_data_file(ref_file)
    t = array_backend.asarray(d["t"])
    p = array_backend.asarray(d["p"])
    v_ref = array_backend.asarray(d[phase])

    mr = thermo.array.saturation_mixing_ratio(t, p, phase=phase)
    assert array_backend.allclose(mr, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("phase", ["mixed", "water", "ice"])
def test_saturation_specific_humidity(phase, array_backend):
    ref_file = "sat_q.csv"

    # t = thermo.array.celsius_to_kelvin(np.linspace(-40.0, 56.0, 25))
    # p = [1000, 950, 850, 700]
    # t_num = len(t)
    # t = np.repeat(t, repeats=len(p))
    # p = np.array(p * t_num) * 100.0
    # o = {"t": t, "p": p}
    # for phase in ["mixed", "water", "ice"]:
    #     o[phase] = thermo.array.saturation_specific_humidity(t, p, phase=phase)
    # save_test_reference(ref_file, o)

    d = read_data_file(ref_file)
    t = array_backend.asarray(d["t"])
    p = array_backend.asarray(d["p"])
    v_ref = array_backend.asarray(d[phase])

    q = thermo.array.saturation_specific_humidity(t, p, phase=phase)
    assert array_backend.allclose(q, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("phase", ["mixed", "water", "ice"])
def test_saturation_vapour_pressure_slope(phase, array_backend):
    ref_file = "sat_vp_slope.csv"

    # o = {"t": thermo.array.celsius_to_kelvin(np.linspace(-40.0, 56.0, 49))}
    # for phase in ["mixed", "water", "ice"]:
    #     o[phase] = thermo.array.saturation_vapour_pressure_slope(o["t"], phase=phase)
    # save_test_reference(ref_file, o)

    d = read_data_file(ref_file)
    t = array_backend.asarray(d["t"])
    v_ref = array_backend.asarray(d[phase])

    svp = thermo.array.saturation_vapour_pressure_slope(t, phase=phase)
    assert array_backend.allclose(svp, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("phase", ["mixed", "water", "ice"])
def test_saturation_mixing_ratio_slope_1(phase, array_backend):
    ref_file = "sat_mr_slope.csv"

    # t = thermo.array.celsius_to_kelvin(np.linspace(-40.0, 56.0, 25))
    # p = [1000, 950, 850, 700]
    # t_num = len(t)
    # t = np.repeat(t, repeats=len(p))
    # p = np.array(p * t_num) * 100.0
    # o = {"t": t, "p": p}
    # for phase in ["mixed", "water", "ice"]:
    #     o[phase] = thermo.array.saturation_mixing_ratio_slope(t, p, phase=phase)
    # save_test_reference(ref_file, o)

    d = read_data_file(ref_file)
    t = array_backend.asarray(d["t"])
    p = array_backend.asarray(d["p"])
    v_ref = array_backend.asarray(d[phase])

    svp = thermo.array.saturation_mixing_ratio_slope(t, p, phase=phase)
    assert array_backend.allclose(svp, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "t,p,v_ref",
    [(283.0, 1e5, 0.0005189819), (600.0, 1e5, np.nan), ([200 + 273.16], [100000.0], [np.nan])],
)
def test_saturation_mixing_ratio_slope_numbers(t, p, v_ref, array_backend):
    t, p, v_ref = array_backend.asarray(t, p, v_ref)
    svp = thermo.array.saturation_mixing_ratio_slope(t, p, phase="mixed")
    assert array_backend.allclose(svp, v_ref, equal_nan=True)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("phase", ["mixed", "water", "ice"])
def test_saturation_specific_humidity_slope_1(phase, array_backend):
    ref_file = "sat_q_slope.csv"
    # phases = ["mixed", "water", "ice"]

    # t = thermo.array.celsius_to_kelvin(np.linspace(-40.0, 56.0, 25))
    # p = [1000, 950, 850, 700]
    # t_num = len(t)
    # t = np.repeat(t, repeats=len(p))
    # p = np.array(p * t_num) * 100.0
    # o = {"t": t, "p": p}
    # for phase in ["mixed", "water", "ice"]:
    #     o[phase] = thermo.array.saturation_specific_humidity_slope(t, p, phase=phase)
    # save_test_reference(ref_file, o)

    d = read_data_file(ref_file)
    t = array_backend.asarray(d["t"])
    p = array_backend.asarray(d["p"])
    v_ref = array_backend.asarray(d[phase])

    svp = thermo.array.saturation_specific_humidity_slope(t, p, phase=phase)
    assert array_backend.allclose(svp, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "t,p,v_ref",
    [(283.0, 1e5, 0.0005111349), (600.0, 1e5, np.nan), ([200 + 273.16], [100000.0], [np.nan])],
)
def test_saturation_specific_humidity_slope_number(t, p, v_ref, array_backend):
    t, p, v_ref = array_backend.asarray(t, p, v_ref)
    svp = thermo.array.saturation_specific_humidity_slope(t, p, phase="mixed")
    assert array_backend.allclose(svp, v_ref, equal_nan=True)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
def test_temperature_from_saturation_vapour_pressure_1(array_backend):
    ref_file = "sat_vp.csv"
    d = read_data_file(ref_file)

    svp = array_backend.asarray(d["water"])
    v_ref = array_backend.asarray(d["t"])

    t = thermo.array.temperature_from_saturation_vapour_pressure(svp)
    assert array_backend.allclose(t, v_ref, equal_nan=True)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "es,v_ref",
    [
        (4.2, 219.7796336743947),
        (0, np.nan),
    ],
)
def test_temperature_from_saturation_vapour_pressure_numbers(es, v_ref, array_backend):
    es, v_ref = array_backend.asarray(es, v_ref)
    t = thermo.array.temperature_from_saturation_vapour_pressure(es)
    assert array_backend.allclose(t, v_ref, equal_nan=True)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "t,td,v_ref",
    [
        (
            [20.0, 20, 0, 35, 5, -15, 25],
            [20.0, 10, -10, 32, -15, -24, -3],
            [
                100.0000000000,
                52.5224541378,
                46.8714823296,
                84.5391163313,
                21.9244774232,
                46.1081101229,
                15.4779832381,
            ],  # reference was tested with an online relhum calculator at:
            # https://bmcnoldy.rsmas.miami.edu/Humidity.html
        ),
    ],
)
def test_relative_humidity_from_dewpoint(t, td, v_ref, array_backend):
    t, td, v_ref = array_backend.asarray(t, td, v_ref)
    t = thermo.array.celsius_to_kelvin(t)
    td = thermo.array.celsius_to_kelvin(td)

    r = thermo.array.relative_humidity_from_dewpoint(t, td)
    assert array_backend.allclose(r, v_ref, rtol=1e-05)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "t,p,q,v_ref",
    [
        (
            [-29.2884, -14.4118, -5.9235, 9.72339, 18.4514],
            [300, 400, 500, 700, 850],
            [
                0.000845416891024797,
                0.00277950354211498,
                0.00464489207661245,
                0.0076785187585422,
                0.0114808182580539,
            ],
            [
                99.70488530734642,
                100.25885732613531,
                97.15956159465799,
                71.37937968160273,
                73.41420898756694,
            ],
        ),
    ],
)
def test_relative_humidity_from_specific_humidity(t, p, q, v_ref, array_backend):
    t, p, q, v_ref = array_backend.asarray(t, p, q, v_ref)
    t = thermo.array.celsius_to_kelvin(t)
    p = p * 100.0

    r = thermo.array.relative_humidity_from_specific_humidity(t, q, p)
    assert array_backend.allclose(r, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "td,p,v_ref",
    [
        (
            [21.78907, 19.90885, 16.50236, 7.104064, -0.3548709, -16.37916],
            [967.5085, 936.3775, 872.248, 756.1647, 649.157, 422.4207],
            [
                0.0169461501,
                0.0155840075,
                0.0134912382,
                0.0083409720,
                0.0057268584,
                0.0025150791,
            ],
        )
    ],
)
def test_specific_humidity_from_dewpoint(td, p, v_ref, array_backend):
    td, p, v_ref = array_backend.asarray(td, p, v_ref)
    td = thermo.array.celsius_to_kelvin(td)
    p = p * 100.0

    q = thermo.array.specific_humidity_from_dewpoint(td, p)
    assert array_backend.allclose(q, v_ref, rtol=1e-05)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "t,p,r,v_ref",
    [
        (
            [-29.2884, -14.4118, -5.9235, 9.72339, 18.4514],
            [300, 400, 500, 700, 850],
            [
                99.70488530734642,
                100.25885732613531,
                97.15956159465799,
                71.37937968160273,
                73.41420898756694,
            ],
            [
                0.000845416891024797,
                0.00277950354211498,
                0.00464489207661245,
                0.0076785187585422,
                0.0114808182580539,
            ],
        )
    ],
)
def test_specific_humidity_from_relative_humidity(t, p, r, v_ref, array_backend):
    t, p, r, v_ref = array_backend.asarray(t, p, r, v_ref)
    t = thermo.array.celsius_to_kelvin(t)
    p = p * 100.0

    q = thermo.array.specific_humidity_from_relative_humidity(t, r, p)
    assert array_backend.allclose(q, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "t,r,v_ref",
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
def test_dewpoint_from_relative_humidity(t, r, v_ref, array_backend):
    # reference was tested with an online relhum calculator at:
    # https://bmcnoldy.rsmas.miami.edu/Humidity.html

    t, r, v_ref = array_backend.asarray(t, r, v_ref)
    t = thermo.array.celsius_to_kelvin(t)
    v_ref = thermo.array.celsius_to_kelvin(v_ref)

    td = thermo.array.dewpoint_from_relative_humidity(t, r)
    assert array_backend.allclose(td, v_ref, equal_nan=True)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "q,p,v_ref",
    [
        (
            [0.0169461501, 0.0155840075, 0.0134912382, 0.0083409720, 0.0057268584, 0.0025150791, 0],
            [967.5085, 936.3775, 872.248, 756.1647, 649.157, 422.4207, 422.4207],
            [21.78907, 19.90885, 16.50236, 7.104064, -0.3548709, -16.37916, np.nan],
        ),
        (0.0169461501, 967.508, 21.78907),
    ],
)
def test_dewpoint_from_specific_humidity(q, p, v_ref, array_backend):
    q, p, v_ref = array_backend.asarray(q, p, v_ref)
    p = p * 100.0
    v_ref = thermo.array.celsius_to_kelvin(v_ref)

    td = thermo.array.dewpoint_from_specific_humidity(q, p)
    assert array_backend.allclose(td, v_ref, equal_nan=True)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "t,q,v_ref",
    [([286.4, 293.4], [0.0196078431, 0.0291262136], [289.8130240470, 298.5937453245])],
)
def test_virtual_temperature(t, q, v_ref, array_backend):
    t, q, v_ref = array_backend.asarray(t, q, v_ref)
    tv = thermo.array.virtual_temperature(t, q)
    assert array_backend.allclose(tv, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "t,q,p,v_ref",
    [([286.4, 293.4], [0.0196078431, 0.0291262136], [100300.0, 95000.0], [289.5651110613, 303.0015650834])],
)
def test_virtual_potential_temperature_temperature(t, q, p, v_ref, array_backend):
    t, q, p, v_ref = array_backend.asarray(t, q, p, v_ref)
    tv = thermo.array.virtual_potential_temperature(t, q, p)
    assert array_backend.allclose(tv, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "t,p,v_ref",
    [([252.16, 298.16], [72350, 100500], [276.588026, 297.735455])],
)
def test_potential_temperature(t, p, v_ref, array_backend):
    t, p, v_ref = array_backend.asarray(t, p, v_ref)
    th = thermo.array.potential_temperature(t, p)
    assert array_backend.allclose(th, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "p,th,v_ref",
    [([72350, 100500], [276.588026, 297.735455], [252.16, 298.16])],
)
def test_temperature_from_potential_temperature(p, th, v_ref, array_backend):
    p, th, v_ref = array_backend.asarray(p, th, v_ref)
    t = thermo.array.temperature_from_potential_temperature(th, p)
    assert array_backend.allclose(t, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "t_def,p_def, p, v_ref",
    [
        ([252.16, 298.16], [72350, 100500], [700, 500], [249.792414, 244.246863]),
        (252.16, 72350, [700, 500], [249.792414, 226.898581]),
    ],
)
def test_temperature_on_dry_adibat(t_def, p_def, p, v_ref, array_backend):
    p_def, t_def, p, v_ref = array_backend.asarray(p_def, t_def, p, v_ref)
    p = p * 100.0

    t = thermo.array.temperature_on_dry_adiabat(p, t_def, p_def)
    assert array_backend.allclose(t, v_ref)

    # cross checking
    if not callable(t.size) and t_def.size > 1:
        th1 = thermo.array.potential_temperature(t_def, p_def)
        th2 = thermo.array.potential_temperature(t, p)
        assert array_backend.allclose(th1, th2)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "t_def,p_def,t,v_ref",
    [
        ([252.16, 298.16], [72350, 100500], [249.792414, 244.246863], [70000.0, 50000.0]),
        (252.16, 72350, [249.792414, 244.246863], [70000, 64709.699161]),
    ],
)
def test_pressure_on_dry_adibat(t_def, p_def, t, v_ref, array_backend):
    p_def, t_def, t, v_ref = array_backend.asarray(p_def, t_def, t, v_ref)

    p = thermo.array.pressure_on_dry_adiabat(t, t_def, p_def)
    assert array_backend.allclose(p, v_ref)

    # cross checking
    if not callable(t.size) and t_def.size > 1:
        th1 = thermo.array.potential_temperature(t_def, p_def)
        th2 = thermo.array.potential_temperature(t, p)
        assert array_backend.allclose(th1, th2)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "t,td,p,t_ref, p_ref,,method",
    [
        (
            [10, 30, 43, 20],
            [-2, 26, 5, 28],
            [95000, 100500, 100100, 100200],
            [268.706024, 298.200936, 270.517934, 303.138144],
            [79081.766347, 94862.350635, 57999.83367, 112654.210439],
            "davies",
        ),
        (
            [10, 30, 43, 20],
            [-2, 26, 5, 28],
            [95000, 100500, 100100, 100200],
            [268.683018, 298.182282, 270.531264, 303.199544],
            [79058.068785, 94841.581142, 58009.838027, 112734.100243],
            "bolton",
        ),
    ],
)
def test_lcl(t, td, p, t_ref, p_ref, method, array_backend):
    t, td, p, t_ref, p_ref = array_backend.asarray(t, td, p, t_ref, p_ref)
    t = thermo.array.celsius_to_kelvin(t)
    td = thermo.array.celsius_to_kelvin(td)

    t_lcl = thermo.array.lcl_temperature(t, td, method=method)
    assert array_backend.allclose(t_lcl, t_ref)

    t_lcl, p_lcl = thermo.array.lcl(t, td, p, method=method)
    assert array_backend.allclose(t_lcl, t_ref)
    assert array_backend.allclose(p_lcl, p_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("method", ["ifs", "bolton35", "bolton39"])
def test_ept(method, array_backend):
    data = ThermoInputData(array_backend)

    ref_file = "eqpt.csv"

    # o = {}
    # for m in methods:
    #     o[f"{m}_td"] = thermo.array.ept_from_dewpoint(data.t, data.td, data.p, method=m)
    #     o[f"{m}_q"] = thermo.array.ept_from_specific_humidity(
    #         data.t, data.q, data.p, method=m
    #     )
    # save_test_reference(ref_file, o)

    ref = read_data_file(ref_file)

    pt = thermo.array.ept_from_dewpoint(data.t, data.td, data.p, method=method)
    v_ref = array_backend.asarray(ref[method + "_td"])
    assert array_backend.allclose(pt, v_ref), f"td {method=}"

    pt = thermo.array.ept_from_specific_humidity(data.t, data.q, data.p, method=method)
    v_ref = array_backend.asarray(ref[method + "_q"])
    assert array_backend.allclose(pt, v_ref), f"q {method=}"


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("method", ["ifs", "bolton35", "bolton39"])
def test_saturation_ept(method, array_backend):
    data = ThermoInputData(array_backend)

    ref_file = "seqpt.csv"

    # o = {}
    # for m in methods:
    #     o[m] = thermo.array.saturation_ept(data.t, data.p, method=m)
    # save_test_reference(ref_file, o)

    ref = read_data_file(ref_file)

    pt = thermo.array.saturation_ept(data.t, data.p, method=method)
    v_ref = array_backend.asarray(ref[method])
    assert array_backend.allclose(pt, v_ref), f"{method=}"


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("ept_method", ["ifs", "bolton35", "bolton39"])
@pytest.mark.parametrize("t_method", ["bisect", "newton"])
def test_temperature_on_moist_adiabat(ept_method, t_method, array_backend):
    ref_file = "t_on_most_adiabat.csv"

    # ept = np.array([220, 250, 273.16, 300, 330, 360, 400, 500, 600, 700, 800, 900])
    # p = np.array([1010, 1000, 925, 850, 700, 500, 300]) * 100
    # o = {"ept": np.repeat(ept, repeats=len(p)), "p": np.array(p.tolist() * len(ept))}
    # for m_ept in ept_methods:
    #     for m_t in t_methods:
    #         o[f"{m_ept}_{m_t}"] = thermo.array.temperature_on_moist_adiabat(
    #             o["ept"], o["p"], ept_method=m_ept, t_method=m_t
    #         )
    # save_test_reference(ref_file, o)

    ref = read_data_file(ref_file)
    ept = array_backend.asarray(ref["ept"])
    p = array_backend.asarray(ref["p"])

    pt = thermo.array.temperature_on_moist_adiabat(ept, p, ept_method=ept_method, t_method=t_method)
    v_ref = array_backend.asarray(ref[f"{ept_method}_{t_method}"])
    assert array_backend.allclose(pt, v_ref, equal_nan=True), f"{ept_method=} {t_method=}"


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("ept_method", ["ifs", "bolton35", "bolton39"])
@pytest.mark.parametrize("t_method", ["bisect", "newton"])
def test_wet_bulb_temperature(ept_method, t_method, array_backend):
    data = ThermoInputData(array_backend)

    ref_file = "t_wet.csv"

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

    ref = read_data_file(ref_file)

    pt = thermo.array.wet_bulb_temperature_from_dewpoint(
        data.t, data.td, data.p, ept_method=ept_method, t_method=t_method
    )

    v_ref = array_backend.asarray(ref[f"{ept_method}_{t_method}_td"])
    assert array_backend.allclose(
        pt, v_ref, rtol=1e-03, atol=0, equal_nan=True
    ), f"td {ept_method=} {t_method=}"

    pt = thermo.array.wet_bulb_temperature_from_specific_humidity(
        data.t, data.q, data.p, ept_method=ept_method, t_method=t_method
    )

    v_ref = array_backend.asarray(ref[f"{ept_method}_{t_method}_q"])
    assert array_backend.allclose(
        pt, v_ref, rtol=1e-03, atol=0, equal_nan=True
    ), f"q {ept_method=} {t_method=}"


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("ept_method", ["ifs", "bolton35", "bolton39"])
@pytest.mark.parametrize("t_method", ["bisect", "newton", "direct"])
def test_wet_bulb_potential_temperature(ept_method, t_method, array_backend):
    data = ThermoInputData(array_backend)

    ref_file = "t_wetpt.csv"

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

    ref = read_data_file(ref_file)

    pt = thermo.array.wet_bulb_potential_temperature_from_dewpoint(
        data.t, data.td, data.p, ept_method=ept_method, t_method=t_method
    )

    v_ref = array_backend.asarray(ref[f"{ept_method}_{t_method}_td"])
    assert array_backend.allclose(
        pt, v_ref, rtol=1e-03, atol=0, equal_nan=True
    ), f"td {ept_method=} {t_method=}"

    pt = thermo.array.wet_bulb_potential_temperature_from_specific_humidity(
        data.t, data.q, data.p, ept_method=ept_method, t_method=t_method
    )

    v_ref = array_backend.asarray(ref[f"{ept_method}_{t_method}_q"])
    assert array_backend.allclose(
        pt, v_ref, rtol=1e-03, atol=0, equal_nan=True
    ), f"q {ept_method=} {t_method=}"


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "q,v_ref",
    [
        ([0.0, 0.0196078431, 0.0291262136], [287.0597, 290.4802941111, 292.1407767004]),
        (0.0196078431, 290.4802941111),
    ],
)
def test_specific_gas_constant(q, v_ref, array_backend):
    q, v_ref = array_backend.asarray(q, v_ref)
    r = thermo.array.specific_gas_constant(q)
    assert array_backend.allclose(r, v_ref)
