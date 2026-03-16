import numpy as np
import pytest
from earthkit.meteo.vertical.array import cape_cin as vertical
from earthkit.meteo import thermo

# Expected values are calculated based on current commit
REFERENCE_CASES = {
    "stable": {
        "p": np.array([  50.        ,  100.        ,  150.        ,  200.        ,
        250.        ,  300.        ,  400.        ,  500.        ,
        600.        ,  700.        ,  850.        ,  925.        ,
       1000.        , 1024.77728271]),
        "t": np.array([225.74617004, 225.61347961, 227.22886658, 227.09010315,
       226.18617249, 223.35209656, 225.96180725, 235.34315491,
       243.71420288, 250.77044678, 257.38471985, 256.66688538,
       255.0138855 , 256.38453674]),
        "zh": np.array([ 2.04774911e+04,  1.59099897e+04,  1.32232798e+04,  1.13071802e+04,
        9.82837671e+03,  8.62787976e+03,  6.74741736e+03,  5.24377783e+03,
        3.96499622e+03,  2.84909973e+03,  1.40258386e+03,  7.65287432e+02,
        1.83113159e+02, -2.58148193e-01]),
        "r": np.array([3.08937478e-06, 2.76289338e-06, 2.70137755e-06, 2.92586029e-06,
       5.03272428e-06, 1.51696122e-05, 8.07112028e-05, 2.11879791e-04,
       3.81853902e-04, 6.51784104e-04, 9.72739431e-04, 8.57611240e-04,
       8.54745316e-04, 8.21906027e-04]),
        "expected": {
            "surface": {"cape": 0.0, "cin": 0.0},
            "mixed":   {"cape": 0.0, "cin": 0.0},
            "mu":      {"cape": 0.0, "cin": 0.0},
        },
    },
    "elevated_instability": {
        "p": np.array([  50.        ,  100.        ,  150.        ,  200.        ,
        250.        ,  300.        ,  400.        ,  500.        ,
        600.        ,  700.        ,  850.        ,  925.        ,
       1000.        ,  890.47729492]),
        "t": np.array([214.33650208, 216.44648743, 220.21812439, 215.66041565,
       215.93861389, 225.26615906, 241.76356506, 254.44471741,
       263.93099976, 272.4520874 , 286.62495422, 292.2938385 ,
       296.6076355 , 292.31422424]),
        "zh": np.array([20616.81323242, 16251.12133789, 13664.76806641, 11819.00488281,
       10421.24108887,  9244.90319824,  7280.13903809,  5659.62902832,
        4276.74450684,  3064.92541504,  1473.99060059,   756.54634285,
          93.43426514,  1080.06802368]),
        "r": np.array([2.83276479e-06, 2.76068147e-06, 3.07064908e-06, 9.38000516e-06,
       3.41134787e-05, 7.95040154e-05, 1.79799931e-04, 5.15130144e-04,
       1.72937450e-03, 3.80528068e-03, 5.27286602e-03, 5.77523488e-03,
       5.88522516e-03, 6.87201680e-03]),
        "expected": {
            "surface": {"cape": 0.0, "cin": 0.0},
            "mixed":   {"cape": 0.0, "cin": 0.0},
            "mu":      {"cape": 978.01972316, "cin": 0.0},
        },
    },
    "unstable": {
        "p": np.array([  50.        ,  100.        ,  150.        ,  200.        ,
        250.        ,  300.        ,  400.        ,  500.        ,
        600.        ,  700.        ,  850.        ,  925.        ,
       1000.        ,  988.72729492]),
        "t": np.array([217.64167786, 218.91816711, 222.0643158 , 219.50270081,
       215.67152405, 223.68998718, 239.55848694, 251.81581116,
       261.70443726, 269.39544678, 280.1933136 , 285.49305725,
       290.56271362, 290.83180237]),
        "zh": np.array([20608.42895508, 16186.18762207, 13575.31860352, 11710.39123535,
       10297.28588867,  9127.82873535,  7177.46325684,  5570.86755371,
        4199.88415527,  3000.80712891,  1436.42687988,   734.22218513,
          74.37207031,   170.89370728]),
        "r": np.array([2.90162485e-06, 2.74880704e-06, 2.85644362e-06, 6.15571631e-06,
       3.57601721e-05, 9.09947695e-05, 3.37356851e-04, 7.66984337e-04,
       1.59063225e-03, 3.02751741e-03, 6.01454274e-03, 7.35454821e-03,
       8.30190131e-03, 8.65592702e-03]),
        "expected": {
            "surface": {"cape": 235.88565888, "cin": 41.60505289},
            "mixed":   {"cape": 329.37140795, "cin": 12.15479423},
            "mu":      {"cape": 645.43837633, "cin": -0.0},
        },
    },
    "large_cape_small_cin": {
        "p": np.array([  50.        ,  100.        ,  150.        ,  200.        ,
        250.        ,  300.        ,  400.        ,  500.        ,
        600.        ,  700.        ,  850.        ,  925.        ,
       1000.        , 1014.50726318]),
        "t": np.array([211.94538879, 206.74043274, 207.76548767, 212.93922424,
       225.02601624, 234.93022156, 250.23524475, 261.38026428,
       271.58529663, 277.85443115, 287.0917511 , 290.94813538,
       293.92599487, 294.93336487]),
        "zh": np.array([20610.54406738, 16410.34655762, 13948.76403809, 12191.68579102,
       10762.78027344,  9534.70947266,  7492.44787598,  5822.09082031,
        4397.38671875,  3154.73168945,  1542.97631836,   823.6462574 ,
         151.02850342,    25.63345337]),
        "r": np.array([2.79327067e-06, 2.85672465e-06, 6.00294481e-06, 3.06504016e-05,
       1.13473873e-04, 2.72005257e-04, 7.75401559e-04, 1.64911460e-03,
       3.58359133e-03, 6.40824917e-03, 8.17005639e-03, 1.22880057e-02,
       1.49398775e-02, 1.52478107e-02]),
        "expected": {
            "surface": {"cape": 927.52859354, "cin": 2.58203251},
            "mixed":   {"cape": 862.33499892, "cin": 2.51359153},
            "mu":      {"cape": 927.52859354, "cin": 2.58203251},
        },
    },
}


@pytest.mark.parametrize("case_data", list(REFERENCE_CASES.values()), ids=list(REFERENCE_CASES.keys()))
def test_cape_cin_surface(case_data):

    p = case_data["p"][:, None]*100
    t = case_data["t"][:, None]
    r = case_data["r"][:, None]
    zh = case_data["zh"][:, None]

    cape_type = "surface"
    expected_values = case_data["expected"][cape_type]
    cape, cin = vertical.cape_cin(p, zh, t, r, cape_type)
    expected_cape = expected_values["cape"]
    expected_cin = expected_values["cin"]
    assert np.isclose(cape, expected_cape, atol=1), f" type '{cape_type}': Expected CAPE {expected_cape}, got {cape}"
    assert np.isclose(cin, expected_cin, atol=1), f" type '{cape_type}': Expected CIN {expected_cin}, got {cin}"


@pytest.mark.parametrize("case_data", list(REFERENCE_CASES.values()), ids=list(REFERENCE_CASES.keys()))
def test_cape_cin_mixed(case_data):

    p = case_data["p"][:, None]*100
    t = case_data["t"][:, None]
    r = case_data["r"][:, None]
    zh = case_data["zh"][:, None]

    cape_type = "mixed"
    expected_values = case_data["expected"][cape_type]
    cape, cin = vertical.cape_cin(p, zh, t, r, cape_type)
    expected_cape = expected_values["cape"]
    expected_cin = expected_values["cin"]
    assert np.isclose(cape, expected_cape, atol=1), f" type '{cape_type}': Expected CAPE {expected_cape}, got {cape}"
    assert np.isclose(cin, expected_cin, atol=1), f" type '{cape_type}': Expected CIN {expected_cin}, got {cin}"


@pytest.mark.parametrize("case_data", list(REFERENCE_CASES.values()), ids=list(REFERENCE_CASES.keys()))
def test_cape_cin_mu(case_data):

    p = case_data["p"][:, None]*100
    t = case_data["t"][:, None]
    r = case_data["r"][:, None]
    zh = case_data["zh"][:, None]

    cape_type = "mu"
    expected_values = case_data["expected"][cape_type]
    cape, cin = vertical.cape_cin(p, zh, t, r, cape_type)
    expected_cape = expected_values["cape"]
    expected_cin = expected_values["cin"]
    assert np.isclose(cape, expected_cape, atol=1), f" type '{cape_type}': Expected CAPE {expected_cape}, got {cape}"
    assert np.isclose(cin, expected_cin, atol=1), f" type '{cape_type}': Expected CIN {expected_cin}, got {cin}"


@pytest.fixture()
def reference_cases_stacked():
    reference_cases = list(REFERENCE_CASES.values())
    p = np.stack([case["p"]*100 for case in reference_cases], axis=-1)
    t = np.stack([case["t"] for case in reference_cases], axis=-1)
    r = np.stack([case["r"] for case in reference_cases], axis=-1)
    zh = np.stack([case["zh"] for case in reference_cases], axis=-1)

    expected_values = {"cape": {}, "cin": {}}
    for cape_type in ["surface", "mixed", "mu"]:
        expected_values["cape"][cape_type] = np.array([case["expected"][cape_type]["cape"] for case in reference_cases])
        expected_values["cin"][cape_type] = np.array([case["expected"][cape_type]["cin"] for case in reference_cases])

    return p, zh, t, r, expected_values


def test_cape_cin_stacked(reference_cases_stacked):
    p, zh, t, r, expected_values = reference_cases_stacked


    for cape_type in ["surface", "mixed", "mu"]:
        cape, cin = vertical.cape_cin(p, zh, t, r, cape_type)
        expected_cape = expected_values["cape"][cape_type]
        expected_cin = expected_values["cin"][cape_type]
        assert (np.isclose(cape, expected_cape, atol=1)).all()
        assert (np.isclose(cin, expected_cin, atol=1)).all()


def test_cape_cin_vertical_axis_minus_1(reference_cases_stacked):
    p, zh, t, r, expected_values = reference_cases_stacked

    # Move vertical axis to the end
    p = np.transpose(p, (1, 0))
    t = np.transpose(t, (1, 0))
    r = np.transpose(r, (1, 0))
    zh = np.transpose(zh, (1, 0))

    for cape_type in ["surface", "mixed", "mu"]:
        cape, cin = vertical.cape_cin(p, zh, t, r, cape_type, vertical_axis=-1)

        expected_cape = expected_values["cape"][cape_type]
        expected_cin = expected_values["cin"][cape_type]

        assert np.isclose(cape, expected_cape, atol=1).all()
        assert np.isclose(cin, expected_cin, atol=1).all()


def test_cape_cin_arbitrary_nd_shape():
    nz = 3
    horiz_shape = (2, 3, 4)

    shape = (nz,) + horiz_shape

    p = np.broadcast_to(np.array([1000, 900, 800])[:, None, None, None], shape)
    t = np.broadcast_to(np.array([290, 280, 270])[:, None, None, None], shape)
    r = np.broadcast_to(np.array([0.01, 0.02, 0.03])[:, None, None, None], shape)
    zh = np.broadcast_to(np.array([100, 200, 300])[:, None, None, None], shape)

    for cape_type in ["surface", "mixed", "mu"]:
        cape, cin = vertical.cape_cin(p, zh, t, r, cape_type)

        assert cape.shape == horiz_shape
        assert cin.shape == horiz_shape


def test_cape_cin_lat_lon(reference_cases_stacked):
    p, zh, t, r, expected_values = reference_cases_stacked

    ny, nx = 2, 2
    nz = p.shape[0]
    p = p.reshape(nz, ny, nx)
    t = t.reshape(nz, ny, nx)
    r = r.reshape(nz, ny, nx)
    zh = zh.reshape(nz, ny, nx)

    for cape_type in ["surface", "mixed", "mu"]:
        cape, cin = vertical.cape_cin(p, zh, t, r, cape_type)

        expected_cape = expected_values["cape"][cape_type].reshape(ny, nx)
        expected_cin = expected_values["cin"][cape_type].reshape(ny, nx)

        assert np.isclose(cape, expected_cape, atol=1).all()
        assert np.isclose(cin, expected_cin, atol=1).all()


def test_cape_cin_missing_values(reference_cases_stacked):
    p, zh, t, r, _ = reference_cases_stacked

    # Introduce NaNs in the input
    t[0, 0] = np.nan
    r[3, 2] = np.nan

    for cape_type in ["surface", "mixed", "mu"]:
        cape, cin = vertical.cape_cin(p, zh, t, r, cape_type)

        assert np.isnan(cape[0])
        assert np.isnan(cin[0])
        assert np.isnan(cape[2])
        assert np.isnan(cin[2])