# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np

from earthkit.meteo import geo

# np.set_printoptions(formatter={"float_kind": "{:.10f}".format})


def test_distance():
    # single values
    lat_ref = 0
    lon_ref = 0

    lats = np.array([0.0, 0, 0, 0, 90, -90, 48, 48, -48, -48, np.nan])
    lons = np.array([0, 90, -90, 180, 0, 0, 20, -20, -20, 20, 1.0])
    v_ref = np.array(
        [
            0.0000000000,
            10007903.1103691217,
            10007903.1103691217,
            20015806.2207382433,
            10007903.1103691217,
            10007903.1103691217,
            5675597.9227914885,
            5675597.9227914885,
            5675597.9227914885,
            5675597.9227914885,
            np.nan,
        ]
    )
    ds = geo.distance(lat_ref, lon_ref, lats, lons)
    np.testing.assert_allclose(ds, v_ref)

    lat_ref = 90.0
    lon_ref = 0
    v_ref = np.array(
        [
            10007903.1103691217,
            10007903.1103691217,
            10007903.1103691217,
            10007903.1103691217,
            0.0000000000,
            20015806.2207382433,
            4670354.7848389242,
            4670354.7848389242,
            15345451.4358993191,
            15345451.4358993191,
            np.nan,
        ]
    )
    ds = geo.distance(lat_ref, lon_ref, lats, lons)
    np.testing.assert_allclose(ds, v_ref)

    lat_ref = -90.0
    lon_ref = 0
    v_ref = np.array(
        [
            10007903.1103691217,
            10007903.1103691217,
            10007903.1103691217,
            10007903.1103691217,
            20015806.2207382433,
            0.0000000000,
            15345451.4358993191,
            15345451.4358993191,
            4670354.7848389242,
            4670354.7848389242,
            np.nan,
        ]
    )
    ds = geo.distance(lat_ref, lon_ref, lats, lons)
    np.testing.assert_allclose(ds, v_ref)

    lat_ref = 48.0
    lon_ref = 20
    v_ref = np.array(
        [
            5675597.9227914885,
            8536770.5279479641,
            11479035.6927902810,
            14340208.2979467567,
            4670354.7848389242,
            15345451.4358993191,
            0.0000000000,
            2942265.1648423159,
            11351195.8455829788,
            10675096.6510603968,
            np.nan,
        ]
    )
    ds = geo.distance(lat_ref, lon_ref, lats, lons)
    np.testing.assert_allclose(ds, v_ref)

    # arrays
    lat_ref = np.array([48.0, 90.0])
    lon_ref = np.array([20, 0.0])
    v_ref = np.array([5675597.9227914885, 10007903.1103691217])


def test_bearing():
    # single ref - multiple other points
    lat_ref = 46
    lon_ref = 20
    lat = np.array([50.0, 46, 40, 46, 46, 46, -40, 80, 46, 50, 50, 40, 40, np.nan])
    lon = np.array([20.0, 24, 20, 16, -80, 100, 20, 20, 20, 22, 18, 18, 22, 22])
    v_ref = np.array(
        [
            0.0,
            90,
            180,
            270,
            270,
            90,
            180,
            0,
            np.nan,
            19.0929472486,
            340.9070527514,
            193.0983348229,
            166.9016651771,
            np.nan,
        ]
    )

    b = geo.bearing(lat_ref, lon_ref, lat, lon)
    np.testing.assert_allclose(b, v_ref)

    # multiple ref - multiple other points
    lat_ref = [46, -14]
    lon_ref = [20, -78]
    lat = np.array([50.0, 46])
    lon = np.array([22.0, 24])
    v_ref = np.array([19.0929472486, 55.0620059697])
    b = geo.bearing(lat_ref, lon_ref, lat, lon)
    np.testing.assert_allclose(b, v_ref)

    # multiple ref - single other points
    lat_ref = [46, -14]
    lon_ref = [20, -78]
    lat = np.array([50.0])
    lon = np.array([22.0])
    v_ref = np.array([19.0929472486, 53.1446672968])
    b = geo.bearing(lat_ref, lon_ref, lat, lon)
    np.testing.assert_allclose(b, v_ref)

    lat_ref = [46, -14]
    lon_ref = [20, -78]
    lat = np.array([46.0])
    lon = np.array([35.0])
    v_ref = np.array([90.0000000000, 54.7036374509])
    b = geo.bearing(lat_ref, lon_ref, lat, lon)
    np.testing.assert_allclose(b, v_ref)

    # non-matching shapes
    lat_ref = [46, -14]
    lon_ref = [20, -78]
    lat = np.array([46.0, 12, 12])
    lon = np.array([35.0, 22, 22])
    try:
        b = geo.bearing(lat_ref, lon_ref, lat, lon)
        assert False
    except ValueError:
        pass
