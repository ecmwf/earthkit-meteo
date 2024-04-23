# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from earthkit.meteo import score


def crps_quaver2(x, y):
    """Computes Continuous Ranked Probability Score (CRPS) from Quaver
    Used for testing

    Parameters
    ----------
    x: numpy array (n_ens, n_points)
        Ensemble forecast
    y: numpy array (n_points)
        Observation/analysis

    Returns
    -------
    numpy array (n_points)
        CRPS values

    The method is described in [Hersbach2000]_.
    """

    n_ens = x.shape[0]
    anarr = y
    earr = x
    # ensemble sorted by fieldset axis
    esarr = np.sort(earr, axis=0)
    aa = np.zeros(earr.shape)  # alpha
    aa = np.concatenate((aa, aa[:1, :]))
    bb = aa.copy()  # beta
    with np.errstate(invalid="ignore"):
        lcond = esarr[0, :] > anarr
        aa[0, lcond] = 1.0
        bb[0, :] = np.where(lcond, esarr[0, :] - anarr, 0.0)
        aa[1:-1, :] = np.where(
            esarr[1:, :] <= anarr, esarr[1:, :] - esarr[:-1, :], anarr - esarr[:-1, :]
        )
        aa[1:-1, :][esarr[: n_ens - 1, :] > anarr] = 0.0  # this would be hard in xarray
        bb[1:-1, :] = np.where(
            esarr[:-1, :] > anarr, esarr[1:, :] - esarr[:-1, :], esarr[1:, :] - anarr
        )
        bb[1:-1, :][esarr[1:, :] <= anarr] = 0.0
        lcond = anarr > esarr[-1, :]
        aa[-1, :] = np.where(lcond, anarr - esarr[-1, :], 0.0)
        bb[-1, lcond] = 1.0
    # back to xarrays
    # alpha = xarray.DataArray(aa, dims=e.dims)
    # beta = xarray.DataArray(bb, dims=e.dims)
    # weight = xarray.DataArray(np.arange(n_ens + 1), dims=ENS_DIM) / float(n_ens)
    # w = np.arange(n_ens+1)/float(n_ens)
    w = (np.arange(n_ens + 1) / float(n_ens)).reshape(n_ens + 1, *([1] * y.ndim))
    # w = np.expand_dims(np.arange(n_ens+1)/float(n_ens), axis=1)
    crps = aa * w**2 + bb * (1.0 - w) ** 2
    crps_sum = np.nansum(crps, axis=0)
    return crps_sum
    # Fair CRPS
    # fcrps = crps - self.ginis_mean_diff() / (2. * n_ens)
    # return alpha, beta, crps, fcrps


def test_crps():
    obs = np.array(
        [
            [
                18.18332,
                16.91703,
                18.41237,
                18.52762,
                19.24018,
                18.92083,
                18.7347,
                18.74177,
                18.84006,
                18.55334,
                17.13083,
                18.84862,
                17.76169,
                18.39295,
                18.69872,
                18.83633,
                20.04871,
                18.70762,
                19.01472,
                19.28151,
                19.58305,
                18.82714,
                17.98188,
                19.38522,
                19.19444,
                20.05244,
                18.75,
            ]
        ]
    )
    ens = np.array(
        [
            [
                18.60203,
                18.39837,
                18.15966,
                18.2023,
                18.33673,
                18.27596,
                18.16505,
                18.18332,
                18.0794,
                18.45462,
                18.511,
                18.30488,
                18.24115,
                18.96865,
                18.35899,
                18.47862,
                18.15919,
                18.33152,
                18.49652,
                18.57579,
                18.41748,
                18.54667,
                18.75921,
                18.6189,
            ],
            [
                17.82097,
                18.14485,
                18.60321,
                17.84873,
                18.45063,
                18.46943,
                18.21171,
                18.01587,
                18.37726,
                18.37997,
                18.25159,
                18.74551,
                18.49753,
                18.30485,
                18.46268,
                18.38053,
                18.7921,
                18.60726,
                18.37832,
                18.63638,
                18.4735,
                18.46691,
                18.40706,
                18.46787,
            ],
            [
                18.25722,
                18.31637,
                17.94259,
                18.08479,
                17.98738,
                17.84325,
                18.11392,
                17.5627,
                18.11146,
                18.66067,
                18.29907,
                18.52028,
                17.97144,
                18.00114,
                17.90269,
                18.42152,
                18.20297,
                18.23269,
                18.26792,
                18.17865,
                18.63547,
                18.18611,
                18.28587,
                18.34955,
            ],
            [
                18.46637,
                18.28559,
                18.61714,
                18.54838,
                18.47381,
                18.58475,
                18.32031,
                18.39696,
                18.40214,
                18.68081,
                18.37716,
                18.30279,
                18.51266,
                18.46895,
                18.69215,
                18.0772,
                18.55367,
                18.33005,
                18.16397,
                18.32986,
                18.69788,
                18.58536,
                18.717,
                18.68816,
            ],
            [
                18.33082,
                18.08299,
                18.28411,
                18.47392,
                18.1853,
                18.70116,
                18.3585,
                18.4014,
                18.50559,
                18.13646,
                18.88681,
                18.22201,
                18.3747,
                18.53531,
                18.33449,
                18.52871,
                18.5113,
                18.46233,
                18.68657,
                18.41774,
                18.35555,
                18.37039,
                18.47539,
                18.3886,
            ],
            [
                18.62395,
                18.23428,
                18.782,
                18.27582,
                18.49167,
                18.27856,
                18.30689,
                18.67729,
                18.15076,
                18.35025,
                18.59495,
                18.65966,
                18.33589,
                18.82887,
                18.20526,
                18.06337,
                18.30195,
                18.94566,
                18.67036,
                18.5322,
                18.79806,
                18.54978,
                18.49529,
                18.91547,
            ],
            [
                18.66162,
                18.41238,
                18.99822,
                18.88024,
                18.56961,
                18.20587,
                18.28942,
                18.71072,
                18.71503,
                18.96295,
                18.48127,
                18.80717,
                18.60179,
                18.77288,
                19.0915,
                18.47368,
                18.62611,
                19.18506,
                18.66757,
                18.54969,
                18.48033,
                18.55008,
                18.76138,
                18.57253,
            ],
            [
                19.0613,
                19.22018,
                19.28419,
                19.1977,
                18.69967,
                18.81336,
                19.29298,
                18.8783,
                19.01464,
                19.29025,
                19.21277,
                18.85543,
                19.04082,
                19.39295,
                18.93398,
                19.02341,
                18.74652,
                19.08729,
                19.47899,
                19.01433,
                19.15233,
                19.51727,
                19.1944,
                19.07091,
            ],
            [
                18.76119,
                18.76257,
                18.77719,
                18.59709,
                18.70659,
                18.19559,
                18.46485,
                19.08298,
                18.87243,
                18.70967,
                18.70817,
                18.71143,
                19.28904,
                18.88789,
                19.00604,
                18.80304,
                18.62888,
                18.84454,
                19.04174,
                18.77071,
                18.76931,
                18.78357,
                18.90928,
                18.48198,
            ],
            [
                18.44427,
                18.80132,
                18.60749,
                18.54743,
                18.76166,
                18.97299,
                18.71958,
                18.6299,
                18.51747,
                18.56865,
                18.831,
                18.87236,
                18.52923,
                18.81183,
                18.69399,
                18.61704,
                18.84944,
                18.48689,
                18.62681,
                18.77607,
                18.64201,
                18.8559,
                18.57702,
                18.46109,
            ],
            [
                18.67784,
                18.42476,
                18.52205,
                18.41963,
                18.32471,
                18.24997,
                18.60797,
                18.54053,
                18.8739,
                18.3279,
                18.03648,
                18.65794,
                18.942,
                18.59777,
                18.63692,
                18.53363,
                19.00416,
                18.58927,
                18.47256,
                18.85754,
                18.55462,
                18.7629,
                18.49969,
                18.55102,
            ],
            [
                18.6331,
                18.71255,
                18.17562,
                18.73818,
                18.96507,
                18.48447,
                18.62908,
                18.73402,
                18.99546,
                18.34555,
                18.64311,
                18.83355,
                18.61156,
                18.92137,
                18.65269,
                18.89733,
                19.15071,
                18.99898,
                18.68672,
                18.60682,
                18.90617,
                18.77038,
                18.27683,
                18.63486,
            ],
            [
                18.76511,
                18.93057,
                18.95042,
                18.84819,
                18.48806,
                19.29665,
                19.13163,
                19.14136,
                18.57607,
                19.05337,
                19.0143,
                19.00267,
                18.90464,
                18.82456,
                18.74383,
                18.87646,
                19.05792,
                19.25116,
                19.20859,
                18.75791,
                18.95366,
                18.74861,
                18.97499,
                18.89206,
            ],
            [
                18.71654,
                18.42542,
                18.70209,
                18.67078,
                18.64717,
                18.34367,
                18.55692,
                18.44075,
                18.4575,
                18.13324,
                18.84722,
                18.28914,
                18.69548,
                18.79686,
                18.63756,
                18.894,
                18.35953,
                18.63873,
                18.83494,
                18.64428,
                18.18089,
                18.4196,
                18.93419,
                18.58388,
            ],
            [
                18.5964,
                18.14932,
                18.75467,
                18.40088,
                18.74566,
                18.79426,
                18.8266,
                18.37112,
                18.36424,
                18.68784,
                18.81087,
                18.45764,
                18.77412,
                18.54716,
                18.63115,
                18.95348,
                18.61597,
                18.53023,
                18.53192,
                18.46523,
                18.41556,
                18.72332,
                18.86127,
                18.53786,
            ],
            [
                18.8482,
                18.43965,
                18.56102,
                18.94522,
                18.94733,
                18.76182,
                18.82269,
                18.68657,
                18.92965,
                18.54021,
                18.62588,
                19.24503,
                18.92584,
                18.7524,
                18.83618,
                18.934,
                18.58884,
                19.05073,
                18.74503,
                18.82212,
                18.98872,
                18.80763,
                19.23685,
                18.80635,
            ],
            [
                18.62136,
                18.63467,
                19.10885,
                18.7396,
                18.99339,
                18.81446,
                18.61559,
                18.85497,
                19.06342,
                18.68385,
                18.89761,
                19.13183,
                18.97966,
                18.58725,
                18.87551,
                18.7846,
                19.12682,
                18.78679,
                19.26764,
                19.46835,
                19.07306,
                18.86632,
                19.19791,
                18.93126,
            ],
            [
                18.80622,
                18.83819,
                18.94128,
                19.40541,
                18.87043,
                19.18884,
                18.98556,
                18.82235,
                18.83538,
                18.50262,
                18.48397,
                19.11869,
                18.86834,
                18.96655,
                18.74451,
                19.18596,
                18.63508,
                19.05358,
                18.56128,
                18.9354,
                19.13598,
                19.11241,
                18.70568,
                19.09391,
            ],
            [
                18.65022,
                18.96196,
                19.22926,
                18.40618,
                18.80822,
                18.95773,
                18.88688,
                18.58439,
                19.49417,
                19.09168,
                18.78202,
                18.82866,
                19.27915,
                18.46376,
                19.23724,
                19.1,
                19.41821,
                19.14601,
                19.27609,
                19.14734,
                18.9948,
                18.98772,
                18.76538,
                19.60412,
            ],
            [
                18.89859,
                18.88165,
                19.00075,
                18.63107,
                18.89169,
                18.88976,
                18.47058,
                18.53601,
                18.98374,
                18.9582,
                19.21795,
                18.9789,
                18.77789,
                19.00836,
                19.13202,
                18.80045,
                19.01323,
                19.35668,
                18.79275,
                19.08904,
                18.58958,
                18.82234,
                19.05752,
                18.79315,
            ],
            [
                18.77299,
                18.68177,
                18.78737,
                18.42972,
                18.46792,
                18.89753,
                18.79876,
                18.60094,
                18.89985,
                18.98094,
                18.95772,
                18.9883,
                18.84575,
                18.76947,
                18.98135,
                19.01894,
                19.44606,
                18.89475,
                19.15412,
                19.20748,
                19.17444,
                19.32477,
                18.86843,
                19.2845,
            ],
            [
                18.86211,
                18.81549,
                18.96763,
                19.0877,
                19.13518,
                18.74888,
                19.04076,
                18.85203,
                18.80738,
                19.47674,
                18.97152,
                19.30402,
                19.17206,
                18.87924,
                18.93817,
                18.84938,
                19.16146,
                18.42349,
                19.14038,
                19.11972,
                18.9862,
                19.01817,
                18.77669,
                18.95876,
            ],
            [
                19.15834,
                18.85331,
                19.00297,
                19.04329,
                19.01327,
                18.69482,
                19.22047,
                18.94505,
                19.09807,
                19.51356,
                18.99829,
                19.15129,
                18.8257,
                18.99315,
                19.18298,
                19.3982,
                18.96748,
                18.94291,
                19.15574,
                19.06957,
                18.67271,
                19.23746,
                19.2181,
                19.26443,
            ],
            [
                18.96979,
                18.69425,
                19.26467,
                19.31992,
                18.99275,
                18.94634,
                19.30543,
                18.72837,
                19.32144,
                19.07595,
                19.0096,
                19.27506,
                19.04715,
                19.17561,
                19.38555,
                19.40679,
                19.21241,
                19.41876,
                19.2357,
                19.27149,
                19.37533,
                19.21089,
                19.30751,
                19.35292,
            ],
            [
                18.84174,
                19.03868,
                19.03287,
                18.97804,
                18.84218,
                19.42819,
                19.23164,
                19.17193,
                19.36265,
                19.07079,
                19.6104,
                19.31643,
                19.23143,
                19.36369,
                19.00115,
                18.96971,
                18.93869,
                19.19646,
                18.91059,
                18.76732,
                19.16999,
                19.43966,
                19.32732,
                19.27241,
            ],
            [
                19.62295,
                19.00364,
                18.9897,
                19.04174,
                19.02866,
                19.57506,
                19.33182,
                19.07238,
                19.41094,
                19.35779,
                19.2587,
                19.17919,
                19.472,
                19.59344,
                19.32774,
                19.08439,
                19.1335,
                19.07593,
                19.82984,
                19.46689,
                19.42777,
                19.27194,
                19.10563,
                19.13054,
            ],
            [
                18.96093,
                18.9821,
                19.13877,
                18.98479,
                19.21401,
                19.10517,
                19.43129,
                19.21763,
                19.51593,
                19.10233,
                19.07561,
                19.32022,
                18.76391,
                19.2432,
                19.13902,
                19.08999,
                19.28826,
                19.23503,
                18.99935,
                19.50145,
                19.09675,
                19.17403,
                19.27126,
                18.91046,
            ],
        ]
    )
    ens_tr = ens.transpose()

    goal = [
        0.11623786458333449,
        1.3377413194444439,
        0.14396145833333263,
        0.05145284722222212,
        0.72659345486110970,
        0.27901552083333270,
        0.067909236111111190,
        0.24382366319444510,
        0.05285151041666695,
        0.06693468750000071,
        1.3187623263888897,
        0.08775225694444466,
        1.06167119791666500,
        0.11707013888888933,
        0.06489800347222260,
        0.041414305555555560,
        1.001697135416666,
        0.12593750000000084,
        0.07581557291666664,
        0.2746188541666673,
        0.51726421875000020,
        0.08657913194444464,
        0.97686340277777720,
        0.10266149305555577,
        0.062085798611111194,
        0.6430974652777783,
        0.30735505208333325,
    ]

    crps_mm = score.crps(ens_tr, obs[0])
    crps_qu = crps_quaver2(ens_tr, obs[0])

    for j in range(ens.shape[0]):
        assert np.allclose(crps_mm[j], goal[j])
        assert np.allclose(crps_qu[j], goal[j])

    mean_goal = 0.36859501543209877
    print(crps_mm.mean())
    assert np.allclose(crps_mm.mean(), mean_goal)
    assert np.allclose(crps_qu.mean(), mean_goal)
