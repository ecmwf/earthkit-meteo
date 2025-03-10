# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from earthkit.meteo.utils.array import array_namespace


def crps(x, y):
    """Compute Continuous Ranked Probability Score (CRPS).

    Parameters
    ----------
    x: array-like (n_ens, n_points)
        Ensemble forecast
    y: array-like (n_points)
        Observation/analysis

    Returns
    -------
    array-like (n_points)
        CRPS values


    The method is described in [Hersbach2000]_.
    """
    xp = array_namespace(x, y)
    x = xp.asarray(x)
    y = xp.asarray(y)

    # first sort ensemble
    x = xp.sort(x, axis=0)

    # construct alpha and beta, size nens+1
    n_ens = x.shape[0]
    shape = (n_ens + 1,) + x.shape[1:]
    alpha = xp.zeros(shape)
    beta = xp.zeros(shape)

    # x[i+1]-x[i] and x[i]-y[i] arrays
    diffxy = x - xp.reshape(y, (1, *(y.shape)))
    diffxx = x[1:] - x[:-1]  # x[i+1]-x[i], size ens-1

    # if i == 0
    zero = xp.asarray(0)

    alpha[0] = 0
    beta[0] = xp.fmax(diffxy[0], zero)  # x(0)-y
    # if i == n_ens
    alpha[-1] = xp.fmax(-diffxy[-1], zero)  # y-x(n)
    beta[-1] = 0
    # else
    alpha[1:-1] = xp.fmin(diffxx, xp.fmax(-diffxy[:-1], zero))  # x(i+1)-x(i) or y-x(i) or 0
    beta[1:-1] = xp.fmin(diffxx, xp.fmax(diffxy[1:], zero))  # 0 or x(i+1)-y or x(i+1)-x(i)

    # compute crps
    p_exp = xp.reshape(xp.arange(n_ens + 1) / float(n_ens), (n_ens + 1, *([1] * y.ndim)))
    crps = xp.sum(alpha * (p_exp**2) + beta * ((1 - p_exp) ** 2), axis=0)

    return crps
