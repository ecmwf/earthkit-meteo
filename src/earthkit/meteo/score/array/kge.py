# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from earthkit.utils.array import array_namespace

from .correlation import pearson


def kge(x, y, nan_policy="propagate", return_components=False):
    """Compute Kling-Gupta efficiency (KGE).

    This is an implementation of the original Kling-Gupta efficiency (KGE) metric
    as described in [Gupta2009]_.

    Parameters
    ----------
    x: array-like (n_points, n_samples)
        Simulations for n_points points with n_samples samples each
    y: array-like (n_points, n_samples)
        Observations/ references for n_points points with n_samples samples each
    nan_policy: str
        Determines how to handle nans.
        Options are 'raise', 'propagate', or 'omit'.
    return_components: bool
        If True, return KGE, rho, alpha, beta as a stack of arrays with shape (4, n_points)
        If False (default), return only KGE with shape (n_points,)
    Returns
    -------
    array-like (n_points,) or (4, n_points)
        KGE values or KGE values and components (rho, alpha, beta)
        If return_components is True, the returned array has shape (4, n_points)
        If nan_policy is 'omit', n_points is the number of points without nans
    """
    if nan_policy not in {"raise", "propagate", "omit"}:
        raise ValueError("Invalid argument: nan_policy must be 'raise', 'propagate', or 'omit'.")

    xp = array_namespace(x, y)
    x = xp.asarray(x)
    y = xp.asarray(y)

    if x.shape != y.shape:
        raise ValueError(f"Input arrays must have the same shape, got {x.shape} and {y.shape}")

    isnan_mask = xp.any(xp.isnan(x), axis=1) | xp.any(xp.isnan(y), axis=1)

    if nan_policy == "raise" and xp.any(isnan_mask):
        raise ValueError(f"Missing values present in input and nan_policy={nan_policy}")
    elif nan_policy == "omit":
        x = x[~isnan_mask, ...]
        y = y[~isnan_mask, ...]

    rho = pearson(x, y, axis=1)
    alpha = xp.std(x, axis=1) / xp.std(y, axis=1)
    beta = xp.mean(x, axis=1) / xp.mean(y, axis=1)

    kge = 1 - xp.sqrt((rho - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    if nan_policy == "propagate":
        kge = xp.where(isnan_mask, xp.nan, kge)
        rho = xp.where(isnan_mask, xp.nan, rho)
        alpha = xp.where(isnan_mask, xp.nan, alpha)
        beta = xp.where(isnan_mask, xp.nan, beta)

    if return_components:
        components = xp.stack((kge, rho, alpha, beta))
        return components
    else:
        return kge


def kge_prime(x, y, nan_policy="propagate", return_components=False):
    """Compute Modified Kling-Gupta efficiency (KGE').

    This is an implementation of the modified Kling-Gupta efficiency (KGE') metric
    as described in [Kling2012]_.

    Parameters
    ----------
    x: array-like (n_points, n_samples)
        Simulations for n_points points with n_samples samples each
    y: array-like (n_points, n_samples)
        Observations/ references for n_points points with n_samples samples each
    nan_policy: str
        Determines how to handle nans.
        Options are 'raise', 'propagate', or 'omit'.
    return_components: bool
        If True, return KGE', rho, beta, gamma as a stack of arrays with shape (4, n_points)
        If False (default), return only KGE' with shape (n_points,)
    Returns
    -------
    array-like (n_points,) or (4, n_points)
        KGE' values or KGE' values and components (rho, beta, gamma)
        If return_components is True, the returned array has shape (4, n_points)
        If nan_policy is 'omit', n_points is the number of points without nans
    """
    if nan_policy not in {"raise", "propagate", "omit"}:
        raise ValueError("Invalid argument: nan_policy must be 'raise', 'propagate', or 'omit'.")

    xp = array_namespace(x, y)
    x = xp.asarray(x)
    y = xp.asarray(y)

    if x.shape != y.shape:
        raise ValueError(f"Input arrays must have the same shape, got {x.shape} and {y.shape}")

    isnan_mask = xp.any(xp.isnan(x), axis=1) | xp.any(xp.isnan(y), axis=1)

    if nan_policy == "raise" and xp.any(isnan_mask):
        raise ValueError(f"Missing values present in input and nan_policy={nan_policy}")
    elif nan_policy == "omit":
        x = x[~isnan_mask, ...]
        y = y[~isnan_mask, ...]

    rho = pearson(x, y, axis=1)
    beta = xp.mean(x, axis=1) / xp.mean(y, axis=1)
    gamma = (xp.std(x, axis=1) / xp.mean(x, axis=1)) / (xp.std(y, axis=1) / xp.mean(y, axis=1))

    kge = 1 - xp.sqrt((rho - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)

    if nan_policy == "propagate":
        kge = xp.where(isnan_mask, xp.nan, kge)
        rho = xp.where(isnan_mask, xp.nan, rho)
        beta = xp.where(isnan_mask, xp.nan, beta)
        gamma = xp.where(isnan_mask, xp.nan, gamma)

    if return_components:
        components = xp.stack((kge, rho, beta, gamma))
        return components
    else:
        return kge
