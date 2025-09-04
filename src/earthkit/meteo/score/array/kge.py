# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import abc

from earthkit.utils.array import array_namespace

from .correlation import pearson


class _BaseKGE(abc.ABC):
    """Simple base class for KGE scores. Subclasses compute the components."""

    @abc.abstractmethod
    def compute_components(self, xp, x, y) -> list:
        """Compute the KGE components.

        Parameters
        ----------
        xp: array module
            The array module (numpy, dask.array, cupy, ...)
        x: array-like (n_points, n_samples)
            Simulations for n_points points with n_samples samples each
        y: array-like (n_points, n_samples)
            Observations/ references for n_points points with n_samples samples each
        Returns
        -------
        list of array-like
            The KGE components as a list of arrays with shape (n_points,)
        """

    def compute(self, x, y, nan_policy, return_components):
        if nan_policy not in {"raise", "propagate", "omit"}:
            raise ValueError("Invalid argument: nan_policy must be 'raise', 'propagate', or 'omit'.")

        xp = array_namespace(x, y)
        x = xp.asarray(x)
        y = xp.asarray(y)

        if x.shape != y.shape:
            raise ValueError(f"Input arrays must have the same shape, got {x.shape} and {y.shape}")

        if x.ndim != 2 or y.ndim != 2:
            # Support 1D inputs would be a nice improvement but needs intuitive
            # behavior when used with nan_policy="omit".
            raise ValueError(
                "x and y must be 2D arrays with shape (n_points, n_samples). "
                "For a single time series, use x[None, :] and y[None, :]."
            )

        isnan_mask = xp.any(xp.isnan(x), axis=1) | xp.any(xp.isnan(y), axis=1)

        if nan_policy == "raise" and xp.any(isnan_mask):
            raise ValueError(f"Missing values present in input and nan_policy={nan_policy}")
        elif nan_policy == "omit":
            x = x[~isnan_mask, ...]
            y = y[~isnan_mask, ...]

        components = self.compute_components(xp, x, y)

        # KGE: 1 - sqrt(sum((c - 1)^2))
        total = None
        for c in components:
            term = (c - 1) ** 2
            total = term if total is None else total + term
        kge_val = 1 - xp.sqrt(total)

        if nan_policy == "propagate":
            kge_val = xp.where(isnan_mask, xp.nan, kge_val)
            components = [xp.where(isnan_mask, xp.nan, c) for c in components]

        if return_components:
            return xp.stack((kge_val, *components))
        return kge_val


class _KGE(_BaseKGE):
    """Implementation of the original KGE score as described in [Gupta2009]_"""

    def compute_components(self, xp, x, y) -> list:
        rho = pearson(x, y, axis=1)
        alpha = xp.std(x, axis=1) / xp.std(y, axis=1)
        beta = xp.mean(x, axis=1) / xp.mean(y, axis=1)
        return [rho, alpha, beta]


class _KGEPrime(_BaseKGE):
    """Implementation of the modified KGE' score as described in [Kling2012]_"""

    def compute_components(self, xp, x, y) -> list:
        rho = pearson(x, y, axis=1)
        beta = xp.mean(x, axis=1) / xp.mean(y, axis=1)
        gamma = (xp.std(x, axis=1) / xp.mean(x, axis=1)) / (xp.std(y, axis=1) / xp.mean(y, axis=1))
        return [rho, beta, gamma]


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
    return _KGE().compute(x, y, nan_policy, return_components)


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
    return _KGEPrime().compute(x, y, nan_policy, return_components)
