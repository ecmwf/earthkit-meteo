"""
error
mean_error
abs_error
mean_abs_error
squared_error
mean_squared_error
root_mean_squared_error
standard_deviation_of_error
correlation
kge
"""

from typing import Literal
from typing import TypeVar

import xarray as xr

T = TypeVar("T", xr.DataArray, xr.Dataset)


def _import_scores_or_prompt_install():
    try:
        import scores
    except ImportError:
        raise ImportError(
            "The 'earthkit-meteo[score]' extra is required to use scoring functions. "
            "Please install it using 'pip install earthkit-meteo[score]'"
        )
    return scores


def error(
    fcst: T,
    obs: T,
    agg_method: Literal["mean"] | None = None,
    agg_dim: str | list[str] | None = None,
    agg_weights: xr.DataArray | None = None,
) -> T:
    r"""
    Calculates the error between a forecast and observations.

    The error is defined as:

    .. math::

        e_i = f_i - o_i

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`e_i` is the error.

    Parameters
    ----------
    fcst : xarray object
        The forecast xarray.
    obs : xarray object
        The observations xarray.
    agg_method : str, optional
        The aggregation method to apply over `agg_dim`. Default is None, which means no aggregation.
    agg_dim : str or list of str, optional
        The dimension(s) over which to aggregate. Default is None.
    agg_weights : xarray object, optional
        Weights to apply during aggregation. Default is None.

    Returns
    -------
    xarray object
        The error between the forecast and observations, possibly aggregated.
    """
    assert agg_method in (None, "mean")
    scores = _import_scores_or_prompt_install()

    # TODO: Add comment explaining behavior here in scores
    reduce_dim = agg_dim or []

    return scores.continuous.additive_bias(
        fcst,
        obs,
        reduce_dims=reduce_dim,
        weights=agg_weights,
    )


def mean_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
) -> T:
    r"""
    Calculates the mean error between a forecast and observations.

    The mean error is defined as:

    .. math::

        e = \frac{1}{N} \sum_{i=1}^N w_i (f_i - o_i)

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`w_i` are the weights,
    - :math:`e` is the mean error.

    Parameters
    ----------
    fcst : xarray object
        The forecast xarray.
    obs : xarray object
        The observations xarray.
    over : str or list of str
        The dimension(s) over which to aggregate.
    weights : xarray object, optional
        Weights to apply during aggregation. Default is None.

    Returns
    -------
    xarray object
        The mean error between the forecast and observations.
    """
    return error(fcst, obs, agg_method="mean", agg_dim=over, agg_weights=weights)


def abs_error(
    fcst: T,
    obs: T,
    agg_method: Literal["mean"] | None = None,
    agg_dim: str | list[str] | None = None,
    agg_weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    assert agg_method in (None, "mean")
    scores = _import_scores_or_prompt_install()
    reduce_dim = agg_dim or []
    return scores.continuous.mae(
        fcst, obs, reduce_dims=reduce_dim, weights=agg_weights, is_angular=is_angular
    )


def mean_abs_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    return abs_error(fcst, obs, agg_method="mean", agg_dim=over, agg_weights=weights, is_angular=is_angular)


def squared_error(
    fcst: T,
    obs: T,
    agg_method: Literal["mean"] | None = None,
    agg_dim: str | list[str] | None = None,
    agg_weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    assert agg_method in (None, "mean")
    scores = _import_scores_or_prompt_install()
    reduce_dim = agg_dim or []
    return scores.continuous.mse(
        fcst, obs, reduce_dims=reduce_dim, weights=agg_weights, is_angular=is_angular
    )


def mean_squared_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    return squared_error(
        fcst, obs, agg_method="mean", agg_dim=over, agg_weights=weights, is_angular=is_angular
    )


def root_mean_squared_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    return mean_squared_error(fcst, obs, over, weights=weights, is_angular=is_angular) ** 0.5


def standard_deviation_of_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
) -> T:
    # TODO: support angular inputs in the future (is_angular)
    # Minimal implementation using xarray operations; supports weights
    error = fcst - obs

    if weights is None:
        mean = error.mean(dim=over)
        var = ((error - mean) ** 2).mean(dim=over)
    else:
        mean = error.weighted(weights).mean(dim=over)
        var = ((error - mean) ** 2).weighted(weights).mean(dim=over)
    return var**0.5
