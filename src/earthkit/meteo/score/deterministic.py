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

    Unreachable nodes are given a distance of :math:`\infty`.

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
    pass


def mean_error(
    fcst: T,
    obs: T,
    agg_dim: str | list[str] | None = None,
    agg_weights: xr.DataArray | None = None,
) -> T:
    pass


def abs_error(
    fcst: T,
    obs: T,
    agg_method: Literal["mean"] | None = None,
    agg_dim: str | list[str] | None = None,
    agg_weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    pass


def mean_abs_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    pass


def squared_error(
    fcst: T,
    obs: T,
    agg_method: Literal["mean"] | None = None,
    agg_dim: str | list[str] | None = None,
    agg_weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    pass


def mean_squared_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    pass


def root_mean_squared_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    pass


def standard_deviation_of_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    pass
