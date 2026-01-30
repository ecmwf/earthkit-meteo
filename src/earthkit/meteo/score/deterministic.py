from typing import Literal
from typing import TypeVar

import numpy as np
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

    .. seealso::

        This function leverages the `scores.continuous.additive_bias <https://scores.readthedocs.io/en/latest/api.html#scores.continuous.additive_bias>`_ function.

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

        e = \frac{\sum_{i=1}^N (f_i - o_i) w_i}{\sum_{i=1}^N w_i}

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`w_i` are the weights,
    - :math:`e` is the mean error.

    .. seealso::

        This function leverages the `scores.continuous.additive_bias <https://scores.readthedocs.io/en/latest/api.html#scores.continuous.additive_bias>`_ function.

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
    return error(
        fcst,
        obs,
        agg_method="mean",
        agg_dim=over,
        agg_weights=weights,
    )


def abs_error(
    fcst: T,
    obs: T,
    agg_method: Literal["mean"] | None = None,
    agg_dim: str | list[str] | None = None,
    agg_weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    r"""
    Calculates the absolute error between a forecast and observations.

    The absolute error is defined as:

    .. math::

        e_i = |f_i - o_i|

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`e_i` is the absolute error.

    .. seealso::

        This function leverages the `scores.continuous.mae <https://scores.readthedocs.io/en/latest/api.html#scores.continuous.mae>`_ function.

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
    reduce_dim = agg_dim or []
    return scores.continuous.mae(
        fcst,
        obs,
        reduce_dims=reduce_dim,
        weights=agg_weights,
        is_angular=is_angular,
    )


def mean_abs_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    r"""
    Calculates the mean absolute error between a forecast and observations.

    The mean absolute error is defined as:

    .. math::

        e = \frac{\sum_{i=1}^N |f_i - o_i| w_i}{\sum_{i=1}^N w_i}

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`w_i` are the weights,
    - :math:`e` is the mean absolute error.

    .. seealso::

        This function leverages the `scores.continuous.mae <https://scores.readthedocs.io/en/latest/api.html#scores.continuous.mae>`_ function.

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
        The mean absolute error between the forecast and observations.
    """
    return abs_error(fcst, obs, agg_method="mean", agg_dim=over, agg_weights=weights, is_angular=is_angular)


def squared_error(
    fcst: T,
    obs: T,
    agg_method: Literal["mean"] | None = None,
    agg_dim: str | list[str] | None = None,
    agg_weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    r"""
    Calculates the squared error between a forecast and observations.

    The absolute error is defined as:

    .. math::

        e_i = (f_i - o_i)^2

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`e_i` is the absolute error.

    .. seealso::

        This function leverages the `scores.continuous.mse <https://scores.readthedocs.io/en/latest/api.html#scores.continuous.mse>`_ function.

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
        The squared error between the forecast and observations, possibly aggregated.
    """
    assert agg_method in (None, "mean")
    scores = _import_scores_or_prompt_install()
    reduce_dim = agg_dim or []
    return scores.continuous.mse(
        fcst,
        obs,
        reduce_dims=reduce_dim,
        weights=agg_weights,
        is_angular=is_angular,
    )


def mean_squared_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    r"""
    Calculates the mean squared error between a forecast and observations.

    The mean squared error is defined as:

    .. math::

        e = \frac{\sum_{i=1}^N (f_i - o_i)^2 w_i}{\sum_{i=1}^N w_i}

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`w_i` are the weights,
    - :math:`e` is the mean squared error.

    .. seealso::

        This function leverages the `scores.continuous.mse <https://scores.readthedocs.io/en/latest/api.html#scores.continuous.mse>`_ function.

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
        The mean squared error between the forecast and observations.
    """
    return squared_error(
        fcst,
        obs,
        agg_method="mean",
        agg_dim=over,
        agg_weights=weights,
        is_angular=is_angular,
    )


def root_mean_squared_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    r"""
    Calculates the root mean squared error between a forecast and observations.

    The root mean squared error is defined as:

    .. math::

        e = \sqrt{ \frac{\sum_{i=1}^N (f_i - o_i)^2 w_i}{\sum_{i=1}^N w_i} }

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`w_i` are the weights,
    - :math:`e` is the root mean squared error.

    .. seealso::

        This function leverages the `scores.continuous.mse <https://scores.readthedocs.io/en/latest/api.html#scores.continuous.mse>`_ function.

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
        The root mean squared error between the forecast and observations.
    """
    return mean_squared_error(fcst, obs, over, weights=weights, is_angular=is_angular) ** 0.5


def standard_deviation_of_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
) -> T:
    r"""
    Calculates the standard deviation of error between a forecast and observations.

    The standard deviation of error is defined as:

    .. math::

        e = \sqrt{ \frac{ \sum_{i=1}^N (f_i - o_i - m_e)^2 w_i }{ \sum_{i=1}^N w_i } }

    where:

    .. math::

        m_e = \frac{\sum_{i=1}^N (f_i - o_i) w_i}{\sum_{i=1}^N w_i}
    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`w_i` are the weights,
    - :math:`e` is the standard deviation of error.

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
        The standard deviation of error between the forecast and observations.
    """
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


def cosine_similarity(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
) -> T:
    r"""
    Calculates the cosine similarity between a forecast and observations.

    The cosine similarity is defined as:

    .. math::
        \frac{\overline{f o}}{\sqrt{\overline{f^2}\ \overline{o^2}}}

    where the averaging operator is defined as:

    .. math::
        \overline{x} = \frac{\sum_{i=1}^N x_i w_i}{\sum_{i=1}^N w_i}

    and

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`w_i` are the weights,
    - :math:`c` is the cosine similarity.

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
        The cosine similarity between the forecast and observations.
    """

    def _weighted_mean(array, weights, dim):
        if weights is None:
            return array.mean(dim)
        return array.weighted(weights=weights).mean(dim=dim)

    def _common_valid_mask(*arrays):
        # return a np bool array of occurrences of all values valid across dim
        # return xr.concat(arrays, dim=dim).notnull().all(dim=dim)
        mask = None
        for array in arrays:
            nmask = array.notnull()
            if mask is None:
                mask = nmask
            else:
                mask = mask & nmask
        return mask

    # TODO: simplify logic
    valid_mask = _common_valid_mask(obs, fcst)
    fcs = fcst.where(valid_mask)
    obs = obs.where(valid_mask)
    fs_var2 = _weighted_mean(fcs**2, weights, over)
    ob_var2 = _weighted_mean(obs**2, weights, over)
    covar = _weighted_mean(fcs * obs, weights, over)
    result = covar / np.sqrt(fs_var2 * ob_var2)
    return result.where(np.isfinite(result))


def pearson_correlation(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
) -> T:
    r"""
    Calculates the Pearson correlation between a forecast and observations.

    The correlation is defined as:

    .. math::
        c = \frac{
        \sum_{i=1}^N {\left(f-\overline{f}\right)\left(o-\overline{o}\right)}
        }
        {\sqrt{
        \overline{\left(f-\overline{f}\right)^2}
        \
        \overline{\left(o-\overline{o}\right)^2}
        }}

    where the averaging operator is defined as:

    .. math::
        \overline{x} = \frac{\sum_{i=1}^N x_i w_i}{\sum_{i=1}^N w_i}

    and

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`w_i` are the weights,
    - :math:`c` is the correlation.

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
        The correlation between the forecast and observations.
    """

    def _weighted_mean(array, weights, dim):
        if weights is None:
            return array.mean(dim)
        return array.weighted(weights=weights).mean(dim=dim)

    def _common_valid_mask(*arrays):
        # return a np bool array of occurrences of all values valid across dim
        # return xr.concat(arrays, dim=dim).notnull().all(dim=dim)
        mask = None
        for array in arrays:
            nmask = array.notnull()
            if mask is None:
                mask = nmask
            else:
                mask = mask & nmask
        return mask

    # TODO: use scores implementation?
    # TODO: call the cosine similarity function?
    # with fcst - mean(fcst) and obs - mean(obs)
    # (mathematically equivalent to correlation)
    valid_mask = _common_valid_mask(obs, fcst)
    fcs = fcst.where(valid_mask)
    obs = obs.where(valid_mask)
    fs_var2 = _weighted_mean(fcs**2, weights, over)
    ob_var2 = _weighted_mean(obs**2, weights, over)
    covar = _weighted_mean(fcs * obs, weights, over)
    fc_mean = _weighted_mean(fcs, weights, over)
    ob_mean = _weighted_mean(obs, weights, over)
    fs_var2 = fs_var2 - fc_mean**2
    ob_var2 = ob_var2 - ob_mean**2
    covar = covar - fc_mean * ob_mean
    result = covar / np.sqrt(fs_var2 * ob_var2)
    return result.where(np.isfinite(result))
