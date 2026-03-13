from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeVar

from ..utils.decorators import dispatch

if TYPE_CHECKING:
    import xarray as xr

    T = TypeVar("T")


def error(
    fcst: T,
    obs: T,
    agg_method: Literal["mean"] | None = None,
    agg_dim: str | list[str] | None = None,
    agg_weights: xr.DataArray | None = None,
) -> T:
    r"""
    Calculates the error between a forecast and observations.

    .. warning:: Experimental API. This function may change or be removed without notice.

    The error is defined as:

    .. math::

        e_i = f_i - o_i

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`e_i` is the error.

    .. seealso::

        This function leverages the
        `scores.continuous.additive_bias <https://scores.readthedocs.io/en/latest/api.html>`_
        function.

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
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(error, fieldlist=False)
    return dispatched(fcst, obs, agg_method, agg_dim, agg_weights)


def mean_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
) -> T:
    r"""
    Calculates the mean error between a forecast and observations.

    .. warning:: Experimental API. This function may change or be removed without notice.

    The mean error is defined as:

    .. math::

        e = \frac{\sum_{i=1}^N (f_i - o_i) w_i}{\sum_{i=1}^N w_i}

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`w_i` are the weights,
    - :math:`e` is the mean error.

    .. seealso::

        This function leverages the
        `scores.continuous.additive_bias <https://scores.readthedocs.io/en/latest/api.html>`_
        function.

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
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(mean_error, fieldlist=False)
    return dispatched(fcst, obs, over, weights)


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

    .. warning:: Experimental API. This function may change or be removed without notice.

    The absolute error is defined as:

    .. math::

        e_i = |f_i - o_i|

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`e_i` is the absolute error.

    .. seealso::

        This function leverages the
        `scores.continuous.mae <https://scores.readthedocs.io/en/latest/api.html#scores.continuous.mae>`_
        function.

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
    is_angular : bool, optional
        Whether the data represents angular quantities in degrees. Default is False.

    Returns
    -------
    xarray object
        The error between the forecast and observations, possibly aggregated.
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(abs_error, fieldlist=False)
    return dispatched(fcst, obs, agg_method, agg_dim, agg_weights, is_angular)


def mean_abs_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    r"""
    Calculates the mean absolute error between a forecast and observations.

    .. warning:: Experimental API. This function may change or be removed without notice.

    The mean absolute error is defined as:

    .. math::

        e = \frac{\sum_{i=1}^N |f_i - o_i| w_i}{\sum_{i=1}^N w_i}

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`w_i` are the weights,
    - :math:`e` is the mean absolute error.

    .. seealso::

        This function leverages the
        `scores.continuous.mae <https://scores.readthedocs.io/en/latest/api.html#scores.continuous.mae>`_
        function.

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
    is_angular : bool, optional
        Whether the data represents angular quantities in degrees. Default is False.

    Returns
    -------
    xarray object
        The mean absolute error between the forecast and observations.
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(mean_abs_error, fieldlist=False)
    return dispatched(fcst, obs, over, weights, is_angular)


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

    .. warning:: Experimental API. This function may change or be removed without notice.

    The squared error is defined as:

    .. math::

        e_i = (f_i - o_i)^2

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`e_i` is the absolute error.

    .. seealso::

        This function leverages the
        `scores.continuous.mse <https://scores.readthedocs.io/en/latest/api.html#scores.continuous.mse>`_
        function.

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
    is_angular : bool, optional
        Whether the data represents angular quantities in degrees. Default is False.

    Returns
    -------
    xarray object
        The squared error between the forecast and observations, possibly aggregated.
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(squared_error, fieldlist=False)
    return dispatched(fcst, obs, agg_method, agg_dim, agg_weights, is_angular)


def mean_squared_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    r"""
    Calculates the mean squared error between a forecast and observations.

    .. warning:: Experimental API. This function may change or be removed without notice.

    The mean squared error is defined as:

    .. math::

        e = \frac{\sum_{i=1}^N (f_i - o_i)^2 w_i}{\sum_{i=1}^N w_i}

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`w_i` are the weights,
    - :math:`e` is the mean squared error.

    .. seealso::

        This function leverages the
        `scores.continuous.mse <https://scores.readthedocs.io/en/latest/api.html#scores.continuous.mse>`_
        function.

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
    is_angular : bool, optional
        Whether the data represents angular quantities in degrees. Default is False.

    Returns
    -------
    xarray object
        The mean squared error between the forecast and observations.
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(mean_squared_error, fieldlist=False)
    return dispatched(fcst, obs, over, weights, is_angular)


def root_mean_squared_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
    is_angular: bool = False,
) -> T:
    r"""
    Calculates the root mean squared error between a forecast and observations.

    .. warning:: Experimental API. This function may change or be removed without notice.

    The root mean squared error is defined as:

    .. math::

        e = \sqrt{ \frac{\sum_{i=1}^N (f_i - o_i)^2 w_i}{\sum_{i=1}^N w_i} }

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`w_i` are the weights,
    - :math:`e` is the root mean squared error.

    .. seealso::

        This function leverages the
        `scores.continuous.mse <https://scores.readthedocs.io/en/latest/api.html#scores.continuous.mse>`_
        function.

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
    is_angular : bool, optional
        Whether the data represents angular quantities in degrees. Default is False.

    Returns
    -------
    xarray object
        The root mean squared error between the forecast and observations.
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(root_mean_squared_error, fieldlist=False)
    return dispatched(fcst, obs, over, weights, is_angular)


def standard_deviation_of_error(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
) -> T:
    r"""
    Calculates the standard deviation of error between a forecast and observations.

    .. warning:: Experimental API. This function may change or be removed without notice.

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
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(standard_deviation_of_error, fieldlist=False)
    return dispatched(fcst, obs, over, weights)


def cosine_similarity(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
) -> T:
    r"""
    Calculates the cosine similarity between a forecast and observations.

    .. warning:: Experimental API. This function may change or be removed without notice.

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
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(cosine_similarity, fieldlist=False)
    return dispatched(fcst, obs, over, weights)


def pearson_correlation(
    fcst: T,
    obs: T,
    over: str | list[str],
    weights: xr.DataArray | None = None,
) -> T:
    r"""
    Calculates the Pearson correlation between a forecast and observations.

    .. warning:: Experimental API. This function may change or be removed without notice.

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
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(pearson_correlation, fieldlist=False)
    return dispatched(fcst, obs, over, weights)
