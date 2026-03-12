from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeVar

from ..utils.decorators import dispatch

if TYPE_CHECKING:
    import xarray as xr

    T = TypeVar("T")


def spread(fcst: T, over: str | list[str], reference: T | None = None) -> T:
    r"""
    Calculates the spread of a forecast compared to a reference.

    .. warning:: Experimental API. This function may change or be removed without notice.

    The spread is defined as:

    .. math::

        s_i = \sqrt{ \frac{1}{N} \sum_{i=1}^N \left(f_i - r\right)^2}

    where:

    - :math:`f_i` is the forecast,
    - :math:`r` is the reference,
    - :math:`s_i` is the spread.

    Parameters
    ----------
    fcst : xarray object
        The forecast xarray.
    over : str or list of str
        The dimension(s) over which to compute the spread.
    reference : xarray object, optional
        The reference xarray to compare against. If not provided, the mean of the forecast over `over` is used.

    Returns
    -------
    xarray object
        The spread of the forecast compared to the reference.

    The function returns an object of the same type as the input arguments.
    """
    return dispatch(spread)(fcst, over, reference)


def quantile_score(fcst: T, obs: T, tau: float, over: str | list[str]) -> T:
    r"""
    Calculates the quantile score of a forecast compared to a set of observations.

    .. warning:: Experimental API. This function may change or be removed without notice.

    The quantile score is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        q_i &= \text{Quantile of the forecast at level } \tau \\
        qs_i &= |o_i - q_i| + (2 \tau - 1) (o_i - q_i)
        \end{align*}

    where:

    - :math:`f_i` is the forecast,
    - :math:`o_i` are the observations,
    - :math:`\tau` is the quantile level,
    - :math:`qs_i` is the quantile score.

    Parameters
    ----------
    fcst : xarray object
        The forecast xarray.
    obs : xarray object
        The observations xarray.
    tau : float
        The quantile level in the range (0, 1).
    over : str or list of str
        The dimension(s) over which to compute the quantile score.

    Returns
    -------
    xarray object
        The quantile score of the forecast compared to the observations.

    The function returns an object of the same type as the input arguments.
    """
    return dispatch(quantile_score)(fcst, obs, tau, over)


# TODO: try to unify returns with crps_from_cdf and crps_from_ensemble
def crps_from_gaussian(fcst: xr.Dataset, obs: xr.DataArray) -> xr.DataArray:
    r"""
    Calculates the continuous ranked probability score (CRPS) of a forecast described by mean and standard deviation.

    .. warning:: Experimental API. This function may change or be removed without notice.

    The CRPS score for a Gaussian distribution is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        \operatorname{CRPS}\left[ \mathcal{N}(\mu, \sigma^2), o \right] = &\sigma \left\{ \frac{o - \mu}{\sigma} \left[ 2 \Phi \left( \frac{o-\mu}{\sigma} \right) - 1\right] \right. \\
        &\left. +2\phi\left( \frac{o - \mu}{\sigma}  \right) - \frac{1}{\sqrt{\pi}} \right\}
        \end{align*}

    where:

    - :math:`\mathcal{N}(\mu, \sigma^2)` is the probabilistic (Gaussian) forecast,
    - :math:`o` are the observations,
    - :math:`\phi\left( (o - \mu)/\sigma \right)` denotes the probability density function of the normal distribution with mean 0 and variance 1 evaluated at the normalised prediction error, :math:`(o - \mu)/\sigma`,
    - :math:`\Phi\left( (o - \mu)/\sigma \right)` denotes the cumulative distribution function of the normal distribution with mean 0 and variance 1 evaluated at the normalised prediction error, :math:`(o - \mu)/\sigma`.

    Reference: Gneiting, Tilmann, et al. "Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation." Monthly weather review 133.5 (2005): 1098-1118.

    Parameters
    ----------
    fcst : xarray Dataset
        The forecast xarray. Must have variables "mean" and "stdev".
    obs : xarray DataArray
        The observations DataArray.

    Returns
    -------
    xarray.DataArray
        The CRPS of the Gaussian forecast compared to the observations.

    The function returns an object of the same type as the input arguments.
    """
    return dispatch(crps_from_gaussian)(fcst, obs)


def crps_from_ensemble(
    fcst: T,
    obs: T,
    over: str | list[str],
    method: Literal["ecdf", "fair"] = "ecdf",
    return_components: bool = False,
    decomposition_method: Literal["underover", "hersbach"] = "underover",
) -> T:
    r"""
    Calculates the continuous ranked probability score (CRPS) of an ensemble forecast.

    .. warning:: Experimental API. This function may change or be removed without notice.

    The CRPS score for an ensemble forecast is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        \operatorname{CRPS}\left[f, o\right] =  \frac{\sum_{i=1}^{M}(|f_i - o|)}{M} - \frac{\sum_{i=1}^{M}\sum_{j=1}^{M}(|f_i - f_j|)}{2K}
        \end{align*}

    where:

    - :math:`f` is the probabilistic ensemble forecast,
    - :math:`o` are the observations,
    - :math:`K=M^2` for the 'ecdf' method and :math:`M(M-1)` for the 'fair' method,

    With `return_components=True`, this function returns an ``xr.Dataset`` with variables for the decompositions defined below.

    If the `decomposition_method="underover"`, the ``xr.Dataset`` variables values are
    ``underforecast_penalty``, ``overforecast_penalty``, ``spread`` and either ``fcrps`` if `method="fair"` or ``crps`` if `method="ecdf"` (ordering is not
    guaranteed and might differ). The overall CRPS is given by
    ``underforecast_penalty + overforecast_penalty - spread``.

    .. math::

        \operatorname{CRPS}[f, o] = O(f, o) + U(f, o) - S(f, f)

    where

    .. math::
        :nowrap:

        \begin{align*}
        O(f, o) &= \frac{1}{M} \sum_{i=1}^{M} (f_i - o)\,\mathbb{1}_{\{f_i > o\}} \quad& \text{(overforecast penalty)} \\
        U(f, o) &= \frac{1}{M} \sum_{i=1}^{M} (o - f_i)\,\mathbb{1}_{\{f_i < o\}} \quad& \text{(underforecast penalty)} \\
        S(f, f) &= \frac{1}{2K} \sum_{i=1}^{M} \sum_{j=1}^{M} |f_i - f_j| \quad& \text{(forecast spread term)}
        \end{align*}

    If the decomposition method is `decomposition_method="hersbach"`, the ``xr.Dataset`` variables values are
    ``alpha``, ``beta``, ``crps`` and additionally also ``fcrps`` if `method="fair"` (ordering is not guaranteed and might differ).

    We denote by :math:`x_1 \le x_2 \le \dots \le x_M` the members of the ensemble forecast :math:`f` after sorting. The unfair CRPS decomposition for `decomposition_method="hersbach"` is then given by

    .. math::
        :nowrap:

        \begin{align*}
        \operatorname{CRPS}\left[f, o\right] =  \sum_{i=1}^{M} \alpha_i p_i^2 + \beta_i (1-p_i)^2
        \end{align*}

    where

    .. math::
        :nowrap:

        \begin{align*}
        \alpha_i = & \begin{cases}
        o - x_M & \text{if } o > x_M \\
        x_{i+1} - x_i & \text{if } o > x_{i+1} \\
        o - x_i & \text{if } x_{i+1} > o > x_{i} \\
        0 & \text{if } o < x_{i} \\
        0 & \text{if } o < x_{1} \\
        \end{cases} \\
        \beta_i = & \begin{cases}
        0 & \text{if } o > x_M \\
        0 & \text{if } o > x_{i+1} \\
        x_{i+1} - o & \text{if } x_{i+1} > o > x_{i} \\
        x_{i+1} - x_i & \text{if } o < x_{i} \\
        x_1 - o & \text{if } o < x_{1} \\
        \end{cases} \\
        p_i = & \begin{cases}
        \frac{i}{M} & \text{if } 0<i<M \\
        0 & \text{if } i=0 \\
        1 & \text{if } i=M \\
        \end{cases}
        \end{align*}

    Fair CRPS is obtained by adding a correcting term :math:`\frac{G}{2M}` to the previous expression i.e.

    .. math::
        :nowrap:

        \begin{align*}
        \operatorname{CRPS}\left[f, o\right] = \sum_{i=1}^{M} \alpha_i p_i^2 + \beta_i (1-p_i)^2 - \frac{G}{2M}
        \end{align*}

    where

    .. math::

        G = \frac{\sum_{i=1}^{M} \sum_{j=1}^{M} |x_i - x_j|}{M (M-1)}

    Note that other CRPS decompositions exist; compare :func:`crps_from_cdf`.

    When ``return_components=False``, only a ``xr.DataArray`` of the total CRPS is
    returned.

    .. seealso::

        This function leverages the `scores.probability.crps_for_ensemble <https://scores.readthedocs.io/en/latest/api.html#scores.probability.crps_for_ensemble>`_ function.

    Parameters
    ----------
    fcst : xarray.DataArray
        The ensemble forecast xarray.
    obs : xarray.DataArray
        The observations xarray.
    over : str or list of str
        The dimension(s) over which to compute the CRPS.
    method : str, optional
        The method to compute the CRPS. Either 'ecdf' or 'fair'. Default is 'ecdf'.
    return_components : bool, optional
        Whether to return the components of the CRPS. Default is False.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The CRPS of the ensemble forecast compared to the observations.

    The function returns an object of the same type as the input arguments.
    """
    return dispatch(crps_from_ensemble)(fcst, obs, over, method, return_components, decomposition_method)


def crps_from_cdf(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    over: str,
    weight: xr.DataArray | None = None,
    return_components: bool = False,
) -> xr.DataArray:
    r"""
    Calculates the continuous ranked probability score (CRPS) for forecasts provided as CDFs.

    .. warning:: Experimental API. This function may change or be removed without notice.

    The CRPS score for a CDF forecast is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        o(x) &= 0 ~\text{if}~ x < \text{obs and}~ 1 ~\text{if}~ x \ge \text{obs}, \\
        \operatorname{CRPS}\left[f, o\right] &= \int_{-\infty}^{\infty}{w(x)\,(f(x) - o(x))^2\,\text{d}x}
        \end{align*}

    where:

    - :math:`f` is the forecast CDF,
    - :math:`o` is the observation converted to CDF form,
    - :math:`w(x)` is an optional non-negative threshold weight function.

    With ``return_components=True``, this function returns the decomposition defined below. The
    output is an ``xarray.Dataset`` with **no new dimension added**; instead it contains data
    variables named ``crps``, ``underforecast_penalty``, and ``overforecast_penalty`` at the same
    non-threshold coordinates. The overall CRPS is given by
    ``underforecast_penalty + overforecast_penalty``.

    .. math::

        \operatorname{CRPS}[f, o] = O(f, o) + U(f, o)

    .. math::
        :nowrap:

        \begin{align*}
        O(f, o) &= \int_{\text{obs}}^{\infty}{w(x)\,(f(x) - 1)^2\,\text{d}x} \quad& \text{(overforecast penalty)} \\
        U(f, o) &= \int_{-\infty}^{\text{obs}}{w(x)\,f(x)^2\,\text{d}x} \quad& \text{(underforecast penalty)}
        \end{align*}

    Note that there are several ways to decompose the CRPS and this decomposition differs from the
    one used in :func:`crps_from_ensemble`.

    .. seealso::

        This function leverages the `scores.probability.crps_cdf <https://scores.readthedocs.io/en/latest/api.html#scores.probability.crps_cdf>`_ function.

    Parameters
    ----------
    fcst : xarray DataArray
        Forecast CDF values with threshold dimension ``over``.
    obs : xarray DataArray
        Observations (not in CDF form). Must not include the ``over`` dimension.
    over : str
        The single threshold dimension for the CDF values.
    weight : xarray DataArray, optional
        Threshold weights along ``over``. Must include ``over`` as a dimension and be broadcastable
        to ``fcst``. If ``None``, a weight of 1 is used.
    return_components : bool, optional
        Whether to return the under/over forecast components as additional data variables.
        Default is False.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The CRPS of the CDF compared to the observations.

    The function returns an object of the same type as the input arguments.
    """
    return dispatch(crps_from_cdf)(fcst, obs, over, weight, return_components)
