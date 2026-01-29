"""
spread
quantile_score

crps_from_ensemble
crps_from_gaussian
crps_from_cdf

continuous_ignorance (maybe)
"""

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


def spread(fcst: T, over: str | list[str], reference: T | None = None) -> T:
    r"""
    Calculates the spread of a forecast compared to a reference.

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
        The reference xarray to compare against. If not provided, the mean of the forecastover `over` is used.

    Returns
    -------
    xarray object
        The spread of the forecast compared to the reference.
    """

    # TODO: this could call the rmse function
    if reference is None:
        reference = fcst.mean(dim=over)
    else:
        if over in reference.dims:
            reference = reference.squeeze(over)
    return ((fcst - reference) ** 2).mean(dim=over).sqrt()


def quantile_score(fcst: T, obs: T, tau: float, over: str | list[str]) -> T:
    r"""
    Calculates the quantile score of a forecast compared to a observations.

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
        The quantile level (between 0 and 1).
    over : str or list of str
        The dimension(s) over which to compute the quantile score.

    Returns
    -------
    xarray object
        The quantile score of the forecast compared to the observations.
    """
    qf = fcst.quantile(tau, dim=over)
    qscore = abs(obs - qf) + (2.0 * tau - 1.0) * (obs - qf)
    return qscore


def crps_gaussian(fcst, obs):
    r"""
    Calculates the continuous ranked probability score (CRPS) of a forecast described by mean and standard deviation.

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
    - :math:`\phi\left( (o - \mu)/\sigma \right)` denotes the probability density function of the normal distribution with mean 0 and variance 1 evaluated at the normalized prediction error, :math:`(o - \mu)/\sigma`,
    - :math:`\Phi\left( (o - \mu)/\sigma \right)` denotes the cumulative distribution function of the normal distribution with mean 0 and variance 1 evaluated at the normalized prediction error, :math:`(o - \mu)/\sigma`.

    Reference: Gneiting, Tilmann, et al. "Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation." Monthly weather review 133.5 (2005): 1098-1118.

    Parameters
    ----------
    fcst : xarray object
        The forecast xarray. Must have variables "mean" and "stdev".
    obs : xarray object
        The observations xarray.

    Returns
    -------
    xarray object
        The CRPS of the Gaussian forecast compared to the observations.
    """
    # TODO: support cupy
    import scipy

    c2 = np.sqrt(2.0 / np.pi)
    za = (obs - fcst["mean"]) / fcst["stdev"]
    return fcst["stdev"] * (
        (2.0 * scipy.stats.norm().cdf(za.values) - 1.0) * za
        + c2 * np.exp(-(za**2) / 2.0)
        - 1.0 / np.sqrt(np.pi)
    )


def crps_from_ensemble(fcst, obs, over, method="ecdf", return_components=False):
    r"""
    Calculates the continuous ranked probability score (CRPS) of an ensemble forecast.

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

    When the `return_components` flag is set to `True`, the CRPS components are calculated as:

    .. math::
        CRPS[f, o] = O(f, o) + U(f, o) - S(f, f)

    where
        - :math:`O(f, o) = \frac{\sum_{i=1}^{M} ((f_i - o) \mathbb{1}{\{f_i > o\}})}{M}` which is the overforecast penalty.
        - :math:`U(f, o) = \frac{\sum_{i=1}^{M} ((o - f_i) \mathbb{1}{\{f_i < o\}})}{M}` which is the underforecast penalty.
        - :math:`S(f, f) = \frac{\sum_{i=1}^{M}\sum_{j=1}^{M}(|f_i - f_j|)}{2K}` which is the forecast spread term.

    Note that there are several ways to decompose the CRPS and this decomposition differs from the
    one used in `crps_from_cdf`.

    Parameters
    ----------
    fcst : xarray object
        The ensemble forecast xarray.
    obs : xarray object
        The observations xarray.
    over : str or list of str
        The dimension(s) over which to compute the CRPS.
    method : str, optional
        The method to compute the CRPS. Either 'ecdf' or 'fair'. Default is 'ecdf'.
    return_components : bool, optional
        Whether to return the components of the CRPS. Default is False.

    Returns
    -------
    xarray object
        The CRPS of the ensemble forecast compared to the observations.
    """

    scores = _import_scores_or_prompt_install()
    reduce_dim = []
    return scores.ensemble.crps_for_ensemble(
        fcst,
        obs,
        over,
        method=method,
        reduce_dims=reduce_dim,
        preserve_dims=None,
        weights=None,
        include_components=return_components,
    )


def crps_from_cdf(fcst, obs, over, weight=None, return_components=False):
    r"""
    Calculates the continuous ranked probability score (CRPS) of the cumulative distribution function (CDF) forecast.

    The CRPS score for a CDF forecast is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        o(x) &= 0 ~\text{if}~ x < \text{obs and}~ 1 ~\text{if}~ x >= \text{obs}), \\
        \operatorname{CRPS}\left[f, o\right] &= \int_{-\infty}^{\infty}{[w(x) \times (f(x) - o(x))^2]\text{d}x},
        \end{align*}

    where:

    - :math:`f` is the CDF ensemble forecast,
    - :math:`o` are the observations,

    When the `return_components` flag is set to `True`, the CRPS components are calculated as:

    .. math::
        :nowrap:

        \begin{align*}
        CRPS[f, o] = O(f, o) + U(f, o)
        \end{align*}

    where

    - :math:`O(f, o) = \int_{-\infty}^{\infty}{[w(x) \times f(x) - o(x))^2]\text{d}x}`, over all thresholds :math:`x` where :math:`x\geq` obs, which is the over-forecast penalty,
    - :math:`U(f, o) = \int_{-\infty}^{\infty}{[w(x) \times f(x) - o(x))^2]\text{d}x}`, over all thresholds :math:`x` where :math:`x\leq` obs, which is the under-forecast penalty.

    Note that there are several ways to decompose the CRPS and this decomposition differs from the
    one used in `crps_from_ensemble`.

    Parameters
    ----------
    fcst : xarray object
        The ensemble forecast xarray.
    obs : xarray object
        The observations xarray.
    over : str or list of str
        The dimension(s) over which to compute the CRPS.
    method : str, optional
        The method to compute the CRPS. Either 'ecdf' or 'fair'. Default is 'ecdf'.
    return_components : bool, optional
        Whether to return the components of the CRPS. Default is False.

    Returns
    -------
    xarray object
        The CRPS of the CDF forecast compared to the observations.
    """

    scores = _import_scores_or_prompt_install()
    reduce_dim = []
    return scores.ensemble.crps_cdf(
        fcst,
        obs,
        threshold_dim=over,
        threshold_weight=weight,
        additional_thresholds=None,
        propagate_nans=True,
        fcst_fill_method="linear",
        threshold_weight_fill_method="forward",
        integration_method="exact",
        reduce_dims=reduce_dim,
        preserve_dims=None,
        weights=None,
        include_components=return_components,
    )
