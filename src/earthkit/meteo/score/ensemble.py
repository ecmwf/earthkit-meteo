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
    return ((fcst - reference) ** 2).mean(dim=over) ** 0.5


def quantile_score(fcst: T, obs: T, tau: float, over: str | list[str]) -> T:
    r"""
    Calculates the quantile score of a forecast compared to a observations.

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
    """
    if not (0.0 < tau < 1.0):
        raise ValueError("tau must be in the range (0, 1)")

    qf = fcst.quantile(tau, dim=over)
    qf = qf.drop_vars("quantile")
    qscore = abs(obs - qf) + (2.0 * tau - 1.0) * (obs - qf)
    return qscore


# TODO: Rename to crps_from_gaussian?
def crps_gaussian(fcst: xr.Dataset, obs: xr.DataArray) -> xr.DataArray:
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
    - :math:`\phi\left( (o - \mu)/\sigma \right)` denotes the probability density function of the normal distribution with mean 0 and variance 1 evaluated at the normalized prediction error, :math:`(o - \mu)/\sigma`,
    - :math:`\Phi\left( (o - \mu)/\sigma \right)` denotes the cumulative distribution function of the normal distribution with mean 0 and variance 1 evaluated at the normalized prediction error, :math:`(o - \mu)/\sigma`.

    Reference: Gneiting, Tilmann, et al. "Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation." Monthly weather review 133.5 (2005): 1098-1118.

    Parameters
    ----------
    fcst : xarray Dataset
        The forecast xarray. Must have variables "mean" and "stdev".
    obs : xarray DataArray
        The observations DataArray.

    Returns
    -------
    xarray object
        The CRPS of the Gaussian forecast compared to the observations.
    """
    if not isinstance(fcst, xr.Dataset):
        raise TypeError(f"Expected fcst to be an xarray.Dataset object, got {type(fcst)}")
    if not {"mean", "stdev"}.issubset(fcst.data_vars):
        raise ValueError(
            f"Expected fcst to have 'mean' and 'stdev' data variables, got {list(fcst.data_vars)}"
        )
    if not isinstance(obs, xr.DataArray):
        raise TypeError(f"Expected obs to be an xarray.DataArray object, got {type(obs)}")

    # TODO: support cupy
    import scipy

    c2 = np.sqrt(2.0 / np.pi)
    za = (obs - fcst["mean"]) / fcst["stdev"]
    return fcst["stdev"] * (
        (2.0 * scipy.stats.norm().cdf(za.values) - 1.0) * za
        + c2 * np.exp(-(za**2) / 2.0)
        - 1.0 / np.sqrt(np.pi)
    )


def crps_from_ensemble(
    fcst: T,
    obs: T,
    over: str | list[str],
    method: Literal["ecdf", "fair"] = "ecdf",
    return_components: bool = False,
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

    With `return_components=True`, this function returns the decomposition defined below with an
    added ``component`` dimension. The ``component`` values are ``total``,
    ``underforecast_penalty``, ``overforecast_penalty``, and ``spread`` (ordering is not
    guaranteed and might differ). The overall CRPS is given by
    ``total = underforecast_penalty + overforecast_penalty - spread``. When
    ``return_components=False``, no ``component`` dimension is added and only the total CRPS is
    returned.

    .. math::

        \operatorname{CRPS}[f, o] = O(f, o) + U(f, o) - S(f, f)

    .. math::
        :nowrap:

        \begin{align*}
        O(f, o) &= \frac{1}{M} \sum_{i=1}^{M} (f_i - o)\,\mathbb{1}_{\{f_i > o\}} \quad& \text{(overforecast penalty)} \\
        U(f, o) &= \frac{1}{M} \sum_{i=1}^{M} (o - f_i)\,\mathbb{1}_{\{f_i < o\}} \quad& \text{(underforecast penalty)} \\
        S(f, f) &= \frac{1}{2K} \sum_{i=1}^{M} \sum_{j=1}^{M} |f_i - f_j| \quad& \text{(forecast spread term)}
        \end{align*}

    Other CRPS decompositions exist; compare :func:`crps_from_cdf`.

    .. seealso::

        This function leverages the `scores.probability.crps_for_ensemble <https://scores.readthedocs.io/en/latest/api.html#scores.probability.crps_for_ensemble>`_ function.

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
    # TODO: revisit component ordering here and in tests
    reduce_dim = []
    return scores.probability.crps_for_ensemble(
        fcst,
        obs,
        over,
        method=method,
        reduce_dims=reduce_dim,
        preserve_dims=None,
        weights=None,
        include_components=return_components,
    )


def crps_from_cdf(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    over: str,
    weight: xr.DataArray | None = None,
    return_components: bool = False,
) -> xr.Dataset:
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
    variables named ``total``, ``underforecast_penalty``, and ``overforecast_penalty`` at the same
    non-threshold coordinates. The overall CRPS is given by
    ``total = underforecast_penalty + overforecast_penalty``.

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
    xarray Dataset
        A dataset containing ``total`` CRPS and, if requested, ``underforecast_penalty`` and
        ``overforecast_penalty`` as data variables. No new dimension is added; the output retains
        all non-threshold dimensions.
    """

    scores = _import_scores_or_prompt_install()
    reduce_dim = [over]
    return scores.probability.crps_cdf(
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
