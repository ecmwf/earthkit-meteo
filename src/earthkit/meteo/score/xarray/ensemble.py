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

    With `return_components=True`, this function returns the decomposition defined below with an
    added ``component`` dimension.

    If the decomposition method is ``underover``, the ``component`` values are ``total``,
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

    If the decomposition method is ``hersbach`, the ``component`` values are ...
    [ TODO: fill ]

    Note that other CRPS decompositions exist; compare :func:`crps_from_cdf`.

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
    assert decomposition_method in ["underover", "hersbach"]
    assert method in ["fair", "ecdf"]
    if decomposition_method == "underover":
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
    else:
        valid_mask, alpha, beta, crps, fcrps = _crps_from_ensemble_hersbach(fcst, obs, over)
        if return_components:
            if method == "fair":
                return xr.Dataset(
                    {
                        "alpha": alpha.where(valid_mask),
                        "beta": beta.where(valid_mask),
                        "total": fcrps.where(valid_mask),
                    }
                )
            else:
                return xr.Dataset(
                    {
                        "alpha": alpha.where(valid_mask),
                        "beta": beta.where(valid_mask),
                        "total": crps.where(valid_mask),
                    }
                )

        else:
            return fcrps.where(valid_mask) if method == "fair" else crps.where(valid_mask)


# TODO: is the output type Dataset right?
# TODO: complete docstring (fair CRPS)
# TODO: test against crps_from_ensemble (CRPS and fairCRPS should be identical arrays)
# TODO: does this work when over is a list of dimensions?
# TODO: decide on the nan distribution strategy and make sure it's consistent with other functions (e.g. crps_from_ensemble)
def _crps_from_ensemble_hersbach(
    fcst: T,
    obs: T,
    over: str | list[str],
    components_coords: np.ndarray | list | None = None,
) -> T:
    r"""
    Calculates the continuous ranked probability score (CRPS) of an ensemble forecast.

    .. warning:: Experimental API. This function may change or be removed without notice.

    The function returns CRPS arrays calculated by two methods

        - ``crps`` the exact CRPS value for the empirical cumulative distribution function
            constructed using the ensemble values;
        - ``fair`` the approximated CRPS where the ensemble values can be interpreted as a
            random sample from the underlying predictive distribution.

    The exact CRPS score for an ensemble forecast is computed based on eqns (26) to (29) in Hersbach (2000) as:

    .. math::
        :nowrap:

        \begin{align*}
        \operatorname{CRPS} = \sum_{i=0}^{M} \left[ \alpha_i p_i^2 + \beta_i (1-p_i)^2 \right]
        \end{align*}

    where the components :math:`\alpha_i` and :math:`\beta_i` are defined by eqns (26) and (27) in Hersbach (2000)
    and the probabilities

    .. math::

        p_i = i / M

    A fair version of CRPS is then obtained by

    .. math::

        \operatorname{fairCRPS} = \operatorname{CRPS} + ...


    Parameters
    ----------
    fcst : xarray object
        The ensemble forecast xarray.
    obs : xarray object
        The observations xarray.
    over : str or list of str
        The dimension(s) over which to compute the CRPS.
    components_coords : np.ndarray or list, optional
        The coordinates to assign to the CRPS alpha and CRPS beta components dimension.
        If None, defaults to a sequence from 1 to ensemble size + 1.

    Returns
    -------
    xarray Dataset
        A dataset containing ``crps`` CRPS, ``fair`` CRPS, and ``alpha`` and ``beta`` CRPS components as data variables.
        The CRPS alpha and CRPS beta components have a dimension named ``over`` with coordinates given
        by ``components_coords``.
    """

    ens_size = fcst.sizes[over]
    if components_coords is None:
        components_coords = np.arange(1, ens_size + 2)
    else:
        assert (
            len(components_coords) == ens_size + 1
        ), "component_coords must have the length of ensemble size + 1"
    # sort forecast values along the ensemble dimension
    fcst_sorted = _sorted_ensemble(fcst, over)
    alpha = xr.concat([xr.zeros_like(fcst_sorted[{over: 0}])] * (ens_size + 1), dim=over)
    beta = alpha.copy()
    # note the order in operations between forecasts and observations matters
    # for the broadcasting to work correctly
    obs_below_ens = fcst_sorted[{over: 0}] > obs
    alpha[{over: 0}] = alpha[{over: 0}].where(~obs_below_ens, 1.0)
    beta[{over: 0}] = (fcst_sorted[{over: 0}] - obs).where(obs_below_ens, 0.0)

    rhs = (
        fcst_sorted.diff(dim=over)
        .where(
            fcst_sorted[{over: slice(1, None)}] <= obs,
            -fcst_sorted[{over: slice(None, -1)}] + obs,
        )
        .where(fcst_sorted[{over: slice(None, -1)}] <= obs, 0.0)
    )
    rhs = rhs.transpose(*alpha.dims)
    alpha[{over: slice(1, -1)}] = rhs

    rhs = (
        fcst_sorted.diff(dim=over)
        .where(
            fcst_sorted[{over: slice(None, -1)}] > obs,
            fcst_sorted[{over: slice(1, None)}] - obs,
        )
        .where(fcst_sorted[{over: slice(1, None)}] > obs, 0.0)
    )
    rhs = rhs.transpose(*beta.dims)
    beta[{over: slice(1, -1)}] = rhs

    obs_above_ens = fcst_sorted[{over: -1}] < obs
    alpha[{over: -1}] = (-fcst_sorted[{over: -1}] + obs).where(obs_above_ens, 0.0)
    beta[{over: -1}] = beta[{over: -1}].where(~obs_above_ens, 1.0)
    alpha = alpha.assign_coords({over: components_coords})
    beta = beta.assign_coords({over: components_coords})
    weight = xr.DataArray(np.arange(ens_size + 1), dims=over) / float(ens_size)
    crps = (alpha * weight**2 + beta * (1.0 - weight) ** 2).sum(over)
    fcrps = crps - _ginis_mean_diff(fcst_sorted, over) / (2.0 * ens_size)
    # get the mask of valid values across observations and ensemble members
    valid_mask = _common_valid_mask(obs, fcst_sorted, dim=over)
    return valid_mask, alpha, beta, crps, fcrps


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


def ginis_mean_diff(fcst, over):
    r"""
    Gini's mean difference. REFERENCE: Ferro et al. (2008)

    .. math::

        \frac{\sum_i^{nens} \sum_j^{nens} |x_i - x_j|}{n (n-1)}

    Parameters
    ----------
    fcst : xarray object
        The ensemble forecast xarray.
    over : str or list of str
        The dimension(s) over which to compute.

    Returns
    -------
    xarray object
        Gini's mean difference.
    """

    valid_mask = _common_valid_mask(fcst, dim=over)
    forecasts = _sorted_ensemble(fcst, dim=over)
    return _ginis_mean_diff(forecasts, over).where(valid_mask)


# TODO: is it really dask-safe?
def _sorted_ensemble(forecasts, dim):
    # sort forecast values along the ensemble dimension
    return xr.apply_ufunc(
        np.sort,
        forecasts,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        exclude_dims=set((dim,)),
        kwargs={"axis": -1},
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {dim: forecasts.sizes[dim]}},
    )


def _common_valid_mask(*arrays, dim=None):
    # return a numpy bool array of occurrences of all values valid across dim
    mask = None
    for array in arrays:
        if dim is not None and dim in array.dims:
            nmask = array.notnull().all(dim=dim)
        else:
            nmask = array.notnull()
        if mask is None:
            mask = nmask
        else:
            mask = mask & nmask
    return mask


def _ginis_mean_diff(fcst_sorted, over):
    # ( sum_i sum_j abs(x_i-x_j) )/(n*(n-1)) along dimension over
    # NB: the algorithm assumes fcst_sorted has been sorted along dimension over
    # pretty much copied from scores.probability.crps_for_ensemble()
    ens_size = fcst_sorted.sizes[over]
    npairs = ens_size * (ens_size - 1)
    i = xr.DataArray(np.arange(ens_size), dims=[over])
    coeffs = 2 * i - fcst_sorted.count(over) + 1
    gmd = 2 * (fcst_sorted * coeffs).sum(dim=over, skipna=True)
    return gmd / npairs
