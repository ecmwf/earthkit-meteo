import numpy as np
import xarray as xr


def import_scores_or_prompt_install():
    try:
        import scores
    except ImportError:
        raise ImportError(
            "The 'earthkit-meteo[score]' extra is required to use scoring functions. "
            "Please install it using 'pip install earthkit-meteo[score]'"
        )
    return scores


# determinstic scores


def error(fcst, obs, agg_method=None, agg_dim=None, agg_weights=None, is_angular=False):
    assert agg_method in (None, "mean")
    scores = import_scores_or_prompt_install()
    return scores.continuous.additive_bias(
        fcst, obs, reduce_dims=agg_dim, weights=agg_weights, is_angular=is_angular
    )


def mean_error(fcst, obs, over, weights=None, is_angular=False):
    return error(fcst, obs, agg_method="mean", agg_dim=over, agg_weights=weights, is_angular=is_angular)


def abs_error(fcst, obs, agg_method=None, agg_dim=None, agg_weights=None, is_angular=False):
    assert agg_method in (None, "mean")
    scores = import_scores_or_prompt_install()
    return scores.continuous.mae(fcst, obs, reduce_dims=agg_dim, weights=agg_weights, is_angular=is_angular)


def mean_abs_error(fcst, obs, over, weights=None, is_angular=False):
    return abs_error(fcst, obs, agg_method="mean", agg_dim=over, agg_weights=weights, is_angular=is_angular)


def squared_error(fcst, obs, agg_method=None, agg_dim=None, agg_weights=None, is_angular=False):
    assert agg_method in (None, "mean")
    scores = import_scores_or_prompt_install()
    return scores.continuous.mse(fcst, obs, reduce_dims=agg_dim, weights=agg_weights, is_angular=is_angular)


def mean_squared_error(fcst, obs, over, weights=None, is_angular=False):
    return squared_error(
        fcst, obs, agg_method="mean", agg_dim=over, agg_weights=weights, is_angular=is_angular
    )


def root_mean_squared_error(fcst, obs, over, weights=None, is_angular=False):
    return mean_squared_error(fcst, obs, over, weights=weights, is_angular=is_angular).sqrt()


def _weighted_mean(array, weights, dim):
    if weights is None:
        return array.mean(dim)
    return array.weighted(weights=weights).mean(dim=dim)


# TODO: check if should support ddof argument (current implementation is ddof=0, like np)
def standard_deviation_of_error(fcst, obs, over, weights=None, is_angular=False):
    assert is_angular is False, "is_angular=True is not supported for standard_deviation_of_error"
    error = fcst - obs
    return _weighted_mean((error - _weighted_mean(error, weights, over)) ** 2, weights, over).sqrt()


def _common_valid_mask(*arrays, dim=None):
    # return a np bool array of occurrences of all values valid across dim
    # return xr.concat(arrays, dim=dim).notnull().all(dim=dim)
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


# TODO: double check happy with unbiased as naming for argument
def correlation(fcst, obs, over, weights=None, is_angular=False, unbiased=True):
    assert is_angular is False, "is_angular=True is not supported for correlation"
    # we have to unset dim as we do not want to reduce ensemble mask to an intersection
    valid_mask = _common_valid_mask(obs, fcst, dim=None)
    fcs = fcst.where(valid_mask)
    obs = obs.where(valid_mask)
    fs_var2 = _weighted_mean(fcs**2, weights, over)
    ob_var2 = _weighted_mean(obs**2, weights, over)
    covar = _weighted_mean(fcs * obs, weights, over)
    if unbiased:
        fc_mean = _weighted_mean(fcs, weights, over)
        ob_mean = _weighted_mean(obs, weights, over)
        fs_var2 = fs_var2 - fc_mean**2
        ob_var2 = ob_var2 - ob_mean**2
        covar = covar - fc_mean * ob_mean
    result = covar / np.sqrt(fs_var2 * ob_var2)
    return result.where(~np.isinf(result))


# ensemble scores
def spread(fcst, over="ensemble", reference=None):
    # TODO: this could call the rmse function
    if reference is None:
        reference = fcst.mean(dim=over)
    else:
        if over in reference.dims:
            reference = reference.squeeze(over)
    return np.sqrt(((fcst - reference) ** 2).mean(dim=over))


# TODO: try to remove this
def _to_list(v):
    import datetime
    import numbers

    import pandas

    # NB: we don't want to convert to list such complex containers
    #     like DataArray, DataFrame, Series
    if isinstance(v, (list, tuple, set)):
        return list(v)
    if isinstance(v, (str, bytes, numbers.Number, datetime.datetime, datetime.timedelta)):
        return [v]
    if isinstance(v, (np.ndarray, pandas.Index)):
        return v.tolist()
    return [v]


def _stack_arr(item, unstacked_dims, order=None):
    _STACKED_DIM_NAME = "__multidim__"
    unstacked_dims = _to_list(unstacked_dims)
    if item is None:
        return None
    if isinstance(item, dict):
        return {
            key: _stack_arr(value, unstacked_dims, order=order)[0] for key, value in item.items()
        }, _STACKED_DIM_NAME
    dims = list(item.dims)
    for dim in unstacked_dims:
        try:
            dims.remove(dim)
        except ValueError:
            pass
    if order is not None:
        priority_dict = {k: i for i, k in enumerate(order)}
        dims = sorted(dims, key=lambda value: priority_dict.get(value, len(dims)))
    return item.stack({_STACKED_DIM_NAME: tuple(dims)}), _STACKED_DIM_NAME


def _unstack_arr(array, stacked_dim="__multidim__"):
    return array.unstack(stacked_dim)


def _ginis_mean_diff(e, stack_dim, dim, mode):
    # ( sum_i sum_j abs(x_i-x_j) )/(n*(n-1)) along dim
    nens = e.sizes[dim]
    npairs = nens * (nens - 1)
    match mode:
        case "fast":
            gmd = abs(e - e.rename({dim: "__tmp"})).sum(("__tmp", dim))
        case "small":
            ngmd = np.fromiter(
                (np.abs(np.subtract.outer(x, x)).sum() for x in e.transpose(stack_dim, ...).values),
                dtype=float,
            )
            template = e[{dim: 0}]
            gmd = xr.DataArray(ngmd, dims=template.dims, coords=template.coords, attrs=template.attrs)
        case _:
            raise ValueError(f"Gini's mean calculation mode {mode} not implemented.")
    return gmd / npairs


# TODO: check if we want toggle for fair crps
# TODO: check if we can use scores (they have different components they return)
# TODO: check if scores could compute fair crps
def crps(fcst, obs, over, return_components=False):
    # NB: The algorithm is implemented on 2D ndarrays
    compute_fcrps = return_components
    # As we need later to export to ndarrays, we force obs to the same coordinates order as fcst
    # not to rely on the alignment role of xr coordinates. This will have to be applied later to all
    # functions once xmetrics interfaces earthkit ndarray functions.
    dims = _to_list(over)
    obs, _ = _stack_arr(obs.loc[{k: v for k, v in fcst.coords.items() if k not in dims}], over)
    fcst, stack_dim = _stack_arr(fcst, over)
    anarr = obs.values
    earr = fcst.transpose(over, ...).values
    # ensemble sorted by fieldset axis
    esarr = np.sort(earr, axis=0)
    nens = fcst.sizes[over]
    aa = np.zeros(earr.shape)
    aa = np.concatenate((aa, aa[:1, :]))
    bb = aa.copy()
    with np.errstate(invalid="ignore"):
        lcond = esarr[0, :] > anarr
        aa[0, lcond] = 1.0
        bb[0, :] = np.where(lcond, esarr[0, :] - anarr, 0.0)
        aa[1:-1, :] = np.where(esarr[1:, :] <= anarr, esarr[1:, :] - esarr[:-1, :], anarr - esarr[:-1, :])
        aa[1:-1, :][esarr[: nens - 1, :] > anarr] = 0.0  # this would be hard in xr
        bb[1:-1, :] = np.where(esarr[:-1, :] > anarr, esarr[1:, :] - esarr[:-1, :], esarr[1:, :] - anarr)
        bb[1:-1, :][esarr[1:, :] <= anarr] = 0.0
        lcond = anarr > esarr[-1, :]
        aa[-1, :] = np.where(lcond, anarr - esarr[-1, :], 0.0)
        bb[-1, lcond] = 1.0
    cc = dict(fcst.coords)
    cc[over] = np.arange(nens + 1)
    alpha = xr.DataArray(aa, dims=fcst.dims, coords=cc, attrs=fcst.attrs)
    beta = xr.DataArray(bb, dims=fcst.dims, coords=cc, attrs=fcst.attrs)
    weight = xr.DataArray(np.arange(nens + 1), dims=over) / float(nens)
    crps = (alpha * weight**2 + beta * (1.0 - weight) ** 2).sum(over)
    if compute_fcrps:
        fcrps = crps - _ginis_mean_diff(fcst, stack_dim, over, "small") / (2.0 * nens)
    valid_mask = _common_valid_mask(obs, fcst, dim=over)

    if return_components:
        return (
            _unstack_arr(alpha.where(valid_mask)),
            _unstack_arr(beta.where(valid_mask)),
            _unstack_arr(crps.where(valid_mask)),
            _unstack_arr(fcrps.where(valid_mask)),
        )
    else:
        return _unstack_arr(crps.where(valid_mask))


# TODO: check if this belongs here
def ginis_mean_diff(fcst, over=None):
    valid_mask = _common_valid_mask(fcst, dim=over)
    fcst, stack_dim = _stack_arr(fcst, over)
    return _unstack_arr(_ginis_mean_diff(fcst, stack_dim, over, "small")).where(valid_mask)


def _cigns(mean, stdev, observations, stdev_ref=None, nens=None, normalize=True):
    import scipy

    def cb12(nens):
        """Compute correction to log score as defined in curly bracket in eqn (12) of Siegert et al (2015)."""
        z = 0.5 * (nens - 1.0)
        c = 0.5 * (scipy.special.digamma(z) - np.log(z) + 1.0 / nens)
        return c

    epsilon = 0.001
    if nens is None:
        fcigns = None
    else:
        if nens < 4:
            raise ValueError("To compute the fair logarithmic score the ensemble size must be at least 4.")
        zfactor = (nens - 3.0) / (nens - 1.0)  # see Siegert et al (2015), eqn (12)
    if stdev_ref is None:
        sig_ref = np.sqrt((stdev**2).mean())  # global standard deviation
        eps_sig_ref = max(epsilon * sig_ref, 1.0e-32)
        # nan_mask_cigns = nan_mask
    else:
        eps_sig_ref = epsilon * stdev_ref
        sig_ref = stdev_ref.where(stdev_ref != 0)
    sig = np.maximum(stdev, eps_sig_ref)
    rcv = (mean - observations) / sig  # reduced centred variable
    fac = np.sqrt(2.0 * np.pi)
    cigns_rel = 0.5 * rcv**2
    if normalize:
        cigns = cigns_rel + np.log(fac * sig / sig_ref)
        if nens is not None:
            fcigns = zfactor * cigns_rel + np.log(fac * sig / sig_ref) - cb12(nens)
    else:
        cigns = cigns_rel + np.log(fac * sig)
        if nens is not None:
            fcigns = zfactor * cigns_rel + np.log(fac * sig) - cb12(nens)
    return cigns, cigns_rel, fcigns


# TODO: double check naming - stdev reference (climatology)
# NB: this is not an event, just a baseline
def continuous_ignorance(fcst, obs, over, stdev_reference=None, normalize=True, return_components=True):
    assert return_components is True, "return_components=False not implemented yet"
    obs, _ = _stack_arr(obs, over)
    fcst, _ = _stack_arr(fcst, over)
    if stdev_reference is not None:
        stdev_reference, _ = _stack_arr(stdev_reference, over)
    mean = fcst.mean(dim=over)
    nens = fcst.sizes[over]
    stdev = np.sqrt(((fcst - mean) ** 2).mean(dim=over) * nens / (nens - 1.0))
    lsg, lsgrel, flsg = _cigns(mean, stdev, obs, stdev_reference, nens, normalize)
    valid_mask = _common_valid_mask(obs, fcst, dim=over)
    if stdev_reference is not None:
        valid_mask &= _common_valid_mask(stdev_reference, fcst, dim=over)
    return (
        _unstack_arr(lsg.where(valid_mask)),
        _unstack_arr(lsgrel.where(valid_mask)),
        _unstack_arr(flsg.where(valid_mask)),
    )


# TODO: check if we can use scores.continuous.quantile_score
def quantile_score(fcst, obs, tau, over):
    qf = fcst.quantile(tau, dim=over)
    # qf = numpy.nanpercentile(e, tau * 100., axis=0)
    qscore = abs(obs - qf) + (2.0 * tau - 1.0) * (obs - qf)
    return qscore


def _stats_to_quantiles(statistics, include_min_max=False, quantile_dim="quantile_"):
    # extract quantile array from the forecast statistics / climatology
    # optionally includes array of min/max
    # converts coordinates from "number:number_of_categories" to fraction (0.,1.)
    quantile_labels = statistics["quantile"][quantile_dim].values
    pvals_pc = [float(n) / float(nn) for n, nn in (p.split(":") for p in quantile_labels)]
    if include_min_max:
        assert (
            0.0 in pvals_pc
        ), "This computation requires the minimum values array to be included as a '0:nquant' quantile"
        assert (
            1.0 in pvals_pc
        ), "This computation requires the maximum values array to be included as a 'nquant:nquant' quantile"
    else:
        for x in (0.0, 1.0):
            try:
                ind = pvals_pc.index(x)
                del pvals_pc[ind]
                del quantile_labels[ind]
            except ValueError:
                pass
    pvals = np.array(pvals_pc)
    return statistics["quantile"].loc[{quantile_dim: quantile_labels}].assign_coords({quantile_dim: pvals})


def _nondistinct_mask(quantile, dim):
    # build nondistinct_mask which will mark points where the current quantile field equals to previous
    # or next quantiles (a provision for discrete variables like cloud cover)
    inner_mask_l = quantile[{dim: slice(0, -1)}] < quantile.shift(**{dim: -1})[{dim: slice(0, -1)}]
    inner_mask_u = quantile[{dim: slice(1, None)}] > quantile.shift(**{dim: 1})[{dim: slice(1, None)}]
    infimum = -np.inf < quantile[{dim: [0]}]
    supremum = quantile[{dim: [-1]}] < np.inf
    return xr.concat((infimum, inner_mask_u), dim=dim) & xr.concat((inner_mask_l, supremum), dim=dim)


# TODO: fill
def diagonal(
    fcst,
    obs,
    over,
    obs_reference_distribution=None,  # ek.GaussianDistribution(ds, mean_dim, std_dim), ek.QuantileDistribution(ds, type, dim)
    fcst_reference_distribution=None,
):
    # get quantiles from distributions, then compute as beforehand
    pass


# TODO: confirm we don't want from ctg_table
# TODO: check if want to be able to compute direct from probabilities
# TODO: implement (requires event implementation + distribution implementation)
# TODO: decide event API e.g.
# x >= ek.Distribution().median # medianthresholdevent
# x - ek.Distribution().mean >= threshold # equivalent to anomaly constantthresholdevent
# PercentileThresholdEvent(baseline=ek.Distribution, condition=">=", p=0.9)
# TercileThresholdEvent(baseline=ek.Distribution, condition=">=", tercile="upper")
# MedianThresholdEvent(baseline=ek.Distribution, condition=">=")
# MeanThresholdEvent(baseline=ek.Distribution, condition=">=")
# ConstantThresholdEvent(condition=">=", threshold=value)
# AnomalyConstantThresholdEvent(condition=">=", threshold=value)
# TODO: decide if should be Anomaly(ConstantThresholdEvent) instead
# StandardDeviationThresholdEvent(baseline=ek.Distribution, condition=">=", n_stdev=1.5)
# AND NB: also anomaly versions, which means first subtract climatology/baseline mean from data
# TODO: can we decide events?
def rps(fcst, obs, over, events):
    pass


# def rps(
#     observations,
#     forecasts,
#     observation_climatology=None,
#     forecast_climatology=None,
#     thr_type="percentile",
#     ncat=10,
#     thr_max=None,
#     dim=None,
# ):
#     pass


def rank_histogram(*args, **kwargs):
    pass


# climatology


def crps_gaussian(*args, **kwargs):
    pass


def crps_quadrature(*args, **kwargs):
    pass


def continuous_ignorance_gaussian(*args, **kwargs):
    pass


def diagonal_quadrature(*outargs, **kwargs):
    pass


# def brier_gaussian(*args, **kwargs):
#     pass


# def brier_quadrature(*args, **kwargs):
#     pass


def ignorance_gaussian(*args, **kwargs):
    pass


def ignorance_quadrature(*args, **kwargs):
    pass


def rps_gaussian(*args, **kwargs):
    pass


def rps_quadrature(*args, **kwargs):
    pass


# event masks & ctg tables


def event_occurrences(*args, **kwargs):
    pass


def contingency_table(
    event,
    observations,
    forecasts,
    observation_climatology=None,
    forecast_climatology=None,
    dim=None,
    ct_dim="ctable",
):
    from .event import event_field

    if observation_climatology is not None:
        observation_climatology, _ = _stack_arr(observation_climatology, dim, order=observations.dims)
    if forecast_climatology is not None:
        forecast_climatology, _ = _stack_arr(forecast_climatology, dim, order=forecasts.dims)
    if observation_climatology is None:
        observation_climatology = forecast_climatology
    if forecast_climatology is None:
        forecast_climatology = observation_climatology
    observations, _ = _stack_arr(observations, dim)
    forecasts, stacked_dim = _stack_arr(forecasts, dim)
    ev_an, an_mask = event_field(event, observations, observation_climatology, dim=dim)
    ev_fc, fc_mask = event_field(event, forecasts, forecast_climatology, dim=dim)
    nens = ev_fc.sizes[dim]
    rnk0 = ev_fc.sum(dim=dim)
    # use rnk0 to decide which element of cnoc /cocc is updated
    where_observed = ev_an.values
    # cocc and cnoc could be just views to ct
    # "Positional indexing with only integers and slices returns a view."
    ct = xr.concat([xr.zeros_like(forecasts[{dim: 0}])] * 2 * (nens + 1), dim=ct_dim).assign_coords(
        {ct_dim: range(1, 2 * (nens + 1) + 1)}
    )
    cocc = ct[{ct_dim: slice(0, nens + 1)}]
    cnoc = ct[{ct_dim: slice(nens + 1, None)}]
    # xarray indexing works slightly differently from numpy
    # ind_rnk must be a DataArray with the dimension FIELD to make sure
    # the following indexing is vectorized (returns ind_rnk where where_observed)
    # and not orthogonal (which would return a region ind_rnk x where_observed)
    cocc[{ct_dim: rnk0[where_observed], stacked_dim: where_observed}] = 1.0
    cnoc[{ct_dim: rnk0[~where_observed], stacked_dim: ~where_observed}] = 1.0
    valid_mask = an_mask & fc_mask  # it includes optionally the nan-mask of climatology as well
    return _unstack_arr(ct.where(valid_mask))


# from ctg tables


def brier_from_ctg_table(contingency_table, over):
    pass


def brier_from_ensemble(fcst, obs, event_thresholds=0.5, ensemble_member_dim="ensemble"):
    pass


def brier_from_cdf(fcst_cdf, obs, event_thresholds=0.5, quantile_dim="quantile_"):  # quadrature
    pass


def brier_gaussian(event, observations, forecast_statistics):
    pass


def area_under_roc(*args, **kwargs):
    pass


def brier(*args, **kwargs):
    pass


def ignorance(*args, **kwargs):
    pass


def equitable_threat_score(*args, **kwargs):
    pass


def frequency_bias(*args, **kwargs):
    pass


def peirce_skill_score(*args, **kwargs):
    pass


def symmetric_extremal_dependence_index(*args, **kwargs):
    pass


# from crps


def crps_potential(*args, **kwargs):
    pass


def crps_reliability(*args, **kwargs):
    pass
