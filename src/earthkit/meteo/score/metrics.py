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


# TODO: add return_components option
# TODO: double check naming - stdev reference (climatology)
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


# TODO: fill
def quantile_score(*args, **kwargs):
    pass


# TODO: fill
def diagonal(*args, **kwargs):
    pass


def _event_field(event, field, statistics=None, dim=None):
    # dim solely indicates which dimension is the ensemble when `field` is an ensemble
    # TODO: check if this actually works for "abs" when field has coordinates along `dim` dimension?
    from .event import BinaryEvent

    bev = BinaryEvent(**event)
    if bev.requires_climatology and statistics is None:
        raise ValueError("Binary event %s requires statistics/climatology but none is present" % bev)
    if bev["type"] == "abs":
        if dim is not None and dim in field.dims:
            threshold = xr.full_like(field[{dim: 0}], bev["value"])
        else:
            threshold = xr.full_like(field, bev["value"])
    elif bev["type"] == "stdev":
        threshold = bev["value"] * statistics["stdev"]
    else:
        threshold = bev.select_quantile_array(statistics["quantile"])
    if bev["is_anomaly"]:
        threshold = threshold + statistics["mean"]
    valid_mask = _common_valid_mask(field, threshold, dim=dim)
    return bev.operator()(field, threshold), valid_mask


def _generate_rps_events(thr_type, ncat, thr_max=None):
    from .event import BinaryEvent

    if thr_type == "percentile":
        for jcat in range(ncat - 1):
            p_thr = float(jcat + 1) / float(ncat)  # 1/ncat ... (ncat-1)/ncat
            yield BinaryEvent(
                value=int(100.0 * p_thr + 0.5),
                type="percentile",
                operator=">",
                is_anomaly=False,
            )
    elif thr_type == "tercile":
        for jcat in (1, 2):
            yield BinaryEvent(value=jcat, type="tercile", operator=">", is_anomaly=False)
    elif thr_type == "stdev":
        for jcat in range(ncat - 1):
            if ncat == 20:
                xthr = 1.64 * (2 * jcat - ncat + 2) / float(ncat - 2)
            elif ncat == 10:
                xthr = 1.28 * (2 * jcat - ncat + 2) / float(ncat - 2)
            elif ncat == 5:
                xthr = 0.84 * (2 * jcat - ncat + 2) / float(ncat - 2)
            else:
                xthr = 1.5 * (2 * jcat - ncat + 2) / float(ncat - 2)
            yield BinaryEvent(value=xthr, type="stdev", operator=">", is_anomaly=False)
    elif thr_type == "abs":
        if thr_max is None:
            raise ValueError("thr_max is required if type is abs")
        for jcat in range(ncat - 1):
            xthr = thr_max * (2 * jcat - ncat + 2) / float(ncat - 2)
            yield BinaryEvent(value=xthr, type="abs", operator=">", is_anomaly=False)
    else:
        raise ValueError("threshold type " + thr_type + " not defined")


# TODO: this was directly copy-pasted, decide on API
def rps(
    observations,
    forecasts,
    observation_climatology=None,
    forecast_climatology=None,
    thr_type="percentile",
    ncat=10,
    thr_max=None,
    dim=None,
):
    if observation_climatology is not None:
        observation_climatology, _ = _stack_arr(observation_climatology, dim)
    if forecast_climatology is not None:
        forecast_climatology, _ = _stack_arr(forecast_climatology, dim)
    if observation_climatology is None:
        observation_climatology = forecast_climatology
    if forecast_climatology is None:
        forecast_climatology = observation_climatology
    observations, _ = _stack_arr(observations, dim)
    forecasts, _ = _stack_arr(forecasts, dim)
    rps = xr.zeros_like(forecasts[{dim: 0}])
    counter = 0
    masks = True
    for be in _generate_rps_events(thr_type, ncat, thr_max):
        obs_events, mask_ob = _event_field(be, observations, observation_climatology, dim=dim)
        ens_events, mask_fc = _event_field(be, forecasts, forecast_climatology, dim=dim)
        masks &= mask_ob & mask_fc
        # probability
        p = ens_events.mean(dim)
        bs = (p - obs_events) ** 2
        rps += bs
        counter += 1
    rps /= float(counter + 1)
    return _unstack_arr(rps.where(masks))


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


def brier_gaussian(*args, **kwargs):
    pass


def brier_quadrature(*args, **kwargs):
    pass


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


def contingency_table(*args, **kwargs):
    pass


# from ctg tables


def roc_curve(*outargs, **kwargs):
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
