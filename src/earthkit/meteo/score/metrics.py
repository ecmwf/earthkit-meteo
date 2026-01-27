import numpy as np


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


# TODO: check if should support ddof argument (current implementation is ddof=0, like numpy)
def standard_deviation_of_error(fcst, obs, over, weights=None, is_angular=False):
    assert is_angular is False, "is_angular=True is not supported for standard_deviation_of_error"
    error = fcst - obs
    return _weighted_mean((error - _weighted_mean(error, weights, over)) ** 2, weights, over).sqrt()


def _common_valid_mask(*arrays, dim=None):
    # return a numpy bool array of occurrences of all values valid across dim
    # return xarray.concat(arrays, dim=dim).notnull().all(dim=dim)
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
def spread(*args, **kwargs):
    pass


def spread_squared(*args, **kwargs):
    pass


def crps(*args, **kwargs):
    pass


def ginis_mean_diff(*args, **kwargs):
    pass


def continuous_ignorance_score(*args, **kwargs):
    pass


def quantile_score(*args, **kwargs):
    pass


def diagonal(*args, **kwargs):
    pass


def rps(*args, **kwargs):
    pass


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
