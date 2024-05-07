import numpy as np


def correlation(obs, fc, axis=None, unbiased=True, weights=None):
    """
    Calculate the correlation between the observations and the forecast.

    Parameters:
    obs: list of floats
        The observations.
    fc: list of floats
        The forecast.
    axis: int
        The axis along which the correlation is calculated.
    unbiased: bool
        If True, the correlation is unbiased.
    weights: list of floats
        The weights used to compute the average along the given axis.
    """

    fc_mean = np.average(fc, axis=axis, weights=weights)
    ob_mean = np.average(fc, axis=axis, weights=weights)
    fc_sq = fc**2
    ob_sq = obs**2
    fc_std_sq = np.average(fc_sq, axis=axis, weights=weights)
    ob_std_sq = np.average(ob_sq, axis=axis, weights=weights)
    corr = np.average(fc*obs, axis=axis, weights=weights)
    if unbiased:
        fc_std_sq = fc_std_sq - fc_mean**2
        ob_std_sq = ob_std_sq - ob_mean**2
        corr = corr - fc_mean * ob_mean
    result = corr / np.sqrt(fc_std_sq * ob_std_sq)
    return result
