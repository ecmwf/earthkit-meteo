"""
spread
quantile_score

crps_from_ensemble
crps_from_gaussian
crps_from_cdf

continuous_ignorance (maybe)
"""

from typing import TypeVar

import xarray as xr

T = TypeVar("T", xr.DataArray, xr.Dataset)


def import_scores_or_prompt_install():
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
        nowrap::

        \begin{align*}
        q_i & = \text{Quantile of the forecast at level } \tau \\
        qs_i = |o_i - q_i| + (2 \tau - 1) (o_i - q_i)
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
    # qf = numpy.nanpercentile(e, tau * 100., axis=0)
    qscore = abs(obs - qf) + (2.0 * tau - 1.0) * (obs - qf)
    return qscore


# def crps_from_ensemble

# def crps_from_gaussian

# def crps_from_cdf
