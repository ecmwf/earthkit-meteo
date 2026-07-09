# (C) Copyright 2025- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import functools
import numbers

import earthkit.transforms.climatology
import earthkit.transforms.temporal
import thermofeel.excess_heat
import xarray as xr


class _with_metadata:
    """Decorator to attach metadata to an output DataArray."""

    # TODO just a quick solution until something better is in place
    # TODO input-dependent unit handling (take input unit(s), check compatibility, transform into output unit)

    def __init__(self, name, **attrs):
        self.name = name
        self.attrs = attrs

    def __call__(self, f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs).rename(self.name).assign_attrs(self.attrs)

        return wrapped


def _rolling_mean(da, n, shift_days=0):
    return earthkit.transforms.temporal.rolling_reduce(
        da, n, center=False, how_reduce="mean", time_shift={"days": shift_days}, how_dropna="any"
    )


def _threshold_as_quantile(da, q):
    return earthkit.transforms.climatology.reduce(da, how="quantile", q=q).drop_vars("quantile")


@_with_metadata("dmt", long_name="Daily mean temperature")  # TODO units
def daily_mean_temperature(t2m, **kwargs):
    r"""Daily mean temperature, computed from min and max.

    Supports custom definitions of "day" (parameter `day_start`) and accounts
    for local time zones when time zone offsets with respect to the time
    coordinate of the input data as given a function of the spatial coordinates.

    Parameters
    ----------
    t2m : xarray.DataArray
        2-metre temperature.
    **kwargs
        Keyword arguments for the daily_min and daily_max functions of
        earthkit.transforms.temporal. Set a `time_shift` to accomodate time
        zones and a local definition of the day. Set `remove_partial_periods`
        together with `time_shift` to remove days with incomplete data at the
        beginning and end.

    Returns
    -------
    xarray.DataArray
        Daily mean temperature.

    Notes
    -----
    The daily mean temperature is defined as

    .. math::

        T({t_i}) = \frac{T_\mathrm{min}(t_i) + T_\mathrm{max}(t_i)}{2},

    where

    - :math:`T_\mathrm{min}(t_i)` is the daily minimum temperature and
    - :math:`T_\mathrm{max}(t_i)` is the daily maximum temperature,

    over a local definition of "day". E.g., [Nairn2014]_ define the day from
    9 am to 9 am local time, so that the daily maximum typically preceeds the
    daily minimum to account for the greater significance of the human
    physiological response to a hot night following a hot day compared to the
    other way around.

    .. tip::
        It is recommended to install flox to improve the computational effiency
        when working with chunked data in dask.
    """
    return thermofeel.excess_heat.daily_mean_temperature(
        t2_min=earthkit.transforms.temporal.daily_min(t2m, **kwargs),
        t2_max=earthkit.transforms.temporal.daily_max(t2m, **kwargs),
    )


@_with_metadata("ehi_sig", long_name="Significance index")  # TODO units
def significance_index(dmt, ndays=3, threshold=None, threshold_q=0.95):
    r"""Excess heat significance index.

    Supports both fixed thresholds to identify heat and cold waves and
    day-of-year climatologies to identify warm and cold spells.

    Parameters
    ----------
    dmt : xarray.DataArray
        Daily mean temperature.
    ndays : int, optional
        Length of evaluation time window. 3 days by default.
    threshold : xarray.DataArray | number | None, optional
        Significance threshold. If no threshold is given, a quantile of the
        values of the input excess heat factor timeseries is computed and used.
    threshold_q : number, optional
        The quantile used for the threshold if none is given.

    Returns
    -------
    xarray.DataArray
        Significance index.

    See Also
    --------
    :py:func:`daily_mean_temperature`
    :py:func:`acclimatisation_index`

    Notes
    -----
    The significance index is defined as

    .. math::

        EHI_{sig} = \frac{T(t_{i}) + \ldots + T(t_{i+n-1})}{n} - T_{95},

    where

    - :math:`T` is daily mean temperature,
    - :math:`t_i` denotes timestep :math:`i`,
    - :math:`n` is the number of timesteps in the evaluation window (`ndays`), and
    - :math:`T_{95}` is the threshold of significance for the daily mean
      temperature.

    Using the 95th percentile of daily mean temperature over a reference period
    as the default threshold follows [Nairn2014]_.
    """
    current = _rolling_mean(dmt, ndays, shift_days=(1 - ndays))
    if threshold is None:
        threshold = _threshold_as_quantile(dmt, threshold_q)
    if isinstance(threshold, numbers.Number):
        threshold = xr.DataArray(threshold)
    return earthkit.transforms.climatology.anomaly(current, threshold)


@_with_metadata("ehi_accl", long_name="Acclimatisation index")  # TODO units
def acclimatisation_index(dmt, ndays=3, ndays_ref=30):
    r"""Excess heat acclimatisation index.

    Parameters
    ----------
    dmt : xarray.DataArray
        Daily mean temperature.
    ndays : int, optional
        Length of evaluation time window. 3 days by default.
    ndays_ref : int, optional
        Length of reference time window (recent past). 30 days by default.

    Returns
    -------
    xarray.DataArray
        Acclimatisation index.

    See Also
    --------
    :py:func:`daily_mean_temperature`
    :py:func:`significance_index`

    Notes
    -----
    The acclimatisation index is defined as

    .. math::

        EHI_{accl}(t_i) = \frac{T(t_{i}) + \ldots + T(t_{i+n-1})}{n} - \frac{T(t_{i-m}) + \ldots + T(t_{i-1})}{m}

    where

    - :math:`T` is daily mean temperature,
    - :math:`t_i` denotes timestep :math:`i`,
    - :math:`n` is the number of timesteps in the evaluation window (`ndays`), and
    - :math:`m` is the number of timesteps in the reference time window (`ndays_ref`).

    The default time window lengths reflect the configuration of [Nairn2014]_.
    """
    return thermofeel.excess_heat.acclimatisation_index(
        dmt=_rolling_mean(dmt, ndays, shift_days=(1 - ndays)), threshold=_rolling_mean(dmt, ndays_ref, shift_days=1)
    )


# https://codes.ecmwf.int/grib/param-db/261024
# TODO: input unit checks
@_with_metadata("exhf", long_name="Excess heat factor", units="K²")
def excess_heat_factor(ehi_sig, ehi_accl, clip=False):
    r"""Excess heat factor.

    Parameters
    ----------
    ehi_sig : xarray.DataArray
        Significance index.
    ehi_accl : xarray.DataArray
        Acclimatisation index.
    clip : bool, optional
        Whether to clip the lower value range at zero. Disabled by default.

    Returns
    -------
    xarray.DataArray
        Excess heat factor.

    See Also
    --------
    :py:func:`significance_index`
    :py:func:`acclimatisation_index`
    :py:func:`excess_cold_factor`

    Notes
    -----
    The excess heat factor is defined as

    .. math::

        EXHF = EHI_{sig} \times \max(1, EHI_{accl}),

    where

    - :math:`EHI_{sig}` is the excess heat index of significance and
    - :math:`EHI_{accl}` is the excess heat index of acclimatisation.

    Example
    -------
    :ref:`Nairn and Fawcett (2014) <Nairn2014>` compute the excess heat factor
    with :math:`EHI_{sig}` relative to the 95th percentile of a 30-year
    climatology of daily mean temperature and :math:`EHI_{accl}` relative to the
    30 days directly preceeding the valid time. The authors use an evaluation
    time window of 3 days starting from the valid day for both indices.
    """
    return thermofeel.excess_heat.excess_heat_factor(ehi_sig, ehi_accl, clip=clip)


# TODO: record threshold in provenance
@_with_metadata("hsev", long_name="Heatwave severity", units="1")
def heatwave_severity(exhf, threshold=None, threshold_q=0.85):
    r"""Heatwave severity index.

    Parameters
    ----------
    exhf : xarray.DataArray
        Excess heat factor.
    threshold : xarray.DataArray | number | None, optional
        Excess heat factor threshold. If no threshold is given, a quantile of
        the *positive* values of the input excess heat factor timeseries is
        computed and used.
    threshold_q : number, optional
        The quantile used for the threshold if none is given.

    Returns
    -------
    xarray.DataArray
        Heatwave severity index.

    See Also
    --------
    :py:func:`excess_heat_factor`

    Notes
    -----
    The heatwave severity index is defined as

    .. math::

        HSEV = \frac{EXHF}{EXHF_{85}},

    where

    - :math:`EXHF` is the excess heat factor and
    - :math:`EXHF_{85}` is a threshold of the excess heat factor.

    Using the 85th percentile of all positive values of the excess heat factor
    over a reference period as the default threshold follows [Nairn2018]_.
    """
    if threshold is None:
        threshold = _threshold_as_quantile(exhf.where(exhf > 0), threshold_q)
    return thermofeel.excess_heat.heatwave_severity(exhf, threshold)


# https://codes.ecmwf.int/grib/param-db/261025
# TODO: input unit checks
@_with_metadata("excf", long_name="Excess cold factor", units="K²")
def excess_cold_factor(ehi_sig, ehi_accl, clip=False):
    r"""Excess cold factor.

    Parameters
    ----------
    ehi_sig : xarray.DataArray
        Significance index.
    ehi_accl : xarray.DataArray
        Acclimatisation index.
    clip : bool, optional
        Whether to clip the upper value range at zero. Disabled by default.

    Returns
    -------
    xarray.DataArray
        Excess cold factor.

    See Also
    --------
    :py:func:`significance_index`
    :py:func:`acclimatisation_index`
    :py:func:`excess_heat_factor`

    Notes
    -----
    The excess cold factor is defined as

    .. math::

        EXCF = -EHI_{sig} \times \min(-1, EHI_{accl}),

    where

    - :math:`EHI_{sig}` is the excess heat index of significance,
    - :math:`EHI_{accl}` is the excess heat index of acclimatisation.

    Example
    -------
    [Nairn2013]_ compute the excess cold factor with :math:`EHI_{sig}` relative
    to the 5th percentile of a 30-year climatology of daily mean temperature
    and :math:`EHI_{accl}` relative to the 30 days directly preceeding the valid
    time. The authors use an evaluation time window of 3 days starting from the
    valid day for both indices.
    """
    return thermofeel.excess_heat.excess_cold_factor(ehi_sig, ehi_accl, clip=clip)
