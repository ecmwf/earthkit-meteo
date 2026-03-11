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
import numpy as np
import xarray as xr


class _with_metadata:
    """Decorator to attach metadata to an output DataArray"""

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


def _compute_threshold_as_quantile(da, q):
    return earthkit.transforms.temporal.reduce(da, how="quantile", q=q).drop_vars("quantile")


__DMT_TIME_SHIFT_COORD = "__daily_mean_temperature_time_shift"


@_with_metadata("dmt", long_name="Daily mean temperature")  # TODO units
def daily_mean_temperature(t2m, day_start=9, time_shift=0, **kwargs):
    """Daily mean temperature, computed from min and max.

    Supports custom definitions of "day" (parameter `day_start`) and accounts
    for local time zones when time zone offsets with respect to the time
    coordinate of the input data as given a function of the spatial coordinates.

    Parameters
    ----------
    t2m : xarray.DataArray
        2-metre temperature.
    day_start : number | numpy.timedelta64
        Constant offset for the start of the day in the aggregations. By
        default, the day is defined from 09:00 to 09:00. Positive offsets
        indicate a late start of the day, negative an early start. A numeric
        value is interpreted as hours.
    time_shift : xarray.DataArray | numpy.timedelta64 | number | str
        Offset relative to time coordinate of input. Can vary in space to
        specify timezones. A numeric value is interpreted as hours. Provide a
        string value to take a correspondingly named coordinate from the input
        DataArray.
    **kwargs
        Keyword arguments for the daily_min and daily_max functions of
        earthkit.transforms.temporal.

    Returns
    -------
    xarray.DataArray
        Daily mean temperature.

    Notes
    -----
    The daily mean temperature is defined as

    .. math::

        T({t_i}) = \\frac{T_\\mathrm{min}(t_i) + T_\\mathrm{max}(t_i)}{2},

    where

    - :math:`T_\\mathrm{min}(t_i)` is the daily minimum temperature and
    - :math:`T_\\mathrm{max}(t_i)` is the daily maximum temperature,

    over a local definition of "day". E.g., [Nairn2014]_ define the day from
    9 am to 9 am local time, so that the daily maximum typically preceeds the
    daily minimum to account for the greater significance of the human
    physiological response to a hot night following a hot day compared to the
    other way around. The default value of `day_start` reflects this definition.

    .. tip::
        It is recommended to install flox to improve the computational effiency
        when working with chunked data in dask.

    Example
    -------
    Assume that the time coordinate of the data is given in UTC and we want to
    define the day from 10:00 to 10:00 Atlantic Standard Time (UTC -4 hours),
    then we need to call::

    >>> daily_mean_temperature(..., day_start=10, time_shift=-4)

    Note that while both offest parameters lead to a "later" start of the day
    relative to the reference time coordinate of the data, the sign of their
    value is different to match their use as outlined in the text above. Set
    `day_start` to zero and integrate the start offset into `time_shift` if
    you prefer to specify only a single offset for both settings. The above
    call is equivalent to::

    >>> daily_mean_temperature(..., day_start=0, time_shift=-14)
    """
    assert isinstance(t2m, xr.DataArray)

    if isinstance(day_start, numbers.Number):
        day_start = np.timedelta64(day_start, "h")
    assert isinstance(day_start, np.timedelta64)

    if isinstance(time_shift, numbers.Number):
        time_shift = np.timedelta64(time_shift, "h")
    if isinstance(time_shift, str):
        time_shift = t2m.coords[time_shift]
    if isinstance(time_shift, xr.DataArray):
        unique_shifts = np.unique(time_shift.values)
        # Can only proceed if all timeseries are in the same time zone. If
        # there are multiple time zones: split, process separately, merge.
        if unique_shifts.size == 1:
            time_shift = unique_shifts[0]
        else:
            assert __DMT_TIME_SHIFT_COORD not in t2m.coords
            # Merging of groups where different partial days were removed
            # fails after map when the time coordinate is of pandas period
            # dtype. The period dtype only works when the same partial days
            # were removed in all groups.
            return (
                t2m
                # Groups don't know their time shift unless we attach it here.
                # Grouping by the time shift means the shift value per group
                # is unique and the recursion ends immediately.
                .assign_coords({__DMT_TIME_SHIFT_COORD: time_shift})
                .groupby(__DMT_TIME_SHIFT_COORD)
                .map(daily_mean_temperature, day_start=day_start, time_shift=__DMT_TIME_SHIFT_COORD, **kwargs)
                # Don't expose the internal shift coordinate to the user
                .drop_vars(__DMT_TIME_SHIFT_COORD)
            )
    time_shift = np.asarray(time_shift)
    assert time_shift.size == 1
    assert np.issubdtype(time_shift.dtype, np.timedelta64)

    agg_kwargs = {"time_shift": time_shift - day_start, "remove_partial_periods": True, **kwargs}
    tmin = earthkit.transforms.temporal.daily_min(t2m, **agg_kwargs)
    tmax = earthkit.transforms.temporal.daily_max(t2m, **agg_kwargs)
    return 0.5 * (tmin + tmax)


@_with_metadata("ehi_sig", long_name="Significance index")  # TODO units
def significance_index(dmt, ndays=3, threshold=None):
    """Excess heat significance index.

    Supports both fixed thresholds to identify heat and cold waves and
    day-of-year climatologies to identify warm and cold spells.

    Parameters
    ----------
    dmt : xarray.DataArray
        Daily mean temperature.
    ndays : int, optional
        Length of evaluation time window. 3 days by default.
    threshold : xarray.DataArray | number | None, optional
        Significance threshold. By default, the gridpoint-wise 95th percentile
        of the input daily mean temperature timeseries is computed and used.

    Returns
    -------
    xarray.DataArray
        Significance index.

    Notes
    -----
    The significance index is defined as

    .. math::

        EHI_{sig} = \\frac{T(t_{i}) + \\ldots + T(t_{i+n-1})}{n} - T_{95},

    where

    - :math:`T` is daily mean temperature,
    - :math:`t_i` denotes timestep :math:`i`,
    - :math:`n` is the number of timesteps in the evaluation window (`ndays`), and
    - :math:`T_{95}` is the threshold of significance for the daily mean
      temperature.

    The 95th percentile as the default threshold follows [Nairn2014]_, except
    that they use a fixed reference period to compute the percentile over rather
    than the full timeseries.

    See also
    --------
    :py:func:`daily_mean_temperature`
    :py:func:`acclimatisation_index`
    """
    if threshold is None:
        threshold = _compute_threshold_as_quantile(dmt, 0.95)
    current = _rolling_mean(dmt, ndays, shift_days=(1 - ndays))
    # TODO: earthkit-transforms also supports weekly and monthly climatologies,
    #       make them work too, ideally without hardcoding coordinate names
    if isinstance(threshold, xr.DataArray) and "dayofyear" in threshold.coords:
        # Support warm/cold spells with threshold as a function of day-of-year
        return earthkit.transforms.climatology.anomaly(current, threshold)
    return current - threshold


@_with_metadata("ehi_accl", long_name="Acclimatisation index")  # TODO units
def acclimatisation_index(dmt, ndays=3, ndays_ref=30):
    """Excess heat acclimatisation index.

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

    Notes
    -----
    The acclimatisation index is defined as

    .. math::

        EHI_{accl}(t_i) = \\frac{T(t_{i}) + \\ldots + T(t_{i+n-1})}{n} - \\frac{T(t_{i-m}) + \\ldots + T(t_{i-1})}{m}

    where

    - :math:`T` is daily mean temperature,
    - :math:`t_i` denotes timestep :math:`i`,
    - :math:`n` is the number of timesteps in the evaluation window (`ndays`), and
    - :math:`m` is the number of timesteps in the reference time window (`ndays_ref`).

    The default time window lengths reflect the configuration of [Nairn2014]_.

    See also
    --------
    :py:func:`daily_mean_temperature`
    :py:func:`significance_index`
    """
    current = _rolling_mean(dmt, ndays, shift_days=(1 - ndays))
    reference = _rolling_mean(dmt, ndays_ref, shift_days=1)
    return current - reference


# https://codes.ecmwf.int/grib/param-db/261024
# TODO: input unit checks
@_with_metadata("exhf", long_name="Excess heat factor", units="K²")
def excess_heat_factor(ehi_sig, ehi_accl, nonnegative=False):
    """Excess heat factor.

    Parameters
    ----------
    ehi_sig : xarray.DataArray
        Significance index.
    ehi_accl : xarray.DataArray
        Acclimatisation index.
    nonnegative : bool, optional
        Whether to clip the lower value range at zero. Disabled by default.

    Returns
    -------
    xarray.DataArray
        Excess heat factor.

    Notes
    -----
    The excess heat factor is defined as

    .. math::

        EXHF = EHI_{sig} \\times \\max(1, EHI_{accl}),

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

    See also
    --------
    :py:func:`significance_index`
    :py:func:`acclimatisation_index`
    :py:func:`excess_cold_factor`
    """
    if nonnegative:
        ehi_sig = np.maximum(0, ehi_sig)
    return ehi_sig * np.maximum(1.0, ehi_accl)


# TODO: record threshold in provenance
@_with_metadata("hsev", long_name="Heatwave severity", units="1")
def heatwave_severity(exhf, threshold=None):
    """Heatwave severity index.

    Parameters
    ----------
    exhf : xarray.DataArray
        Excess heat factor.
    threshold : xarray.DataArray | number | None, optional
        Excess heat factor threshold. By default, the gridpoint-wise 85th
        percentile of the input excess heat factor timeseries is computed and
        used.

    Returns
    -------
    xarray.DataArray
        Heatwave severity index.

    Notes
    -----
    The heatwave severity index is defined as

    .. math::

        HSEV = \\frac{EXHF}{EXHF_{85}},

    where

    - :math:`EXHF` is the excess heat factor and
    - :math:`EXHF_{85}` is a threshold of the excess heat factor.

    The 85th percentile as the default threshold follows [Nairn2018]_, except
    that they use a fixed reference period to compute the percentile over rather
    than the full timeseries.

    See also
    --------
    :py:func:`excess_heat_factor`
    """
    if threshold is None:
        threshold = _compute_threshold_as_quantile(exhf, 0.85)
    return exhf / threshold


# https://codes.ecmwf.int/grib/param-db/261025
# TODO: input unit checks
@_with_metadata("excf", long_name="Excess cold factor", units="K²")
def excess_cold_factor(ehi_sig, ehi_accl, nonpositive=False):
    """Excess cold factor.

    Parameters
    ----------
    ehi_sig : xarray.DataArray
        Significance index.
    ehi_accl : xarray.DataArray
        Acclimatisation index.
    nonpositive : bool, optional
        Whether to clip the upper value range at zero. Disabled by default.

    Returns
    -------
    xarray.DataArray
        Excess cold factor.

    Notes
    -----
    The excess cold factor is defined as

    .. math::

        EXCF = -EHI_{sig} \\times \\min(-1, EHI_{accl}),

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

    See also
    --------
    :py:func:`significance_index`
    :py:func:`acclimatisation_index`
    :py:func:`excess_heat_factor`
    """
    if nonpositive:
        ehi_sig = np.minimum(0, ehi_sig)
    return -ehi_sig * np.minimum(-1.0, ehi_accl)
