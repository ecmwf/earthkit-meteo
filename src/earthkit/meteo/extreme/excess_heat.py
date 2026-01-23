# (C) Copyright 2025 - ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import functools
import numbers

import earthkit.transforms._tools as _ekt_tools
import earthkit.transforms.temporal
import numpy as np
import xarray as xr


class _with_metadata:
    """Decorator to attach metadata to an output DataArray"""

    # TODO just a quick solution until something better is in place
    # TODO input-dependent unit handling (take input unit and transform)

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


__DMT_TIME_SHIFT_COORD = "__daily_mean_temperature_time_shift"


@_with_metadata("dmt", long_name="Daily mean temperature")
def daily_mean_temperature(t2m, day_start=9, time_shift=0, **kwargs):
    """Daily mean temperature, computed from min and max.

    Recommended to install flox for efficient aggregations.

    Parameters
    ----------
    t2m : xr.DataArray
        2-metre temperature.
    day_start : number | np.timedelta64
        Constant offset for the start of the day in the aggregations. By
        default, the day is defined from 09:00 to 09:00. Positive offsets
        indicate a late start of the day (see default), negative an early
        start. Numeric values are interpreted as hours.
    time_shift : np.timedelta64 | number | str | xr.DataArray
        Numeric values are interpreted as hours. Provide a string value to
        take from DataArray coordinates.
    **kwargs
        Keyword arguments for the daily_min and daily_max functions of
        earthkit.transforms.temporal.
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
            # dtype. The period dtype works if the same partial days are
            # removed.
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


@_with_metadata("ehi_sig", long_name="Significance index")
def significance_index(dmt, threshold=("quantile", 0.95), ndays=3, time_dim=None):
    """Significance index

    Parameters
    ----------
    dmt : xr.DataArray
        Daily mean temperature.
    threshold : number
        TODO
    ndays : number
        Length of evaluation time window.
    time_dim : None | str
        Name of time dimension in dmt DataArray.

    Returns
    -------
    xr.DataArray
        Significance index.
    """
    # Time coordinate detection compatible with earthkit.transforms
    if time_dim is None:
        time_dim = _ekt_tools.get_dim_key(dmt, "t", raise_error=True)
    # Compute threshold as quantile
    if isinstance(threshold, tuple):
        assert len(threshold) == 2
        if threshold[0] == "quantile":
            threshold = dmt.quantile(threshold[1], dim=time_dim)
        else:
            raise NotImplementedError
    # TODO: also support day-of-year climatology to detect warm spells
    current = _rolling_mean(dmt, ndays, shift_days=(1 - ndays))
    return current - threshold


@_with_metadata("ehi_accl", long_name="acclimatisation_index")
def acclimatisation_index(dmt, ndays=3, ndays_ref=30):
    """Acclimatisation index

    Parameters
    ----------
    dmt : xr.DataArray
        Daily mean temperature.
    ndays : number
        Length of evaluation time window.
    ndays_ref : number
        Length of reference time window (recent past).
    """
    current = _rolling_mean(
        dmt, ndays, shift_days=(1 - ndays)
    )  # TODO: shared with significance index, would be nice to not compute it twice
    reference = _rolling_mean(dmt, ndays_ref, shift_days=1)
    return current - reference


# https://codes.ecmwf.int/grib/param-db/261024
@_with_metadata("exhf", long_name="Excess heat factor")
def excess_heat_factor(ehi_sig, ehi_accl, nonnegative=True):
    """Excess heat factor

    Parameters
    ----------
    ehi_sig : xr.DataArray | array_like
        Significance index.
    ehi_accl : xr.DataArray | array_like
        Acclimatisation index.
    nonnegative : bool
        Whether to clip the lower value range by zero.
    """
    if nonnegative:
        ehi_sig = np.maximum(0, ehi_sig)
    return ehi_sig * np.maximum(1.0, ehi_accl)


# https://codes.ecmwf.int/grib/param-db/261025
@_with_metadata("excf", long_name="Excess cold factor")
def excess_cold_factor(ehi_sig, ehi_accl, nonnegative=True):
    return NotImplementedError  # TODO
