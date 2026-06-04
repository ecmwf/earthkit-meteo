# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import dataclasses as dc

import numpy as np
from scipy import interpolate

from earthkit.meteo import thermo
from earthkit.meteo.constants import constants

C_pl = 4218.0


@dc.dataclass(frozen=True)
class PressureLevel:
    """Pressure, temperature and height at a single key level.

    All arrays have the shape of the horizontal dimensions of the input
    (i.e. the vertical axis is absent).  Values are NaN where the level does
    not exist (e.g. no LFC in a stable profile).
    """

    p: np.ndarray
    """Pressure (Pa)."""
    t: np.ndarray
    """Temperature (K)."""
    zh: np.ndarray
    """Geopotential height above ground (m)."""


@dc.dataclass(frozen=True)
class ParcelOrigin:
    """Properties of the lifted parcel at its launch level.

    All arrays have the shape of the horizontal dimensions of the input.
    """

    p: np.ndarray
    """Pressure (Pa)."""
    t: np.ndarray
    """Temperature (K)."""
    r: np.ndarray
    """Mixing ratio (kg/kg)."""


@dc.dataclass(frozen=True)
class ParcelPath:
    """Temperature profile of the lifted parcel and its environment.

    Profile arrays (``p``, ``t``, ``r``, ``tv``, ``tv_env``) have shape
    ``(n_levels, ...)``, where the levels are sorted in ascending pressure
    order regardless of the order supplied by the caller.

    The key diagnostic levels (``lcl``, ``lfc``, ``el``) and the parcel
    ``origin`` have shape ``(...)`` (horizontal dimensions only) so that the
    object is self-contained for skew-T / parcel-path plots.
    """

    # Profile arrays — shape (n_levels, ...)
    p: np.ndarray
    """Pressure grid, sorted ascending (Pa)."""
    zh: np.ndarray
    """Geopotential height above ground, sorted ascending in pressure (m)."""
    t: np.ndarray
    """Parcel temperature (K)."""
    r: np.ndarray
    """Parcel mixing ratio (kg/kg)."""
    tv: np.ndarray
    """Parcel virtual temperature (K)."""
    tv_env: np.ndarray
    """Environment virtual temperature (K)."""

    # Key levels — shape (...) i.e. horizontal dims only
    lcl: PressureLevel
    """Lifted Condensation Level."""
    lfc: PressureLevel
    """Level of Free Convection (NaN where no convection)."""
    el: PressureLevel
    """Equilibrium Level (NaN where no convection)."""

    # Parcel origin
    origin: ParcelOrigin
    """Parcel properties at the launch level."""


def _ept_from_mixing_ratio(t, p, r, method="bolton39"):
    specific_humidity = thermo.specific_humidity_from_mixing_ratio(r)
    return thermo.ept_from_specific_humidity(t, specific_humidity, p, method=method)


def _where_is_param_zero(level, p, param):

    n_levels = p.shape[0]

    level = np.clip(level, 1, n_levels - 1)

    p_above = np.take_along_axis(p, level[None], axis=0).squeeze(0)
    p_below = np.take_along_axis(p, (level - 1)[None], axis=0).squeeze(0)
    param_above = np.take_along_axis(param, level[None], axis=0).squeeze(0)
    param_below = np.take_along_axis(param, (level - 1)[None], axis=0).squeeze(0)

    result = p_below + ((-param_below) / (param_above - param_below)) * (p_above - p_below)

    return result


def _vertical_weighted_mean(p, param, p_bottom, p_top):
    """
    Pressure-weighted mean for ASCENDING pressure.

    p[0, :]  = top / low pressure
    p[-1, :] = bottom / high pressure

    p_bottom > p_top
    """
    p = np.asarray(p)
    param = np.asarray(param)

    p_bottom = np.asarray(p_bottom)
    p_top = np.asarray(p_top)

    # Layer endpoints for ascending p
    p0 = p[:-1]  # upper / lower-pressure side of layer
    p1 = p[1:]  # lower / higher-pressure side of layer

    dp = p1 - p0  # positive for valid ascending layers

    valid = np.isfinite(dp) & (dp > 0.0)

    # Layer-mean parameter
    layer_param = 0.5 * (param[1:] + param[:-1])

    pb = np.expand_dims(p_bottom, 0)
    pt = np.expand_dims(p_top, 0)

    b = np.zeros_like(dp, dtype=float)
    c = np.zeros_like(dp, dtype=float)

    # Ascending-p equivalent of your original weighting
    np.divide(pb - p0, dp, out=b, where=valid)
    np.divide(p1 - pt, dp, out=c, where=valid)

    b = np.clip(b, 0.0, 1.0)
    c = np.clip(c, 0.0, 1.0)

    weights = b * c * dp
    weights[~valid] = 0.0

    sum_of_weights = np.nansum(weights, axis=0)
    sum_of_weights[sum_of_weights == 0.0] = np.nan

    weightedmean = np.nansum(weights * layer_param, axis=0) / sum_of_weights

    return weightedmean


def _lfc_index(z, b, z_lcl, min_depth=1000.0, threshold=0.0):
    """
    LFC index for DESCENDING z.

    z     : (n_levels, n_profiles), descending with index
            z[0, :] high/top, z[-1, :] low/surface
    b     : (n_levels, n_profiles), buoyancy
    z_lcl : (n_profiles,), LCL height
    min_depth : required contiguous buoyant depth in metres
    threshold : buoyancy threshold
    """
    n_levels = b.shape[0]

    # Above LCL and positively buoyant
    is_buoyant = (z >= z_lcl[None]) & (b > threshold)

    # contig_depth[i] = depth of contiguous buoyant layer above level i
    contig_depth = np.zeros_like(b, dtype=float)

    # Start near the top and move downward.
    # For descending z, thickness between level i-1 and i is z[i-1] - z[i].
    for i in range(1, n_levels):
        dz = z[i - 1] - z[i]  # positive for descending z

        contig_depth[i] = np.where(is_buoyant[i], contig_depth[i - 1] + dz, 0.0)

    has_deep_buoyancy = contig_depth >= min_depth

    exists = np.any(has_deep_buoyancy, axis=0)

    # For descending z, surface is at high index.
    # LFC = lowest/base index of sufficiently deep buoyant layer
    idx_lfc = n_levels - 1 - np.argmax(has_deep_buoyancy[::-1], axis=0)

    # Use 0 if no LFC exists, preserving your original convention
    idx_lfc = np.where(exists, idx_lfc, 0)

    return idx_lfc


def _moist_ascent_lookup_table(ept_method):
    def dt_dp_moist(t_parcel, p):
        # moist adiabatic gradient according to Emanuel, 1995 (Eq. 4.7.3) ignoring liquid and solid water,
        # i.e. r_l = 0 and r_t = r
        es_parcel = thermo.saturation_vapour_pressure(t_parcel, phase="water")
        r_parcel = constants.epsilon * es_parcel / (p - es_parcel)

        dlv_dt = constants.c_pv - C_pl
        lv = constants.Lv + dlv_dt * (t_parcel - constants.T0)

        # Terms from Emanuel, 1995 (Eq. 4.7.3)
        a_prefactor = (
            -(constants.g / constants.c_pd) * (1 + r_parcel) / (1 + r_parcel * (constants.c_pv / constants.c_pd))
        )
        b_factor = 1 + (lv * r_parcel) / (constants.Rd * t_parcel)
        c_term = lv * lv * r_parcel * (1 + r_parcel / constants.epsilon)
        d_term = constants.Rv * np.power(t_parcel, 2) * (constants.c_pd + r_parcel * constants.c_pv)
        dt_dz = a_prefactor * b_factor / (1 + (c_term / d_term))

        t_v = thermo.virtual_temperature(t_parcel, thermo.specific_humidity_from_mixing_ratio(r_parcel))
        dz_dp = -(constants.Rd * t_v) / (p * constants.g)
        return dt_dz * dz_dp

    p_max = 110000
    p_min = 1000

    t_initial = np.arange(180, 320, 2)
    es_initial = thermo.saturation_vapour_pressure(t_initial, phase="water")
    r_initial = constants.epsilon * (es_initial / (p_max - es_initial))
    theta_ep_range = _ept_from_mixing_ratio(t_initial, p_max, r_initial, method=ept_method)

    pressure_levels = np.arange(p_max, p_min, -100)

    t_lookup = np.empty((pressure_levels.shape[0], t_initial.shape[0]))
    r_lookup = np.empty((pressure_levels.shape[0], r_initial.shape[0]))

    t_lookup[0, :] = t_initial
    r_lookup[0, :] = r_initial

    for level in range(1, pressure_levels.shape[0]):
        p_mid = (pressure_levels[level - 1] + pressure_levels[level]) / 2
        dp = pressure_levels[level] - pressure_levels[level - 1]
        t_lookup[level, :] = t_lookup[level - 1, :] + dt_dp_moist(t_lookup[level - 1, :], p_mid) * dp
        es_level = thermo.saturation_vapour_pressure(t_lookup[level, :], phase="water")
        r_lookup[level, :] = constants.epsilon * es_level / (pressure_levels[level] - es_level)

    t_lookup = t_lookup[::-10, :]
    pressure_levels = pressure_levels[::-10]

    return {"temperature": t_lookup, "theta_ep": theta_ep_range, "pressure": pressure_levels}


_VALID_EXTRA_OUTPUTS = frozenset(["lcl", "lfc", "el", "parcel", "parcel_path"])


class _CapeCinComp:
    def __init__(
        self,
        h_bottom=None,
        h_top=None,
        layer_depth=None,
        extra_outputs=None,
        lcl_method="davies",
        ept_method="bolton39",
    ):
        self.h_bottom = h_bottom
        self.h_top = h_top
        self.layer_depth = layer_depth
        self.extra_outputs = extra_outputs or []
        self.lcl_method = lcl_method
        self.ept_method = ept_method

    def _lifted_condensation_level_from_mixing_ratio(self, t_departure, p_departure, r_departure):
        specific_humidity = thermo.specific_humidity_from_mixing_ratio(r_departure)
        dewpoint = thermo.dewpoint_from_specific_humidity(specific_humidity, p_departure)
        t_LCL, p_LCL = thermo.lcl(t_departure, dewpoint, p_departure, method=self.lcl_method)
        return p_LCL, t_LCL

    def _moist_ascent_lookup_table(self):
        return _moist_ascent_lookup_table(ept_method=self.ept_method)

    def _lift_parcel(self, p_start, t_start, r_start, zh_start, p, t, r, zh):
        p_shape = p.shape
        t_parcel = np.zeros(p_shape) * np.nan
        r_parcel = np.zeros(p_shape) * np.nan

        p_lcl, t_lcl = self._lifted_condensation_level_from_mixing_ratio(t_start, p_start, r_start)
        cond = (p <= p_start[None, :]) & (p <= p_lcl[None, :])

        has_lcl = cond.any(axis=0)

        # find index of first layer for which p <= p_lcl
        idx_lcl_level = np.where(
            has_lcl,
            p.shape[0] - 1 - np.argmax(cond[::-1], axis=0),
            -1,
        )
        z_lcl = _where_is_param_zero(idx_lcl_level, zh, p - p_lcl)

        theta_parcel = thermo.potential_temperature(t_start, p_start)
        theta_ep_parcel = _ept_from_mixing_ratio(t_start, p_start, r_start, method=self.ept_method)

        # Moist adiabatic ascent
        above_lcl = p_lcl[None, ...] > p
        p_2d = p * np.ones(p_shape)
        # theta_ep_parcel_2d = theta_ep_parcel[None, ...] * np.ones(p_shape)
        theta_ep_1d = np.broadcast_to(theta_ep_parcel, (p_shape)).flatten()
        points = np.concatenate((theta_ep_1d[:, None], p.flatten()[:, None]), axis=1)

        lookup_table = self._moist_ascent_lookup_table()
        t_moist_adiabat, theta_ep_range, p_range = (
            lookup_table["temperature"],
            lookup_table["theta_ep"],
            lookup_table["pressure"],
        )
        # t_interp = interpolate.RectBivariateSpline(p_range, theta_ep_range, t_moist_adiabat)
        t_interp = interpolate.RegularGridInterpolator(
            (theta_ep_range, p_range), t_moist_adiabat.T, method="linear", bounds_error=False, fill_value=np.nan
        )
        t_parcel = t_interp(points)[:, None].reshape((p_shape))

        es_t_parcel = thermo.saturation_vapour_pressure(t_parcel[above_lcl], phase="water")
        r_parcel[above_lcl] = constants.epsilon * es_t_parcel / (p_2d[above_lcl] - es_t_parcel)

        # Mask out t_parcel below parcel source
        t_parcel[p > p_start] = np.nan

        # Dry adiabatic ascent to LCL
        between_start_and_lcl = (p > p_lcl[None, ...]) * (p <= p_start[None, ...])
        t_parcel[between_start_and_lcl] = thermo.temperature_from_potential_temperature(theta_parcel[None, ...], p)[
            between_start_and_lcl
        ]
        r_parcel[between_start_and_lcl] = (r_start[None, ...] * np.ones(p_shape))[between_start_and_lcl]

        # Calculate buoyancy
        specific_humidity_arr = thermo.specific_humidity_from_mixing_ratio(r)
        tv_env = thermo.virtual_temperature(t, specific_humidity_arr)

        specific_humidity_parcel = thermo.specific_humidity_from_mixing_ratio(r_parcel)
        tv_parcel = thermo.virtual_temperature(t_parcel, specific_humidity_parcel)
        dtv = tv_parcel - tv_env
        buoyancy = dtv / tv_env

        # Level of Free Convection (LFC)
        # ---------------------------------
        # LFC is calculated assuming by default min_depths of the layer with positive buoyancy,
        # threshold=0.1 K for buoyancy;
        # min_depth and threshold could be changes as they are arguments of LFC_index func.
        # min_depth and threshold parameters are determined to avoid fake LFC selection
        # due to shallow buoyant layers or numerical errors.

        idx_lfc_level = _lfc_index(zh, dtv, z_lcl)
        p_lfc = _where_is_param_zero(idx_lfc_level, p, dtv)
        z_lfc = _where_is_param_zero(idx_lfc_level, zh, dtv)

        p_lfc[idx_lfc_level == -1] = np.nan
        z_lfc[idx_lfc_level == -1] = np.nan

        # In cases where the LCL was not reached at the bottom, but was reached at the top
        # set the p_LFC to p_LCL instead.
        p_at_lfc_minus1 = np.take_along_axis(p, np.maximum(idx_lfc_level - 1, 0)[None], axis=0).squeeze(0)
        lfc_below_lcl = p_at_lfc_minus1 >= p_lcl
        z_lfc[lfc_below_lcl] = z_lcl[lfc_below_lcl]
        p_lfc[lfc_below_lcl] = p_lcl[lfc_below_lcl]

        # Temperatures at LFC: interpolate t_parcel to where dtv crosses zero
        t_lfc = _where_is_param_zero(idx_lfc_level, t_parcel, dtv)
        t_lfc[idx_lfc_level == -1] = np.nan
        t_lfc[lfc_below_lcl] = t_lcl[lfc_below_lcl]

        # Equilibrium Level (EL)
        # -------------------------
        el_level = dtv.shape[0] - np.argmax(
            dtv[::-1] > 0, axis=0
        )  # finds index of first layer (going from top to bottom through profile) for which b > 0
        z_el = _where_is_param_zero(el_level, zh, dtv)
        p_el = _where_is_param_zero(el_level, p, dtv)

        # Temperature at EL: interpolate t_parcel to where dtv crosses zero
        t_el = _where_is_param_zero(el_level, t_parcel, dtv)

        return (
            buoyancy,
            p_lcl,
            z_lcl,
            t_lcl,
            p_lfc,
            z_lfc,
            t_lfc,
            p_el,
            z_el,
            t_el,
            t_parcel,
            r_parcel,
            tv_parcel,
            tv_env,
        )

    def _sort_pressure_levels(self, p, t, r, zh):
        # NaN values (sub-ground levels) should sort to the end (treated as infinity)
        p_sort_key = np.where(np.isnan(p), np.inf, p)
        is_sorted = (np.diff(p_sort_key, axis=0) >= 0).all()
        if not is_sorted:
            sorted_inds = np.argsort(p_sort_key, axis=0, kind="stable")
            p = np.take_along_axis(p, sorted_inds, axis=0)
            t = np.take_along_axis(t, sorted_inds, axis=0)
            r = np.take_along_axis(r, sorted_inds, axis=0)
            zh = np.take_along_axis(zh, sorted_inds, axis=0)
        return p, t, r, zh

    def _integrate_buoyancy(self, buoyancy, p, zh, p_lfc):
        layer_thickness = -np.diff(zh, axis=0)
        dcape = constants.g * ((buoyancy[:-1] + buoyancy[1:]) / 2) * layer_thickness
        dcin = np.copy(dcape)

        dcape[dcape < 0] = 0
        above_lfc = p[1:] <= p_lfc[None]
        dcape[~above_lfc] = 0
        cape = np.nansum(dcape, axis=0)
        cape[np.isnan(p_lfc)] = 0

        dcin[dcin > 0] = 0
        dcin[above_lfc] = 0

        cin = -np.nansum(dcin, axis=0)
        pos_cape = cape > 0
        cin[~pos_cape] = 0
        return cape, cin

    def _determine_parcel(self, p, zh, t, r, p_sfc, t_sfc, r_sfc, zh_sfc, h_bottom, h_top, layer_depth):
        raise NotImplementedError("This method should be implemented in the subclass")

    def _cape_cin(self, p, zh, t, r, p_sfc, t_sfc, r_sfc, zh_sfc):
        # Profile arrays have shape (n_pressure_levels, ...) with the vertical axis
        # as axis=0. Surface arrays have the horizontal-only shape (...). Internally
        # the surface is concatenated as an additional level.
        p = np.concatenate([p_sfc[None], p], axis=0)
        zh = np.concatenate([zh_sfc[None], zh], axis=0)
        t = np.concatenate([t_sfc[None], t], axis=0)
        r = np.concatenate([r_sfc[None], r], axis=0)

        # Identify subground levels using height
        subground = zh < zh_sfc[None, ...]

        # Detect bad-data NaN: NaN at above-ground grid positions or NaN in the surface inputs.
        #    NaN at subground positions (which may already be present in the input) is expected
        #    and should NOT be treated as bad data.
        nan_in_grid = np.isnan(p) | np.isnan(t) | np.isnan(r) | np.isnan(zh)
        bad_data_in_grid = nan_in_grid & ~subground
        bad_data_in_sfc = np.isnan(p_sfc) | np.isnan(t_sfc) | np.isnan(r_sfc) | np.isnan(zh_sfc)
        unexpected_nan = np.any(bad_data_in_grid, axis=0) | bad_data_in_sfc

        # Mask sub-ground levels to NaN so they are excluded from all computations
        p = np.where(subground, np.nan, p)
        t = np.where(subground, np.nan, t)
        r = np.where(subground, np.nan, r)
        zh = np.where(subground, np.nan, zh)

        # Sort ascending by pressure; NaN (sub-ground) levels sort to the end
        p, t, r, zh = self._sort_pressure_levels(p, t, r, zh)

        # Heights relative to the surface
        zh = zh - zh_sfc[None]
        zh_sfc_rel = np.zeros_like(p_sfc)

        p_start, t_start, r_start, zh_start = self._determine_parcel(
            p, zh, t, r, p_sfc, t_sfc, r_sfc, zh_sfc_rel, self.h_bottom, self.h_top, self.layer_depth
        )

        buoyancy, p_lcl, z_lcl, t_lcl, p_lfc, z_lfc, t_lfc, p_el, z_el, t_el, t_parcel, r_parcel, tv_parcel, tv_env = (
            self._lift_parcel(p_start, t_start, r_start, zh_start, p, t, r, zh)
        )
        cape, cin = self._integrate_buoyancy(buoyancy, p, zh, p_lfc)

        cape[unexpected_nan] = np.nan
        cin[unexpected_nan] = np.nan

        # TODO include LI calculation here and add to extra outputs if requested

        if not self.extra_outputs:
            return cape, cin

        lcl = PressureLevel(p=p_lcl, t=t_lcl, zh=z_lcl)
        lfc = PressureLevel(p=p_lfc, t=t_lfc, zh=z_lfc)
        el = PressureLevel(p=p_el, t=t_el, zh=z_el)
        origin = ParcelOrigin(p=p_start, t=t_start, r=r_start)

        extras = {}
        for key in self.extra_outputs:
            if key == "lcl":
                extras["lcl"] = lcl
            elif key == "lfc":
                extras["lfc"] = lfc
            elif key == "el":
                extras["el"] = el
            elif key == "parcel":
                extras["parcel"] = origin
            elif key == "parcel_path":
                extras["parcel_path"] = ParcelPath(
                    p=p,
                    zh=zh,
                    t=t_parcel,
                    r=r_parcel,
                    tv=tv_parcel,
                    tv_env=tv_env,
                    lcl=lcl,
                    lfc=lfc,
                    el=el,
                    origin=origin,
                )
        return cape, cin, extras


class _CapeCinSurface(_CapeCinComp):
    def _determine_parcel(self, p, zh, t, r, p_sfc, t_sfc, r_sfc, zh_sfc, h_bottom, h_top, layer_depth):
        return p_sfc, t_sfc, r_sfc, zh_sfc


class _CapeCinMixed(_CapeCinComp):
    def _determine_parcel(self, p, zh, t, r, p_sfc, t_sfc, r_sfc, zh_sfc, h_bottom, h_top, layer_depth=None):
        """
        Compute mixed-layer parameters.

        :param p: pressure array in Pa
        :param t: temperature array in K
        :param r: mixing ratio array in kg/kg
        :param layer_depth: in Pa
        :return: bottom pressure, mixed-layer t, mixed_layer r, bottom_height
        """
        if layer_depth is None:
            layer_depth = 5000

        p_bottom = p_sfc
        zh_bottom = zh_sfc

        theta = thermo.potential_temperature(t, p)

        theta_mean = _vertical_weighted_mean(p, theta, p_bottom, p_bottom - layer_depth)
        t_mixed = thermo.temperature_from_potential_temperature(theta_mean, p_bottom)
        r_mixed = _vertical_weighted_mean(p, r, p_bottom, p_bottom - layer_depth)

        return p_bottom, t_mixed, r_mixed, zh_bottom


class _CapeCinMostUnstable(_CapeCinComp):
    def _determine_parcel(self, p, zh, t, r, p_sfc, t_sfc, r_sfc, zh_sfc, h_bottom=None, h_top=None, layer_depth=None):
        if h_bottom is None:
            h_bottom = 0
        if h_top is None:
            h_top = 3000

        theta_ep_env = _ept_from_mixing_ratio(t, p, r, method=self.ept_method)

        # finding the most unstable parcel between h_bottom to h_top in [m]
        condition = (zh < h_bottom) | (zh > h_top)
        theta_ep_copy = np.copy(theta_ep_env)
        theta_ep_copy[condition] = np.nan

        theta_ep_safe = np.where(np.isnan(theta_ep_copy), -np.inf, theta_ep_copy)
        level_max_theta_ep = np.argmax(theta_ep_safe, axis=0)

        t_start = np.take_along_axis(t, level_max_theta_ep[None], axis=0).squeeze(0)
        p_start = np.take_along_axis(p, level_max_theta_ep[None], axis=0).squeeze(0)
        r_start = np.take_along_axis(r, level_max_theta_ep[None], axis=0).squeeze(0)
        z_start = np.take_along_axis(zh, level_max_theta_ep[None], axis=0).squeeze(0)

        return p_start, t_start, r_start, z_start


_PARCEL_CLASSES = {
    "surface": _CapeCinSurface,
    "mixed": _CapeCinMixed,
    "mu": _CapeCinMostUnstable,
}


def cape_cin(
    p,
    zh,
    t,
    r,
    p_sfc,
    t_sfc,
    r_sfc,
    zh_sfc,
    parcel_type,
    h_bottom=None,
    h_top=None,
    layer_depth=None,
    extra_outputs=None,
    vertical_axis=0,
    ept_method="bolton39",
    lcl_method="davies",
):
    r"""Compute Convective Available Potential Energy (CAPE) and Convective Inhibition (CIN).

    Parameters
    ----------
    p : array-like
        Pressure (Pa) on pressure levels. The vertical axis must be the first axis
        (axis=0) unless ``vertical_axis`` is set.
    zh : array-like
        Geopotential height (m) on pressure levels, same shape as ``p``.
    t : array-like
        Temperature (K) on pressure levels, same shape as ``p``.
    r : array-like
        Mixing ratio (kg/kg) on pressure levels, same shape as ``p``.
    p_sfc : array-like
        Surface pressure (Pa), shape equal to the horizontal dimensions of ``p``.
        The surface is included as an additional level in the computation.
    t_sfc : array-like
        Surface temperature (K), same horizontal shape as ``p_sfc``.
    r_sfc : array-like
        Surface mixing ratio (kg/kg), same horizontal shape as ``p_sfc``.
    zh_sfc : array-like
        Surface geopotential height (m), same horizontal shape as ``p_sfc``.
        Used as the height reference: all profile heights are expressed relative
        to ``zh_sfc`` internally.
    parcel_type : str
        Method used to define the lifted parcel. One of:

        * ``"surface"`` — parcel taken from ``p_sfc``/``t_sfc``/``r_sfc``.
        * ``"mixed"`` — parcel properties averaged over a mixed layer of depth
          ``layer_depth`` (Pa) above the surface.
        * ``"mu"`` — most-unstable parcel: the level within the height range
          ``[h_bottom, h_top]`` (m) that maximises equivalent potential temperature.
    h_bottom : number, optional
        Height (m above surface) of the bottom of the search range used by the
        most-unstable parcel. Defaults to ``0``.
    h_top : number, optional
        Height (m above surface) of the top of the search range used by the
        most-unstable parcel. Defaults to ``3000``.
    layer_depth : number, optional
        Depth (Pa) of the layer used to define the mixed-layer parcel.
        Defaults to 5000 Pa for ``"mixed"``.
    extra_outputs : list of str, optional
        Optional diagnostics to compute and return as a third element.
        Allowed keys:

        * ``"lcl"`` — :class:`PressureLevel` for the Lifted Condensation Level.
        * ``"lfc"`` — :class:`PressureLevel` for the Level of Free Convection.
        * ``"el"``  — :class:`PressureLevel` for the Equilibrium Level.
        * ``"parcel"`` — :class:`ParcelOrigin` with the parcel launch properties.
        * ``"parcel_path"`` — :class:`ParcelPath` with the full parcel profile
          and all key levels bundled together.

        When ``None`` or an empty list the function returns only ``(cape, cin)``.
    vertical_axis : int, optional
        Axis of the input arrays that corresponds to the vertical dimension.
        Defaults to ``0``. ``-1`` may also be used to indicate the last axis.
        Surface arrays (``p_sfc``, ``t_sfc``, ``r_sfc``, ``zh_sfc``) have no
        vertical axis and are not affected by this parameter.
    ept_method : str, optional
        Method used to compute equivalent potential temperature. Passed to
        :func:`earthkit.meteo.thermo.array.ept_from_specific_humidity`.
        Defaults to ``"bolton39"``.
    lcl_method : str, optional
        Method used to compute the Lifted Condensation Level. Passed to
        :func:`earthkit.meteo.thermo.array.lcl`. Defaults to ``"davies"``.

    Returns
    -------
    cape : array-like
        CAPE (J/kg), shape equal to the horizontal dimensions of the input arrays.
    cin : array-like
        CIN (J/kg), shape equal to the horizontal dimensions of the input arrays.
    extras : dict, optional
        Only present when ``extra_outputs`` is non-empty. Keys are a subset of
        the strings listed in the ``extra_outputs`` parameter description.

    """
    if extra_outputs is not None:
        unknown = set(extra_outputs) - _VALID_EXTRA_OUTPUTS
        if unknown:
            raise ValueError(
                f"Invalid extra_outputs keys: {sorted(unknown)}. Allowed values are: {sorted(_VALID_EXTRA_OUTPUTS)}"
            )

    if parcel_type not in _PARCEL_CLASSES:
        raise ValueError(f"Invalid parcel_type '{parcel_type}'. Must be one of {list(_PARCEL_CLASSES)}")

    if vertical_axis != 0:
        if vertical_axis == -1:
            vertical_axis = p.ndim - 1
        if vertical_axis < 0 or vertical_axis >= p.ndim:
            raise ValueError(f"Invalid vertical_axis {vertical_axis} for input arrays with {p.ndim} dimensions")

        p = np.swapaxes(p, 0, vertical_axis)
        zh = np.swapaxes(zh, 0, vertical_axis)
        t = np.swapaxes(t, 0, vertical_axis)
        r = np.swapaxes(r, 0, vertical_axis)

    if h_bottom is None:
        h_bottom = 0
    if h_top is None:
        h_top = 3000

    p_sfc = np.asarray(p_sfc, dtype=float)
    t_sfc = np.asarray(t_sfc, dtype=float)
    r_sfc = np.asarray(r_sfc, dtype=float)
    zh_sfc = np.asarray(zh_sfc, dtype=float)

    result = _PARCEL_CLASSES[parcel_type](
        h_bottom=h_bottom,
        h_top=h_top,
        layer_depth=layer_depth,
        extra_outputs=extra_outputs,
        lcl_method=lcl_method,
        ept_method=ept_method,
    )._cape_cin(p, zh, t, r, p_sfc, t_sfc, r_sfc, zh_sfc)

    # Swap the vertical axis of profile arrays back to match caller's layout.
    if vertical_axis != 0 and len(result) == 3:
        cape, cin, extras = result
        if "parcel_path" in extras:
            path = extras["parcel_path"]
            extras["parcel_path"] = ParcelPath(
                p=np.swapaxes(path.p, 0, vertical_axis),
                zh=np.swapaxes(path.zh, 0, vertical_axis),
                t=np.swapaxes(path.t, 0, vertical_axis),
                r=np.swapaxes(path.r, 0, vertical_axis),
                tv=np.swapaxes(path.tv, 0, vertical_axis),
                tv_env=np.swapaxes(path.tv_env, 0, vertical_axis),
                lcl=path.lcl,
                lfc=path.lfc,
                el=path.el,
                origin=path.origin,
            )
        result = cape, cin, extras

    return result
