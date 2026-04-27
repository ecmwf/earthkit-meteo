# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import dataclasses as dc
from typing import Any, TypeAlias

import numpy as np
from scipy import interpolate

from earthkit.meteo import thermo
from earthkit.meteo.constants import constants

ArrayLike: TypeAlias = Any

C_pl = 4180


@dc.dataclass(frozen=True)
class PressureLevel:
    """Pressure and temperature at a single key level.

    Both arrays have the shape of the horizontal dimensions of the input
    (i.e. the vertical axis is absent).  Values are NaN where the level does
    not exist (e.g. no LFC in a stable profile).
    """

    p: np.ndarray
    """Pressure (Pa)."""
    t: np.ndarray
    """Temperature (K)."""


@dc.dataclass(frozen=True)
class ParcelOrigin:
    """Properties of the lifted parcel at its launch level.

    All arrays have the shape of the horizontal dimensions of the input.
    """

    p: np.ndarray
    """Pressure (Pa)."""
    t: np.ndarray
    """Temperature (K)."""
    q: np.ndarray
    """Specific humidity (kg/kg)."""


@dc.dataclass(frozen=True)
class ParcelPath:
    """Temperature profile of the lifted parcel and its environment.

    Profile arrays (``p``, ``t``, ``q``, ``tv``, ``tv_env``) have shape
    ``(n_levels, ...)``, where the levels are sorted in ascending pressure
    order regardless of the order supplied by the caller.

    The key diagnostic levels (``lcl``, ``lfc``, ``el``) and the parcel
    ``origin`` have shape ``(...)`` (horizontal dimensions only) so that the
    object is self-contained for skew-T / parcel-path plots.
    """

    # Profile arrays — shape (n_levels, ...)
    p: np.ndarray
    """Pressure grid, sorted ascending (Pa)."""
    t: np.ndarray
    """Parcel temperature (K)."""
    q: np.ndarray
    """Parcel specific humidity (kg/kg)."""
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


def _moist_ascent_lookup_table(ept_method):
    def dt_dp_moist(t_parcel, p):
        # moist adiabatic gradient according to Emanuel, 1995 (Eq. 4.7.3) ignoring liquid and solid water,
        # i.e. r_l = 0 and r_t = r
        es_parcel = thermo.saturation_vapour_pressure(t_parcel, phase="water")
        r_parcel = constants.epsilon * es_parcel / p

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
    q_initial = thermo.specific_humidity_from_mixing_ratio(r_initial)
    theta_ep_range = thermo.ept_from_specific_humidity(t_initial, q_initial, p_max, method=ept_method)

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
    def __init__(self, layer_depth=None, extra_outputs=None, lcl_method="davies", ept_method="bolton43"):
        self.layer_depth = layer_depth
        self.extra_outputs = extra_outputs or []
        self.lcl_method = lcl_method
        self.ept_method = ept_method

    def _lifted_condensation_level(self, t_departure, p_departure, q_departure):
        dewpoint = thermo.dewpoint_from_specific_humidity(q_departure, p_departure)
        t_LCL, p_LCL = thermo.lcl(t_departure, dewpoint, p_departure, method=self.lcl_method)
        return p_LCL, t_LCL

    def _moist_ascent_lookup_table(self):
        return _moist_ascent_lookup_table(ept_method=self.ept_method)

    def _lift_parcel(self, p_start, t_start, q_start, p, t, q):
        p_shape = p.shape
        t_parcel = np.full(p_shape, np.nan)
        q_parcel = np.full(p_shape, np.nan)

        p_lcl, t_lcl = self._lifted_condensation_level(t_start, p_start, q_start)

        theta_parcel = thermo.potential_temperature(t_start, p_start)
        theta_ep_parcel = thermo.ept_from_specific_humidity(t_start, q_start, p_start, method=self.ept_method)

        # Dry adiabatic ascent to LCL (only at valid, non-NaN pressure levels)
        valid_p = ~np.isnan(p)
        between_start_and_lcl = valid_p & (p > p_lcl[None, ...]) & (p <= p_start[None, ...])
        t_parcel[between_start_and_lcl] = thermo.temperature_from_potential_temperature(theta_parcel[None, ...], p)[
            between_start_and_lcl
        ]
        q_parcel[between_start_and_lcl] = (q_start[None, ...] * np.ones(p_shape))[between_start_and_lcl]

        # Moist adiabatic ascent
        above_lcl = valid_p & (p_lcl[None, ...] > p)
        p_2d = p * np.ones(p_shape)
        theta_ep_parcel_2d = theta_ep_parcel[None, ...] * np.ones(p_shape)

        lookup_table = self._moist_ascent_lookup_table()
        t_moist_adiabat, theta_ep_range, p_range = (
            lookup_table["temperature"],
            lookup_table["theta_ep"],
            lookup_table["pressure"],
        )
        t_interp = interpolate.RectBivariateSpline(p_range, theta_ep_range, t_moist_adiabat)
        t_parcel[above_lcl] = t_interp(p_2d[above_lcl], theta_ep_parcel_2d[above_lcl], grid=False)
        es_t_parcel = thermo.saturation_vapour_pressure(t_parcel[above_lcl], phase="water")
        r_parcel_moist = constants.epsilon * es_t_parcel / (p_2d[above_lcl] - es_t_parcel)
        q_parcel[above_lcl] = thermo.specific_humidity_from_mixing_ratio(r_parcel_moist)

        # Calculate buoyancy
        tv_env = thermo.virtual_temperature(t, q)
        tv_parcel = thermo.virtual_temperature(t_parcel, q_parcel)
        dtv = tv_parcel - tv_env
        buoyancy = dtv / tv_env

        buoyant_layer_mask = (buoyancy > 0.0) & (t < t_lcl[None, ...])

        lfc_index = buoyant_layer_mask.shape[0] - np.argmax(buoyant_layer_mask[::-1, ...], axis=0) - 1
        el_index = np.argmax(buoyant_layer_mask, axis=0)

        # For now, this gives the pressure at which the parcel is not buoyant i.e. it does not interpolate
        p_lfc = np.take_along_axis(p, lfc_index[None, ...], axis=0).squeeze(0)
        t_lfc = np.take_along_axis(t, lfc_index[None, ...], axis=0).squeeze(0)

        p_el = np.take_along_axis(p, el_index[None, ...], axis=0).squeeze(0)
        t_el = np.take_along_axis(t, el_index[None, ...], axis=0).squeeze(0)

        buoyant_mask = np.max(buoyant_layer_mask, axis=0)
        p_lfc = np.where(buoyant_mask, p_lfc.astype(float), np.nan)

        return buoyancy, p_lcl, t_lcl, p_lfc, t_lfc, p_el, t_el, tv_parcel, tv_env, t_parcel, q_parcel

    def _sort_pressure_levels(self, p, t, q, zh):
        # np.argsort places NaN at the end, which is what we want:
        # subground levels (NaN) sort to the high-pressure tail.
        sorted_inds = np.argsort(p, axis=0)
        p = np.take_along_axis(p, sorted_inds, axis=0)
        t = np.take_along_axis(t, sorted_inds, axis=0)
        q = np.take_along_axis(q, sorted_inds, axis=0)
        zh = np.take_along_axis(zh, sorted_inds, axis=0)
        return p, t, q, zh

    def _integrate_buoyancy(self, buoyancy, p, zh, p_lfc, p_el):
        layer_thickness = -np.diff(zh, axis=0)
        dcape = constants.g * ((buoyancy[:-1] + buoyancy[1:]) / 2) * layer_thickness
        dcin = np.copy(dcape)

        dcape[dcape < 0] = 0
        above_lfc = p[1:, :] <= p_lfc[None, :]
        dcape[~above_lfc] = 0
        cape = np.nansum(dcape, axis=0)
        cape[np.isnan(p_lfc)] = 0

        above_el = p[1:, :] <= p_el[None, :]
        dcin[above_el] = 0
        dcin[dcin > 0] = 0

        cin = -np.nansum(dcin, axis=0)
        cin[cape <= 1] = 0
        return cape, cin

    def _determine_parcel(self, p, zh, t, q, layer_depth, p_sfc, t_sfc, q_sfc):
        raise NotImplementedError("This method should be implemented in the subclass")

    def _cape_cin(self, p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc):
        # Shapes of all arrays should be (n_vertical_levels, ...) where the vertical axis is the first axis (axis=0)
        # Surface arrays have shape (...) — i.e. horizontal dims only.

        # 1. Identify subground levels using height
        subground = zh < zh_sfc[None, ...]

        # 2. Detect bad-data NaN: NaN at above-ground grid positions or NaN in the surface inputs.
        #    NaN at subground positions (which may already be present in the input) is expected
        #    and should NOT be treated as bad data.
        nan_in_grid = np.isnan(p) | np.isnan(t) | np.isnan(q) | np.isnan(zh)
        bad_data_in_grid = nan_in_grid & ~subground
        bad_data_in_sfc = np.isnan(p_sfc) | np.isnan(t_sfc) | np.isnan(q_sfc) | np.isnan(zh_sfc)
        bad_data_mask = np.any(bad_data_in_grid, axis=0) | bad_data_in_sfc

        # 3. Mask subground levels to NaN in working copies
        p = np.where(subground, np.nan, p)
        zh = np.where(subground, np.nan, zh)
        t = np.where(subground, np.nan, t)
        q = np.where(subground, np.nan, q)

        # 4. Concatenate surface level into the grid
        p = np.concatenate([p, p_sfc[None, ...]], axis=0)
        zh = np.concatenate([zh, zh_sfc[None, ...]], axis=0)
        t = np.concatenate([t, t_sfc[None, ...]], axis=0)
        q = np.concatenate([q, q_sfc[None, ...]], axis=0)

        # 5. Sort pressure ascending (NaN subground levels go to the end)
        p, t, q, zh = self._sort_pressure_levels(p, t, q, zh)

        # 6. Determine parcel (using explicit surface properties)
        p_start, t_start, q_start = self._determine_parcel(p, zh, t, q, self.layer_depth, p_sfc, t_sfc, q_sfc)

        # 7. Lift parcel and compute buoyancy
        buoyancy, p_lcl, t_lcl, p_lfc, t_lfc, p_el, t_el, tv_parcel, tv_env, t_parcel, q_parcel = self._lift_parcel(
            p_start, t_start, q_start, p, t, q
        )

        # 8. Integrate buoyancy (nansum naturally skips NaN subground levels)
        cape, cin = self._integrate_buoyancy(buoyancy, p, zh, p_lfc, p_el)

        # 9. Apply bad-data mask (only genuine missing data, not subground NaN)
        cape[bad_data_mask] = np.nan
        cin[bad_data_mask] = np.nan

        # TODO include LI calculation and see if we can use earthkit's vertical interpolation function
        # instead of the custom Interpolate function from the reference implementation
        # [LI] = Interpolate(pressure_arr, [dTv], 500)
        # LI = -LI
        # LI[np.isnan(LI)] = 0

        if not self.extra_outputs:
            return cape, cin

        lcl = PressureLevel(p=p_lcl, t=t_lcl)
        lfc = PressureLevel(p=p_lfc, t=t_lfc)
        el = PressureLevel(p=p_el, t=t_el)
        origin = ParcelOrigin(p=p_start, t=t_start, q=q_start)

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
                    t=t_parcel,
                    q=q_parcel,
                    tv=tv_parcel,
                    tv_env=tv_env,
                    lcl=lcl,
                    lfc=lfc,
                    el=el,
                    origin=origin,
                )
        return cape, cin, extras


class _CapeCinSurface(_CapeCinComp):
    def _determine_parcel(self, p, zh, t, q, layer_depth, p_sfc, t_sfc, q_sfc):
        return p_sfc, t_sfc, q_sfc


class _CapeCinMixed(_CapeCinComp):
    def _determine_parcel(self, p, zh, t, q, layer_depth, p_sfc, t_sfc, q_sfc):
        if layer_depth is None:
            layer_depth = 5000

        p_bottom = p_sfc
        p_bound = p_bottom - layer_depth
        indx = (np.abs(np.nan_to_num(p, nan=0.0) - p_bound)).argmin(axis=0)
        p_top = np.take_along_axis(p, indx[None, ...], axis=0).squeeze(0)

        theta = thermo.potential_temperature(t, p)
        theta[(p > p_bottom[None, ...]) | (p < p_top[None, ...])] = np.nan
        theta_mean = np.nanmean(theta, axis=0)
        t_mixed = thermo.temperature_from_potential_temperature(theta_mean, p_bottom)

        q_copy = np.copy(q)
        q_copy[(p > p_bottom[None, ...]) | (p < p_top[None, ...])] = np.nan
        q_mixed = np.nanmean(q_copy, axis=0)

        return p_bottom, t_mixed, q_mixed


class _CapeCinMostUnstable(_CapeCinComp):
    def _determine_parcel(self, p, zh, t, q, layer_depth, p_sfc, t_sfc, q_sfc):
        t_shape = t.shape

        theta_ep_env = thermo.ept_from_specific_humidity(t, q, p, method=self.ept_method)
        if layer_depth is None:
            layer_depth = 50000

        # find local maxima of theta_ep in the vertical profile at pressures below layer_depth
        theta_ep_env[p < layer_depth] = np.nan
        theta_ep_copy = np.nan_to_num(theta_ep_env)
        theta_grad = theta_ep_copy[1:, :] - theta_ep_copy[:-1, :]

        # localmax is a boolean array with Trues where a local maximum of theta_ep was found
        maxima = (theta_grad[1:, :] < 0) * (theta_grad[:-1, :] > 0)
        localmax = np.ones((t_shape), dtype=bool)
        localmax[1:-1, :] = maxima
        trues = localmax.sum(axis=0)
        maxtrues = np.amax(trues)

        # localmaxarg is an integer array with values of k where a local maximum of theta_ep was found
        nz = t.shape[0]
        vertical_indices = np.arange(nz)[(...,) + (None,) * (t.ndim - 1)]

        localmaxarg = np.where(localmax, vertical_indices, 0)
        localmaxarg = np.sort(localmaxarg, axis=0)

        localmaxarg = localmaxarg[: -maxtrues - 1 : -1, :]

        cape_max = np.zeros(t_shape[1:])
        start_index_max = np.zeros(t_shape[1:], dtype=int)

        layer_thickness = -np.diff(zh, axis=0)

        for k_candidate in np.arange(0, localmaxarg.shape[0]):
            start_level_indices = localmaxarg[k_candidate, ...]
            p_start_candidate = np.take_along_axis(p, start_level_indices[None, ...], axis=0).squeeze(0)
            t_start_candidate = np.take_along_axis(t, start_level_indices[None, ...], axis=0).squeeze(0)
            q_start_candidate = np.take_along_axis(q, start_level_indices[None, ...], axis=0).squeeze(0)

            buoyancy, _, _, _, _, _, _, _, _, _, _ = self._lift_parcel(
                p_start_candidate, t_start_candidate, q_start_candidate, p, t, q
            )

            dcape = constants.g * ((buoyancy[:-1, :] + buoyancy[1:, :]) / 2) * layer_thickness
            dcape[dcape < 0] = 0
            cape = np.nansum(dcape, axis=0)

            is_greater = cape > cape_max
            is_valid = localmaxarg[k_candidate, :] > 0
            mask = is_greater & is_valid

            cape_max[mask] = cape[mask]
            start_index_max[mask] = localmaxarg[k_candidate, :][mask]

        p_start = np.take_along_axis(p, start_index_max[None, ...], axis=0).squeeze(0)
        t_start = np.take_along_axis(t, start_index_max[None, ...], axis=0).squeeze(0)
        q_start = np.take_along_axis(q, start_index_max[None, ...], axis=0).squeeze(0)
        return p_start, t_start, q_start


_PARCEL_CLASSES = {
    "surface": _CapeCinSurface,
    "mixed": _CapeCinMixed,
    "mu": _CapeCinMostUnstable,
}


def cape_cin(
    p: "ArrayLike",
    zh: "ArrayLike",
    t: "ArrayLike",
    q: "ArrayLike",
    p_sfc: "ArrayLike",
    zh_sfc: "ArrayLike",
    t_sfc: "ArrayLike",
    q_sfc: "ArrayLike",
    parcel_type: str,
    layer_depth: float | None = None,
    extra_outputs: list | None = None,
    vertical_axis: int = 0,
    ept_method: str = "bolton43",
    lcl_method: str = "davies",
):
    r"""Compute Convective Available Potential Energy (CAPE) and Convective Inhibition (CIN).

    The surface level must be provided separately from the pressure-level grid.
    Grid levels below the surface (sub-ground) are automatically masked
    using geopotential height and do not affect the result.

    Parameters
    ----------
    p : array-like
        Pressure on model/pressure levels (Pa), shape ``(n_levels, ...)``.
        The vertical axis must be the first axis (axis=0) unless
        ``vertical_axis`` is set.
    zh : array-like
        Geopotential height on model/pressure levels (m), same shape as ``p``.
    t : array-like
        Temperature on model/pressure levels (K), same shape as ``p``.
    q : array-like
        Specific humidity on model/pressure levels (kg/kg), same shape as ``p``.
    p_sfc : array-like
        Surface pressure (Pa), shape ``(...)`` (horizontal dimensions only).
    zh_sfc : array-like
        Surface geopotential height (m), same shape as ``p_sfc``.
    t_sfc : array-like
        Surface temperature (K), same shape as ``p_sfc``.
    q_sfc : array-like
        Surface specific humidity (kg/kg), same shape as ``p_sfc``.
    parcel_type : str
        Method used to define the lifted parcel. One of:

        * ``"surface"`` — parcel taken from the surface level.
        * ``"mixed"`` — parcel properties averaged over a mixed layer of depth
          ``layer_depth`` (Pa) above the surface.
        * ``"mu"`` — most-unstable parcel: the level within ``layer_depth`` (Pa)
          of the surface that maximises CAPE.
    layer_depth : number, optional
        Depth (Pa) of the layer used to define the mixed-layer or most-unstable
        parcel. Defaults to 5000 Pa for ``"mixed"`` and 50000 Pa for ``"mu"``.
    extra_outputs : list of str, optional
        Optional diagnostics to compute and return as a third element.
        Allowed keys:

        * ``"lcl"`` — :class:`PressureLevel` for the Lifted Condensation Level.
        * ``"lfc"`` — :class:`PressureLevel` for the Level of Free Convection.
        * ``"el"``  — :class:`PressureLevel` for the Equilibrium Level.
        * ``"parcel"`` — :class:`ParcelOrigin` with the parcel launch properties.
        * ``"parcel_path"`` — :class:`ParcelPath` with the full parcel profile
          and all key levels bundled together (includes the surface level).

        When ``None`` or an empty list the function returns only ``(cape, cin)``.
    vertical_axis : int, optional
        Axis of the input arrays that corresponds to the vertical dimension.
        Defaults to ``0``. ``-1`` may also be used to indicate the last axis.
    ept_method : str, optional
        Method used to compute equivalent potential temperature. Passed to
        :func:`earthkit.meteo.thermo.array.ept_from_specific_humidity`.
        Defaults to ``"bolton43"``.
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

    p = np.asarray(p, dtype=float)
    zh = np.asarray(zh, dtype=float)
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    p_sfc = np.asarray(p_sfc, dtype=float)
    zh_sfc = np.asarray(zh_sfc, dtype=float)
    t_sfc = np.asarray(t_sfc, dtype=float)
    q_sfc = np.asarray(q_sfc, dtype=float)

    if vertical_axis != 0:
        if vertical_axis == -1:
            vertical_axis = p.ndim - 1
        if vertical_axis < 0 or vertical_axis >= p.ndim:
            raise ValueError(f"Invalid vertical_axis {vertical_axis} for input arrays with {p.ndim} dimensions")

        p = np.swapaxes(p, 0, vertical_axis)
        zh = np.swapaxes(zh, 0, vertical_axis)
        t = np.swapaxes(t, 0, vertical_axis)
        q = np.swapaxes(q, 0, vertical_axis)

    result = _PARCEL_CLASSES[parcel_type](
        layer_depth=layer_depth, extra_outputs=extra_outputs, lcl_method=lcl_method, ept_method=ept_method
    )._cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc)

    # Swap the vertical axis of profile arrays back to match caller's layout.
    if vertical_axis != 0 and len(result) == 3:
        cape, cin, extras = result
        if "parcel_path" in extras:
            path = extras["parcel_path"]
            extras["parcel_path"] = ParcelPath(
                p=np.swapaxes(path.p, 0, vertical_axis),
                t=np.swapaxes(path.t, 0, vertical_axis),
                q=np.swapaxes(path.q, 0, vertical_axis),
                tv=np.swapaxes(path.tv, 0, vertical_axis),
                tv_env=np.swapaxes(path.tv_env, 0, vertical_axis),
                lcl=path.lcl,
                lfc=path.lfc,
                el=path.el,
                origin=path.origin,
            )
        result = cape, cin, extras

    return result
