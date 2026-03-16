import numpy as np
from scipy import interpolate

from earthkit.meteo import thermo
from earthkit.meteo.constants import constants

C_pl = 4180


def _ept_from_mixing_ratio(t, p, r, method="bolton39"):
    # TODO add option to use the method "bolton43" for ept calculation, which is the method used in the reference implementation.
    # Use "bolton39" for now, the difference is small.
    specific_humidity = thermo.specific_humidity_from_mixing_ratio(r)
    return thermo.ept_from_specific_humidity(t, specific_humidity, p, method=method)


def _moist_ascent_lookup_table(ept_method):
    def dt_dp_moist(t_parcel, p):
        # moist adiabatic gradient according to Emanuel, 1995 (Eq. 4.7.3) ignoring liquid and solid water, i.e. r_l = 0 and r_t = r
        es_parcel = thermo.saturation_vapour_pressure(t_parcel, phase="water")
        r_parcel = constants.epsilon * es_parcel / p

        dlv_dt = constants.c_pv - C_pl
        lv = constants.Lv + dlv_dt * (t_parcel - constants.T0)

        # Terms from Emanuel, 1995 (Eq. 4.7.3)
        a_prefactor = (
            -(constants.g / constants.c_pd)
            * (1 + r_parcel)
            / (1 + r_parcel * (constants.c_pv / constants.c_pd))
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


class _CapeCinComp:
    CM = {}

    def __init__(self, layer_depth=None, output="cape_cin", lcl_method="davies", ept_method="bolton39"):
        self.layer_depth = layer_depth
        self.output = output
        self.lcl_method = lcl_method
        self.ept_method = ept_method

    @staticmethod
    def make(method):
        return _CapeCinComp.CM[method]()

    def _lifted_condensation_level_from_mixing_ratio(self, t_departure, p_departure, r_departure):
        specific_humidity = thermo.specific_humidity_from_mixing_ratio(r_departure)
        dewpoint = thermo.dewpoint_from_specific_humidity(specific_humidity, p_departure)
        t_LCL, p_LCL = thermo.lcl(t_departure, dewpoint, p_departure, method=self.lcl_method)
        return p_LCL, t_LCL

    def _moist_ascent_lookup_table(self):
        return _moist_ascent_lookup_table(ept_method=self.ept_method)

    def _lift_parcel(self, p_start, t_start, r_start, p, t, r):
        p_shape = p.shape
        t_parcel = np.zeros(p_shape) * np.nan
        r_parcel = np.zeros(p_shape) * np.nan

        p_lcl, t_lcl = self._lifted_condensation_level_from_mixing_ratio(t_start, p_start, r_start)

        theta_parcel = thermo.potential_temperature(t_start, p_start)
        theta_ep_parcel = _ept_from_mixing_ratio(t_start, p_start, r_start, method=self.ept_method)

        # Dry adiabatic ascent to LCL
        between_start_and_lcl = (p > p_lcl[None, ...]) * (p <= p_start[None, ...])
        t_parcel[between_start_and_lcl] = thermo.temperature_from_potential_temperature(
            theta_parcel[None, ...], p
        )[between_start_and_lcl]
        r_parcel[between_start_and_lcl] = (r_start[None, ...] * np.ones(p_shape))[between_start_and_lcl]

        # Moist adiabatic ascent
        above_lcl = p_lcl[None, ...] > p
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
        r_parcel[above_lcl] = constants.epsilon * es_t_parcel / (p_2d[above_lcl] - es_t_parcel)

        # Calculate buoyancy
        specific_humidity_arr = thermo.specific_humidity_from_mixing_ratio(r)
        tv_env = thermo.virtual_temperature(t, specific_humidity_arr)

        specific_humidity_parcel = thermo.specific_humidity_from_mixing_ratio(r_parcel)
        tv_parcel = thermo.virtual_temperature(t_parcel, specific_humidity_parcel)
        dtv = tv_parcel - tv_env
        buoyancy = dtv / tv_env

        buoyant_layer_mask = (buoyancy > 0.0) * (t < t_lcl[None, ...])

        lfc_index = buoyant_layer_mask.shape[0] - np.argmax(buoyant_layer_mask[::-1, ...], axis=0) - 1
        el_index = np.argmax(buoyant_layer_mask, axis=0)

        # For now, this gives the pressure at which the parcel is not buoyant i.e. it does not interpolate
        p_lfc = np.take_along_axis(p, lfc_index[None, ...], axis=0).squeeze(0)
        t_lfc = np.take_along_axis(t, lfc_index[None, ...], axis=0).squeeze(0)

        p_el = np.take_along_axis(p, el_index[None, ...], axis=0).squeeze(0)
        t_el = np.take_along_axis(t, el_index[None, ...], axis=0).squeeze(0)

        buoyant_mask = np.max(buoyant_layer_mask, axis=0)
        p_lfc = np.where(buoyant_mask, p_lfc.astype(float), np.nan)

        return buoyancy, p_lcl, t_lcl, p_lfc, t_lfc, p_el, t_el, tv_parcel, tv_env

    def _sort_pressure_levels(self, p, t, r, zh):
        is_sorted = (np.diff(p, axis=0) >= 0).all()
        if not is_sorted:
            sorted_inds = np.argsort(p, axis=0)
            p = np.take_along_axis(p, sorted_inds, axis=0)
            t = np.take_along_axis(t, sorted_inds, axis=0)
            r = np.take_along_axis(r, sorted_inds, axis=0)
            zh = np.take_along_axis(zh, sorted_inds, axis=0)
        return p, t, r, zh

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

    def _determine_parcel(self, p, zh, t, r, layer_depth):
        raise NotImplementedError("This method should be implemented in the subclass")

    def _cape_cin(self, p, zh, t, r):
        # Shapes of all arrays should be (n_vertical_levels, ...) where the vertical axis is the first axis (axis=0)

        # Make sure pressure levels are in ascending order
        p, t, r, zh = self._sort_pressure_levels(p, t, r, zh)

        # Check for NaN values in the input arrays and mask them out in the output
        # If any input value for a vertical profile is NaN, the output for that profile will be NaN
        nan_mask = np.any(np.isnan(p) | np.isnan(t) | np.isnan(r) | np.isnan(zh), axis=0)

        p_start, t_start, r_start = self._determine_parcel(p, zh, t, r, self.layer_depth)

        buoyancy, p_lcl, t_lcl, p_lfc, t_lfc, p_el, t_el, tv_parcel, tv_env = self._lift_parcel(
            p_start, t_start, r_start, p, t, r
        )
        cape, cin = self._integrate_buoyancy(buoyancy, p, zh, p_lfc, p_el)

        cape[nan_mask] = np.nan
        cin[nan_mask] = np.nan

        # TODO include LI calculation and see if we can use earthkit's vertical interpolation function instead of the custom Interpolate function from the reference implementation
        # [LI] = Interpolate(pressure_arr, [dTv], 500)
        # LI = -LI
        # LI[np.isnan(LI)] = 0

        if self.output == "full":
            # TODO return structured output instead of tuple?
            return (
                cape,
                cin,
                p_start,
                t_start,
                r_start,
                p_lcl,
                t_lcl,
                p_lfc,
                t_lfc,
                p_el,
                t_el,
                tv_parcel,
                tv_env,
            )
        return cape, cin


class _CapeCinSurface(_CapeCinComp):
    def _determine_parcel(self, p, zh, t, r, layer_depth):
        return p[-1], t[-1], r[-1]


class _CapeCinMixed(_CapeCinComp):
    def _determine_parcel(self, p, zh, t, r, layer_depth=None):
        """
        Compute mixed-layer parameters
        :param p: pressure array in Pa
        :param t: temperature array in K
        :param r: mixing ratio array in kg/kg
        :param layer_depth: in Pa
        :return:
        bottom pressure, mixed-layer t, mixed_layer r
        """

        if layer_depth is None:
            layer_depth = 5000

        p_bottom = p[-1, :]
        p_bound = p_bottom - layer_depth
        indx = (np.abs(p - p_bound)).argmin(axis=0)
        p_top = np.take_along_axis(p, indx[None, ...], axis=0).squeeze(0)

        theta = thermo.potential_temperature(t, p)
        theta[(p > p_bottom) | (p < p_top)] = np.nan
        theta_mean = np.nanmean(theta, axis=0)
        t_mixed = thermo.temperature_from_potential_temperature(theta_mean, p_bottom)

        r_copy = np.copy(r)
        r_copy[(p > p_bottom) | (p < p_top)] = np.nan
        r_mixed = np.nanmean(r_copy, axis=0)

        return p_bottom, t_mixed, r_mixed


class _CapeCinMostUnstable(_CapeCinComp):
    def _determine_parcel(self, p, zh, t, r, layer_depth=None):
        t_shape = t.shape

        theta_ep_env = _ept_from_mixing_ratio(t, p, r, method=self.ept_method)
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

        # TODO can we vectorise this loop? less readable but potentially faster
        for k_candidate in np.arange(0, localmaxarg.shape[0]):

            start_level_indices = localmaxarg[k_candidate, ...]
            p_start_candidate = np.take_along_axis(p, start_level_indices[None, ...], axis=0).squeeze(0)
            t_start_candidate = np.take_along_axis(t, start_level_indices[None, ...], axis=0).squeeze(0)
            r_start_candidate = np.take_along_axis(r, start_level_indices[None, ...], axis=0).squeeze(0)

            buoyancy, _, _, _, _, _, _, _, _ = self._lift_parcel(
                p_start_candidate, t_start_candidate, r_start_candidate, p, t, r
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
        r_start = np.take_along_axis(r, start_index_max[None, ...], axis=0).squeeze(0)
        return p_start, t_start, r_start


_CapeCinComp.CM = {
    "surface": _CapeCinSurface,
    "mixed": _CapeCinMixed,
    "mu": _CapeCinMostUnstable,
}


def cape_cin(
    p,
    zh,
    t,
    r,
    parcel_type,
    layer_depth=None,
    output="cape_cin",
    vertical_axis=0,
    ept_method="bolton39",
    lcl_method="davies",
):
    # TODO add options for output: "cape_cin", "cape_cin_li", "full" where full includes parcel_path and intermediate variables for debugging/validation
    # For full output, we need to decide on a format for the output, e.g. a dictionary or a structured array, and we need to handle the case where vertical_axis is not 0
    if output not in ["cape_cin"]:
        raise ValueError(f"Invalid output option '{output}'")

    if vertical_axis != 0:
        if vertical_axis == -1:
            vertical_axis = p.ndim - 1
        if vertical_axis < 0 or vertical_axis >= p.ndim:
            raise ValueError(
                f"Invalid vertical_axis {vertical_axis} for input arrays with {p.ndim} dimensions"
            )

        p = np.swapaxes(p, 0, vertical_axis)
        zh = np.swapaxes(zh, 0, vertical_axis)
        t = np.swapaxes(t, 0, vertical_axis)
        r = np.swapaxes(r, 0, vertical_axis)

    return (
        _CapeCinComp(layer_depth=layer_depth, output=output, lcl_method=lcl_method, ept_method=ept_method)
        .make(parcel_type)
        ._cape_cin(p, zh, t, r)
    )
