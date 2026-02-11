import numpy as np
# TODO remove scipy dependency if earthkit has suitable interpolation functions
from scipy import interpolate
from earthkit.meteo import thermo
from earthkit.meteo.constants import constants


# TODO check if replacing Cpd_ and Cpv_ with constants.Cpd and constants.Cpv is acceptable given the deviation of results
Cpd_    = 1005.7  # heat capacity at constant pressure for dry air [J/kg-1K-1] # constants.Cpd
Cpv_    = 1870.0 # heat capacity at constant pressure of water vapour [J/kg-1K-1] # constants.Cpv


def ept_from_mixing_ratio(temperature, pressure, mixing_ratio):
    specific_humidity = thermo.specific_humidity_from_mixing_ratio(mixing_ratio)
    return thermo.ept_from_specific_humidity(temperature, specific_humidity, pressure, method="bolton39")


# TODO check if earthkit has functionality for moist adiabatic ascent
def MoistAscentLookupTable():
# still uses hPa internally, but returns pressure in Pa

    def dTdp(T_parcel, pressure_hPa):
        pressure = pressure_hPa
        # moist adiabatic gradient according to Emanuel, 1995 (Eq. 4.7.3) ignoring liquid and solid water, i.e. r_l = 0 and r_t = r
        es_p = es_hPa(T_parcel)
        r_p = constants.epsilon * es_p/pressure 
        Lv = 2501000 - 2370 * (T_parcel - constants.T0)
        aa = - (constants.g / Cpd_) * (1 + r_p) / (1 + r_p * (Cpv_ / Cpd_))
        bb = 1 + (Lv * r_p) / (constants.Rd * T_parcel)
        cc = Lv * Lv * r_p * (1 + r_p / constants.epsilon)
        dd = constants.Rv * np.power(T_parcel, 2) * (Cpd_ + r_p * Cpv_)
        
        dTdz = aa * bb / (1 + (cc / dd))
        dzdp = - (constants.Rd * (T_parcel + 0.608 * r_p)) / (pressure * 100 * constants.g)
        dTdp = dTdz * dzdp * 100

        return dTdp

    def es_hPa(T):
        # Calculate the saturation water vapor (partial) pressure
        # T in [K]
        # es in hPa/mb
        T_C = T - constants.T0 # T in deg C
        esat = 6.112 * np.exp((18.678 - (T_C / 234.5)) * (T_C / (257.14 + T_C )))  # Buck equation
        return esat
    
    max_p = 1100 # in hPa
    min_p = 10 # in hPa
    
    my_T_start = np.arange(180, 320, 2)
    my_r_start = constants.epsilon * (es_hPa(my_T_start) / (max_p - es_hPa(my_T_start)))
    theta_ep_range = ept_from_mixing_ratio(my_T_start, max_p * 100, my_r_start) # convert pressure to Pa for theta_ep function
    
    p_range = np.arange(max_p, min_p, -1)
    
    my_T = np.empty((p_range.shape[0], my_T_start.shape[0]))
    my_r = np.empty((p_range.shape[0], my_r_start.shape[0]))
    
    my_T[0,:] = my_T_start
    my_r[0,:] = my_r_start
    
    for k in range(1, p_range.shape[0]):
        my_T[k,:] = my_T[k - 1, :] + dTdp(my_T[k - 1, :], ((p_range[k - 1] + p_range[k]) / 2)) * (p_range[k] - p_range[k - 1])
        my_r[k, :] = constants.epsilon * es_hPa(my_T[k, :]) / (p_range[k] - es_hPa(my_T[k, :]))

    my_T = my_T[::-10, :]
    p_range = p_range[::-10]

    return my_T, theta_ep_range, p_range * 100  # return pressure in Pa


def _determine_mixed_layer_parcel(pressure, temperature, mixing_ratio, layer_depth=None):
    '''
    Compute mixed-layer parameters
    :param pressure: pressure array in hPa
    :param T: temperature array in K
    :param r: mixing ration array in in kg/kg
    :param layer_depth: in hPa
    :return:
    mixed-layer T, r and bottom_pressure
    '''
    # Check if array is sorted in ascending order. If not, sort it.
    is_sorted = (np.diff(pressure, axis=0) >= 0).all()
    if is_sorted == False:
        sorted_inds = np.argsort(pressure, axis=0)
        pressure = np.take_along_axis(pressure, sorted_inds, axis = 0)
        temperature = np.take_along_axis(temperature, sorted_inds, axis = 0)
        mixing_ratio = np.take_along_axis(mixing_ratio, sorted_inds, axis = 0)

    # Default mixed layer depth is 50 hPa
    if layer_depth == None:
        layer_depth = 5000 # in Pa

    bottom_pressure = pressure[-1, :]
    bound_pressure = bottom_pressure - layer_depth
    indx = (np.abs(pressure - bound_pressure)).argmin(axis=0)
    top_pressure = pressure[indx, np.arange(len(indx))]

    theta = thermo.potential_temperature(temperature, pressure)
    theta[(pressure > bottom_pressure) | (pressure < top_pressure)] = np.nan
    theta_mean = np.nanmean(theta, axis=0)
    T_mixed = thermo.temperature_from_potential_temperature(theta_mean, bottom_pressure)

    r_copy = np.copy(mixing_ratio)
    r_copy[(pressure > bottom_pressure) | (pressure < top_pressure)] = np.nan
    r_mixed = np.nanmean(r_copy, axis=0)

    return bottom_pressure, T_mixed, r_mixed


def _determine_most_unstable_parcel(pressure_arr, zh_arr, T_arr, r_arr, layer_depth=None):
    n_pressures = T_arr.shape[0]
    n_profiles = T_arr.shape[1]

    theta_ep_env = ept_from_mixing_ratio(T_arr, pressure_arr, r_arr) # pseudoequivalent potential temperature
    # TODO should this be layer_depth, with default 50000 Pa?
    mupl = 50000 # top pressure level in Pa below which most unstable parcel will be found

    # find local maxima of theta_ep in the vertical profile at pressures below mupl hPa
    theta_ep_env[pressure_arr < mupl] = np.nan
    theta_ep_copy = np.nan_to_num(theta_ep_env)
    theta_grad = theta_ep_copy[1:, :] - theta_ep_copy[:-1, :]

    # localmax is a boolean array with Trues where a local maximum of theta_ep was found
    maxima = (theta_grad[1:, :] < 0) * (theta_grad[:-1, :] > 0)
    localmax = np.ones((n_pressures, n_profiles), dtype=bool)
    localmax[1:-1, :] = maxima
    trues = localmax.sum(axis=0)
    maxtrues = np.amax(trues)
    
    # localmaxarg is an integer array with values of k where a local maximum of theta_ep was found
    localmaxarg = localmax * np.meshgrid(np.arange(0, n_profiles), np.arange(0, n_pressures))[1]
    localmaxarg = np.sort(localmaxarg, axis=0)

    localmaxarg = localmaxarg[:-maxtrues-1 :-1, :]

    CAPEtmp = np.zeros(n_profiles)
    indices = np.zeros(n_profiles, dtype=int)
    for k in np.arange(0, localmaxarg.shape[0]):
        row_indices = np.arange(0, n_profiles)
        col_indices = localmaxarg[k, :]
        pk_start = pressure_arr[col_indices, row_indices]
        Tk_start = T_arr[col_indices, row_indices]
        rk_start = r_arr[col_indices, row_indices]
        B, dTv, p_LCL, T_LCL, p_LFC, T_LFC, p_EL, T_EL, Tv_parcel, Tv_env = lift_parcel(pk_start, Tk_start, rk_start, pressure_arr, T_arr, r_arr)
        dCAPE = constants.g * ((B[:-1, :] + B[1:, :]) / 2) * (-np.diff(zh_arr, axis=0))
        dCAPE[dCAPE < 0] = 0
        CAPE = np.nansum(dCAPE, axis=0)
        mask = (CAPE > CAPEtmp) * (localmaxarg[k, :] > 0)
        CAPEtmp[mask] = CAPE[mask]
        indices[mask] = localmaxarg[k, :][mask]
    p_start = pressure_arr[indices, np.arange(n_profiles)]
    T_start = T_arr[indices, np.arange(n_profiles)]
    r_start = r_arr[indices, np.arange(n_profiles)]
    return p_start, T_start, r_start

# TODO check difference between earthkit LCL function and Ivan's reference implementation (see below)
# only included for comparison purposes
def lifted_condensation_level_from_mixing_ratio(T_departure, p_departure, r_departure):
    specific_humidity = thermo.specific_humidity_from_mixing_ratio(r_departure)
    dewpoint = thermo.dewpoint_from_specific_humidity(specific_humidity, p_departure)
    T_LCL, p_LCL = thermo.lcl(T_departure, dewpoint, p_departure)
    return p_LCL, T_LCL


def lift_parcel(p_start, T_start, r_start, p_arr, T_arr, r_arr):
    npressures = p_arr.shape[0]
    nprofiles = p_arr.shape[1]
    T_parcel = np.zeros([npressures, nprofiles]) * np.nan
    r_parcel = np.zeros([npressures, nprofiles]) * np.nan


    # TODO lcl differs from earthkit implementation above -- for lcl we see the biggest deviations from reference implementation
    # original implementation of LCL commented out below
    # ------------------------------
    # def lifted_condensation_level(T_departure, p_departure, e_departure):
    #     # Compute pressure p_LCL given temperature, pressure and water vapour pressure at departute level
    #     T_LCL = (2840 / (3.5 * np.log(T_departure) - np.log(e_departure) - 4.805)) + 55  # Bolton's approximation from Emanuel, 1996 (Eq. 4.6.24)
    #     p_LCL = p_departure * np.power((T_LCL / T_departure), Cpd_ / constants.Rd)
    #     return p_LCL, T_LCL
    # water vapour pressure at departure level
    # e_start = (r_start * p_start) / (r_start + constants.epsilon)
    # p_LCL, T_LCL = lifted_condensation_level(T_start, p_start, e_start)
    # ------------------------------

    # Lifted Condensation Level - LCL
    p_LCL, T_LCL = lifted_condensation_level_from_mixing_ratio(T_start, p_start, r_start)

    # Potential temperature of the parcel - conserved for dry adiabatic processes
    theta_parcel = thermo.potential_temperature(T_start, p_start)

    # Pseudoequivalent potential temperature of the parcel
    theta_ep_parcel = ept_from_mixing_ratio(T_start, p_start, r_start)

    # Dry adiabatic ascent to LCL
    # ------------------------------
    between_start_and_LCL = (p_arr > p_LCL[None, :]) * (p_arr <= p_start[None, :])
    T_parcel[between_start_and_LCL] = thermo.temperature_from_potential_temperature(theta_parcel[None, :], p_arr)[between_start_and_LCL]
    r_parcel[between_start_and_LCL] = (r_start[None, :] * np.ones((npressures, nprofiles)))[between_start_and_LCL]

    # Moist adiabatic ascent
    # ------------------------------
    # first integrate to the first full pressure level above the LCL
    # for k in range(npressures - 1, 0, -1): # loop from top to bottom
    # boolean mask with True values where given point is first level above LCL
    above_LCL = (p_LCL[None, :] > p_arr)
    p_2d = p_arr * np.ones((npressures, nprofiles))
    theta_ep_parcel_2d = theta_ep_parcel[None, :] * np.ones((npressures, nprofiles))

    # Create Lookup table for moist ascent and define functions:
    my_Tp, theta_ep_range, p_range = MoistAscentLookupTable()
    # TODO replace scipy interpolate with earthkit interpolation function if available
    T_p_lookup = interpolate.RectBivariateSpline(p_range, theta_ep_range, my_Tp)
    T_parcel[above_LCL] = T_p_lookup(p_2d[above_LCL], theta_ep_parcel_2d[above_LCL], grid=False)
    es_T_parcel = thermo.saturation_vapour_pressure(T_parcel[above_LCL], phase="water")
    r_parcel[above_LCL] = constants.epsilon * es_T_parcel / (p_2d[above_LCL] - es_T_parcel)

    # Calculate buoyancy
    specific_humidity_arr = thermo.specific_humidity_from_mixing_ratio(r_arr)
    Tv_env = thermo.virtual_temperature(T_arr, specific_humidity_arr)

    specific_humidity_parcel = thermo.specific_humidity_from_mixing_ratio(r_parcel)
    Tv_parcel = thermo.virtual_temperature(T_parcel, specific_humidity_parcel)
    dTv = Tv_parcel - Tv_env
    B = dTv / Tv_env
    #B[~above_LCL] = 0

    buoyant_3d = (B > 0.0) * (T_arr < T_LCL[None, :])
    
    LFC_index = buoyant_3d.shape[0] - np.argmax(buoyant_3d[::-1, :], axis=0) - 1
    EL_index = np.argmax(buoyant_3d, axis=0)


    # For now, this gives the pressure at which the parcel is not buoyant i.e. it does not interpolate
    p_LFC = p_arr[LFC_index, np.arange(0, p_arr.shape[1])]
    T_LFC = T_arr[LFC_index, np.arange(0, p_arr.shape[1])]

    p_EL = p_arr[EL_index, np.arange(0, p_arr.shape[1])]
    T_EL = T_arr[EL_index, np.arange(0, p_arr.shape[1])]

    buoyant_2d = np.max(buoyant_3d, axis=0)
    p_LFC[~buoyant_2d] = np.nan
    
    return B, dTv, p_LCL, T_LCL, p_LFC, T_LFC, p_EL, T_EL, Tv_parcel, Tv_env


def cape_cin(pressure_arr, zh_arr, T_arr, r_arr, CAPE_type, layer_depth=None):
    # shapes of all arrays should be (n_vertical_levels, n_horizontal_locations)
    # pressure levels should be in ascending order (from top to bottom)
    
    # Make sure pressure levels are in ascending order
    sorted_inds = np.argsort(pressure_arr, axis=0)
    pressure_arr = np.take_along_axis(pressure_arr, sorted_inds, axis=0)
    zh_arr = np.take_along_axis(zh_arr, sorted_inds, axis=0)
    T_arr = np.take_along_axis(T_arr, sorted_inds, axis=0)
    r_arr = np.take_along_axis(r_arr, sorted_inds, axis=0)

    if (CAPE_type == 'surface'):
        # Surface-based parcel is one of those:
        # parcel starting with 2-metre temperature and humidity (if they are in the dataset)
        # parcel starting from the lowest model level or pressure level (if 2m parameteres not in the dataset)
        print("Surface-based parcel")
        p_start = pressure_arr[-1, :]
        T_start = T_arr[-1, :]
        r_start = r_arr[-1, :]
    elif (CAPE_type == 'mixed'):
        print("Mixed-layer parcel")
        p_start, T_start, r_start = _determine_mixed_layer_parcel(pressure_arr, T_arr, r_arr, layer_depth)
    elif (CAPE_type == 'mu'):
        print("Most-unstable parcel")
        p_start, T_start, r_start = _determine_most_unstable_parcel(pressure_arr, zh_arr, T_arr, r_arr, layer_depth)
    else:
        raise NotImplementedError(f"CAPE type '{CAPE_type}' not implemented")
        
    B, dTv, p_LCL, T_LCL, p_LFC, T_LFC, p_EL, T_EL, Tv_parcel, Tv_env = lift_parcel(p_start, T_start, r_start, pressure_arr, T_arr, r_arr)
    
    dCAPE = constants.g * ((B[:-1, :] + B[1:, :]) / 2) * (-np.diff(zh_arr, axis=0))
    dCIN = np.copy(dCAPE)
    dCAPE[dCAPE < 0] = 0
    above_LFC = (pressure_arr[1:, :] <= p_LFC[None, :])
    dCAPE[above_LFC == False] = 0
    CAPE = np.nansum(dCAPE, axis=0)
    CAPE[np.isnan(p_LFC)] = 0
    
    # dCIN[above_LFC] = 0
    above_EL = (pressure_arr[1:, :] <= p_EL[None, :])
    dCIN[above_EL] = 0
    dCIN[dCIN > 0] = 0

    CIN = -np.nansum(dCIN, axis=0)
    CIN[CAPE <= 1] = 0

    # TODO include LI calculation and see if we can use earthkit's vertical interpolation function instead of the custom Interpolate function from the reference implementation
    # [LI] = Interpolate(pressure_arr, [dTv], 500)
    # LI = -LI
    # LI[np.isnan(LI)] = 0


    return CAPE, CIN #, LI, p_start, T_start, p_LFC, T_LFC, p_LCL, T_LCL, p_EL, Tv_parcel, Tv_env
