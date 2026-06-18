from earthkit.meteo.utils.decorators import dispatch


def interpolate_monotonic(
    data,
    coord,
    target_coord,
    interpolation: str = "linear",
    aux_min_level_data=None,
    aux_min_level_coord=None,
    aux_max_level_data=None,
    aux_max_level_coord=None,
    vertical_dim=None,
):
    if vertical_dim is not None:
        return dispatch(interpolate_monotonic, xarray=True, fieldlist=False, array=True)(
            data,
            coord,
            target_coord,
            interpolation=interpolation,
            aux_min_level_data=aux_min_level_data,
            aux_min_level_coord=aux_min_level_coord,
            aux_max_level_data=aux_max_level_data,
            aux_max_level_coord=aux_max_level_coord,
            vertical_dim=vertical_dim,
        )
    else:
        return dispatch(interpolate_monotonic, xarray=True, fieldlist=False, array=True)(
            data,
            coord,
            target_coord,
            interpolation=interpolation,
            aux_min_level_data=aux_min_level_data,
            aux_min_level_coord=aux_min_level_coord,
            aux_max_level_data=aux_max_level_data,
            aux_max_level_coord=aux_max_level_coord,
        )


def interpolate_to_pressure_levels(data, p, target_p, target_p_units="Pa", interpolation="linear", vertical_dim="z"):
    return dispatch(interpolate_to_pressure_levels, xarray=True, fieldlist=False, array=False)(
        data,
        p,
        target_p,
        target_p_units=target_p_units,
        interpolation=interpolation,
        vertical_dim=vertical_dim,
    )


def interpolate_sleve_to_coord_levels(data, h, coord, target_coord, folding_mode="undef_fold", vertical_dim="z"):
    return dispatch(interpolate_sleve_to_coord_levels, xarray=True, fieldlist=False, array=False)(
        data, h, coord, target_coord, folding_mode=folding_mode, vertical_dim=vertical_dim
    )


def interpolate_sleve_to_theta_levels(
    data, g, theta, target_theta, target_t_units="K", folding_mode="undef_fold", vertical_dim="z"
):
    return dispatch(interpolate_sleve_to_theta_levels, xarray=True, fieldlist=False, array=False)(
        data,
        g,
        theta,
        target_theta,
        target_t_units=target_t_units,
        folding_mode=folding_mode,
        vertical_dim=vertical_dim,
    )
