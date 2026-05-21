# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from earthkit.meteo import constants
from earthkit.meteo.utils.decorators import dispatch


def pressure_at_model_levels(A, B, sp, alpha_top="ifs"):
    return dispatch(pressure_at_model_levels, xarray=False, fieldlist=False, array=True)(A, B, sp, alpha_top=alpha_top)


def relative_geopotential_thickness(alpha, delta, t, q):
    return dispatch(relative_geopotential_thickness, xarray=False, fieldlist=False, array=True)(alpha, delta, t, q)


def pressure_at_height_levels(height, t, q, sp, A, B, alpha_top="ifs"):
    return dispatch(pressure_at_height_levels, xarray=False, fieldlist=False, array=True)(
        height, t, q, sp, A, B, alpha_top=alpha_top
    )


def geopotential_height_from_geopotential(z):
    return dispatch(geopotential_height_from_geopotential, xarray=False, fieldlist=False, array=True)(z)


def geopotential_from_geopotential_height(gh):
    return dispatch(geopotential_from_geopotential_height, xarray=False, fieldlist=False, array=True)(gh)


def geopotential_height_from_geometric_height(h, R_earth=constants.R_earth):
    return dispatch(geopotential_height_from_geometric_height, xarray=False, fieldlist=False, array=True)(
        h, R_earth=R_earth
    )


def geopotential_from_geometric_height(h, R_earth=constants.R_earth):
    return dispatch(geopotential_from_geometric_height, xarray=False, fieldlist=False, array=True)(h, R_earth=R_earth)


def geometric_height_from_geopotential_height(gh, R_earth=constants.R_earth):
    return dispatch(geometric_height_from_geopotential_height, xarray=False, fieldlist=False, array=True)(
        gh, R_earth=R_earth
    )


def geometric_height_from_geopotential(z, R_earth=constants.R_earth):
    return dispatch(geometric_height_from_geopotential, xarray=False, fieldlist=False, array=True)(z, R_earth=R_earth)


# TODO: figure out to handle this case gracefully
def hybrid_level_parameters(*args, **kwargs):
    from earthkit.meteo.vertical.array import hybrid

    return hybrid.hybrid_level_parameters(*args, **kwargs)


def pressure_on_hybrid_levels(
    sp,
    A=None,
    B=None,
    levels=None,
    alpha_top="ifs",
    output="full",
    vertical_dim=0,
):
    return dispatch(pressure_on_hybrid_levels, xarray=False, fieldlist=False, array=True)(
        sp, A=A, B=B, levels=levels, alpha_top=alpha_top, output=output, vertical_dim=vertical_dim
    )


def relative_geopotential_thickness_on_hybrid_levels(t, q, sp, A, B, alpha_top="ifs", vertical_dim=0):
    return dispatch(relative_geopotential_thickness_on_hybrid_levels, xarray=False, fieldlist=False, array=True)(
        t, q, sp, A, B, alpha_top=alpha_top, vertical_dim=vertical_dim
    )


def relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta(t, q, alpha, delta, vertical_dim=0):
    return dispatch(
        relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta,
        xarray=False,
        fieldlist=False,
        array=True,
    )(t, q, alpha, delta, vertical_dim=vertical_dim)


def geopotential_on_hybrid_levels(t, q, zs, sp, A, B, alpha_top="ifs", vertical_dim=0):
    return dispatch(geopotential_on_hybrid_levels, xarray=False, fieldlist=False, array=True)(
        t, q, zs, sp, A, B, alpha_top=alpha_top, vertical_dim=vertical_dim
    )


def height_on_hybrid_levels(
    t, q, zs, sp, A, B, alpha_top="ifs", h_type="geometrics", h_reference="ground", vertical_dim=0
):
    return dispatch(height_on_hybrid_levels, xarray=False, fieldlist=False, array=True)(
        t,
        q,
        zs,
        sp,
        A,
        B,
        alpha_top=alpha_top,
        h_type=h_type,
        h_reference=h_reference,
        vertical_dim=vertical_dim,
    )


def interpolate_hybrid_to_pressure_levels(
    data,
    target_p,
    sp,
    A,
    B,
    alpha_top="ifs",
    interpolation: str = "linear",
    aux_bottom_data=None,
    aux_bottom_p=None,
    aux_top_data=None,
    aux_top_p=None,
    vertical_dim=0,
):
    return dispatch(interpolate_hybrid_to_pressure_levels, xarray=False, fieldlist=False, array=True)(
        data,
        target_p,
        sp,
        A,
        B,
        alpha_top=alpha_top,
        interpolation=interpolation,
        aux_bottom_data=aux_bottom_data,
        aux_bottom_p=aux_bottom_p,
        aux_top_data=aux_top_data,
        aux_top_p=aux_top_p,
        vertical_dim=vertical_dim,
    )


def interpolate_hybrid_to_height_levels(
    data,
    target_h,
    t,
    q,
    za,
    sp,
    A,
    B,
    alpha_top="ifs",
    h_type: str = "geometric",
    h_reference: str = "ground",
    interpolation: str = "linear",
    aux_bottom_data=None,
    aux_bottom_h=None,
    aux_top_data=None,
    aux_top_h=None,
    vertical_dim=0,
):
    return dispatch(interpolate_hybrid_to_height_levels, xarray=False, fieldlist=False, array=True)(
        data,
        target_h,
        t,
        q,
        za,
        sp,
        A,
        B,
        alpha_top=alpha_top,
        h_type=h_type,
        h_reference=h_reference,
        interpolation=interpolation,
        aux_bottom_data=aux_bottom_data,
        aux_bottom_h=aux_bottom_h,
        aux_top_data=aux_top_data,
        aux_top_h=aux_top_h,
        vertical_dim=vertical_dim,
    )


def interpolate_pressure_to_height_levels(
    data,
    target_h,
    z,
    zs,
    h_type: str = "geometric",
    h_reference: str = "ground",
    interpolation: str = "linear",
    aux_bottom_data=None,
    aux_bottom_h=None,
    aux_top_data=None,
    aux_top_h=None,
    vertical_dim: int = 0,
):
    return dispatch(interpolate_pressure_to_height_levels, xarray=False, fieldlist=False, array=True)(
        data,
        target_h,
        z,
        zs,
        h_type=h_type,
        h_reference=h_reference,
        interpolation=interpolation,
        aux_bottom_data=aux_bottom_data,
        aux_bottom_h=aux_bottom_h,
        aux_top_data=aux_top_data,
        aux_top_h=aux_top_h,
        vertical_dim=vertical_dim,
    )
