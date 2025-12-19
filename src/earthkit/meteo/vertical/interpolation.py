import dataclasses as dc
from typing import Any, Literal, Sequence

import numpy as np
import xarray as xr


__all__ = [
    "TargetCoordinates",
    "interpolate_monotonic",
    "interpolate_to_pressure_levels",
    "interpolate_sleve_to_coord_levels",
    "interpolate_sleve_to_theta_levels",
]


@dc.dataclass
class TargetCoordinates:
    """Target Coordinates."""

    type_of_level: str
    values: Sequence[float]

    @property
    def size(self):
        return len(self.values)


def interpolate_monotonic(
    data: xr.DataArray,
    coord: xr.DataArray,
    target_coord: TargetCoordinates,
    interpolation: Literal["linear", "log", "nearest"] = "linear",
) -> xr.DataArray:
    """Interpolate a field to isolevels of a monotonic target field.

    Example for vertical interpolation to isosurfaces of a target field,
    which is strictly monotonically decreasing with height.

    Parameters
    ----------
    data : xarray.DataArray
        field to interpolate
    coord : xarray.DataArray
        target coordinate field data on the same levels as data
    target_coord : TargetCoordinates
        target coordinate definition
    interpolation : str
        interpolation algorithm, one of {"linear", "log", "nearest"}

    Returns
    -------
    field_on_target : xarray.DataArray
        field on target (i.e., pressure) coordinates

    """
    # Initializations
    if interpolation not in {"linear", "log", "nearest"}:
        raise ValueError(f"Unknown interpolation: {interpolation}")

    # ... determine direction of target field
    dtdz = coord.diff("z")
    positive = np.all(dtdz > 0)

    if not positive and not np.all(dtdz < 0):
        raise ValueError("target data is not monotonic in the vertical dimension")

    # Prepare output field field_on_target on target coordinates
    field_on_target = _init_field_with_vcoord(
        data.broadcast_like(coord), target_coord, np.nan,
    )

    # Interpolate
    # ... prepare interpolation
    tkm1 = coord.shift(z=1)
    fkm1 = data.shift(z=1)

    # ... loop through target values
    for target_idx, t0 in enumerate(target_coord.values):
        # ... find the 3d field where pressure is > p0 on level k
        #     and was <= p0 on level k-1
        # ... extract the index k of the vertical layer at which p2 adopts its minimum
        #     (corresponds to search from top of atmosphere to bottom)
        # ... note that if the condition above is not fulfilled, minind will
        #     be set to k_top
        if positive:
            t2 = coord.where((coord < t0) & (tkm1 >= t0))
        else:
            t2 = coord.where((coord > t0) & (tkm1 <= t0))

        minind = t2.fillna(np.inf).argmin(dim="z")

        # ... extract pressure and field at level k
        t2 = t2[{"z": minind}]
        f2 = data[{"z": minind}]
        # ... extract pressure and field at level k-1
        # ... note that f1 and p1 are both undefined, if minind equals k_top
        f1 = fkm1[{"z": minind}]
        t1 = tkm1[{"z": minind}]

        # ... compute the interpolation weights
        if interpolation == "linear":
            # ... note that p1 is undefined, if minind equals k_top, so ratio will
            # be undefined
            ratio = (t0 - t1) / (t2 - t1)

        if interpolation == "log":
            # ... note that p1 is undefined, if minind equals k_top, so ratio will
            #  be undefined
            ratio = (np.log(t0) - np.log(t1)) / (np.log(t2) - np.log(t1))

        if interpolation == "nearest":
            # ... note that by construction, p2 is always defined;
            #     this operation sets ratio to 0 if p1 (and by construction also f1)
            #     is undefined; therefore, the interpolation formula below works
            #     correctly also in this case
            ratio = xr.where(np.abs(t0 - t1) >= np.abs(t0 - t2), 1.0, 0.0)

        # ... interpolate and update field_on_target
        field_on_target[{"z": target_idx}] = (1.0 - ratio) * f1 + ratio * f2

    return field_on_target


def interpolate_to_pressure_levels(
    data: xr.DataArray,
    p: xr.DataArray,
    target_p: Sequence[float],
    target_p_units: Literal["Pa", "hPa"] = "Pa",
    interpolation: Literal["linear", "log", "nearest"] = "linear",
) -> xr.DataArray:
    """Interpolate a field from model (k) levels to pressure coordinates.

    Example for vertical interpolation to isosurfaces of a target field,
    which is strictly monotonically decreasing with height.

    Parameters
    ----------
    data : xarray.DataArray
        field to interpolate
    p : xarray.DataArray
        pressure field in Pa
    target_p : list of float
        pressure target coordinate values
    target_p_units : str
        pressure target coordinate units
    interpolation : str
        interpolation algorithm, one of {"linear", "log", "nearest"}

    Returns
    -------
    field_on_target : xarray.DataArray
        field on target (i.e., pressure) coordinates

    """
    # TODO: check that p is the pressure field, given in Pa (can only be done
    #       if attributes are consequently set)
    #       print warn message if result contains missing values

    # Initializations
    # ... supported target units and corresponding conversion factors to Pa
    target_p_unit_conversions = dict(Pa=1.0, hPa=100.0)
    if target_p_units not in target_p_unit_conversions.keys():
        raise ValueError(
            f"unsupported value of target_p_units: {target_p_units}"
        )
    # ... supported range of pressure target values (in Pa)
    target_p_min = 1.0
    target_p_max = 120000.0

    # Define vertical target coordinates (target)
    target_factor = target_p_unit_conversions[target_p_units]
    target_values = np.array(sorted(target_p)) * target_factor
    if np.any((target_values < target_p_min) | (target_values > target_p_max)):
        raise ValueError(
            "target coordinate value out of range "
            f"(must be in interval [{target_p_min}, {target_p_max}]Pa)"
        )
    target = TargetCoordinates(
        type_of_level="isobaricInPa",
        values=target_values.tolist(),
    )

    return interpolate_monotonic(data, p, target, interpolation)


def interpolate_sleve_to_coord_levels(
    data: xr.DataArray,
    h: xr.DataArray,
    coord: xr.DataArray,
    target_coord: TargetCoordinates,
    folding_mode: Literal["low_fold", "high_fold", "undef_fold"] = "undef_fold",
) -> xr.DataArray:
    """Interpolate a field from sleve levels to coordinates w.r.t. an arbitrary field.

    Example for vertical interpolation to isosurfaces of a target field
    that is no monotonic function of height.

    Parameters
    ----------
    data : xarray.DataArray
        field to interpolate (e.g. on model levels)
    h : xarray.DataArray
        height on same levels as data field
    coord : xarray.DataArray
        target field on same levels as data field
    target : TargetCoordinates
        target coordinate definition
    folding_mode : str
        handle when the target is observed multiple times in a column,
        one of {"low_fold", "high_fold", "undef_fold"}

    Returns
    -------
    xarray.DataArray
        field on target coordinates

    """
    folding_modes = ("low_fold", "high_fold", "undef_fold")
    if folding_mode not in folding_modes:
        raise ValueError(f"Unsupported mode: {folding_mode}")

    # ... tc values outside range of meaningful values of height,
    # used in tc interval search (in m amsl)
    h_min = -1000.0
    h_max = 100000.0

    # Prepare output field on target coordinates
    field_on_target = _init_field_with_vcoord(data.broadcast_like(coord), target_coord, np.nan)

    # Interpolate
    # ... prepare interpolation
    tkm1 = coord.shift(z=1)
    fkm1 = data.shift(z=1)

    # ... loop through tc values
    for t_idx, t0 in enumerate(target_coord.values):
        folding_coord_exception = xr.full_like(h[{"z": 0}], False)
        # ... find the height field where target is >= t0 on level k and was <= t0
        #     on level k-1 or where theta is <= th0 on level k
        #     and was >= th0 on level k-1
        ht = h.where(
            ((coord >= t0) & (tkm1 <= t0)) | ((coord <= t0) & (tkm1 >= t0))
        )
        if folding_mode == "undef_fold":
            # ... find condition where more than one interval is found, which
            # contains the target coordinate value
            tmp = xr.where(ht.notnull(), 1, 0).sum(dim=["z"])
            folding_coord_exception = tmp.where(tmp > 1).notnull()
        if folding_mode in ("low_fold", "undef_fold"):
            # ... extract the index k of the smallest height at which
            # the condition is fulfilled
            tcind = ht.fillna(h_max).argmin(dim="z")
        if folding_mode == "high_fold":
            # ... extract the index k of the largest height at which the condition
            # is fulfilled
            tcind = ht.fillna(h_min).argmax(dim="z")

        # ... extract theta and field at level k
        t2 = coord[{"z": tcind}]
        f2 = data[{"z": tcind}]
        # ... extract theta and field at level k-1
        f1 = fkm1[{"z": tcind}]
        t1 = tkm1[{"z": tcind}]

        # ... compute the interpolation weights
        ratio = xr.where(np.abs(t2 - t1) > 0, (t0 - t1) / (t2 - t1), 0.0)

        # ... interpolate and update field on target
        field_on_target[{"z": t_idx}] = xr.where(
            folding_coord_exception, np.nan, (1.0 - ratio) * f1 + ratio * f2
        )

    return field_on_target


def interpolate_sleve_to_theta_levels(
    data: xr.DataArray,
    h: xr.DataArray,
    theta: xr.DataArray,
    target_theta: Sequence[float],
    target_t_units: Literal["K", "cK"] = "K",
    folding_mode: Literal["low_fold", "high_fold", "undef_fold"] = "undef_fold",
) -> xr.DataArray:
    """Interpolate a field from sleve levels to potential temperature coordinates.

    Example for vertical interpolation to isosurfaces of a target field
    that is no monotonic function of height.

    Parameters
    ----------
    data : xarray.DataArray
        field to interpolate
    h : xarray.DataArray
        height
    theta : xarray.DataArray
        potential temperature theta in K
    target_theta : list of float
        target coordinate values
    target_t_units : str
        target coordinate units, one of {"K", "cK"}
    folding_mode : str
        handle when the target is observed multiple times in a column,
        one of {"low_fold", "high_fold", "undef_fold"}

    Returns
    -------
    xarray.DataArray
        field on target (i.e., theta) coordinates

    """
    # TODO: check that th_field is the theta field, given in K
    #       (can only be done if attributes are consequently set)
    #       print warn message if result contains missing values

    # Parameters
    # ... supported folding modes
    folding_modes = ("low_fold", "high_fold", "undef_fold")
    if folding_mode not in folding_modes:
        raise ValueError(f"unsupported mode: {folding_mode}")

    # ... supported tc units and corresponding conversion factor to K
    # (i.e. to the same unit as theta); according to GRIB2
    #     isentropic surfaces are coded in K; fieldextra codes
    #     them in cK for NetCDF (to be checked)
    th_tc_unit_conversions = dict(K=1.0, cK=0.01)
    if target_t_units not in th_tc_unit_conversions.keys():
        raise ValueError(
            f"unsupported value of th_tc_units: {target_t_units}"
        )
    # ... supported range of tc values (in K)
    th_tc_min = 1.0
    th_tc_max = 1000.0

    # Define vertical target coordinates
    # Sorting cannot be exploited for optimizations, since theta is
    # not monotonous wrt to height tc values are stored in K
    tc_values = np.array(target_theta) * th_tc_unit_conversions[target_t_units]
    if np.any((tc_values < th_tc_min) | (tc_values > th_tc_max)):
        raise ValueError(
            "target coordinate value "
            f"out of range (must be in interval [{th_tc_min}, {th_tc_max}]K)"
        )
    tc = TargetCoordinates(
        type_of_level="theta",
        values=tc_values.tolist(),
    )

    return interpolate_sleve_to_coord_levels(data, theta, tc, h, folding_mode)


def _init_field_with_vcoord(
    parent: xr.DataArray,
    vcoord: TargetCoordinates,
    fill_value: Any,
    dtype: np.dtype | None = None,
) -> xr.DataArray:
    """Initialize an xarray.DataArray with new vertical coordinates.

    Properties except for those related to the vertical coordinates,
    and optionally dtype, are inherited from the parent xarray.DataArray.

    Parameters
    ----------
    parent : xarray.DataArray
        parent field
    vcoord: TargetCoordinates
        target vertical coordinates for the output field
    fill_value : Any
        value the data array of the new field is initialized with
    dtype : np.dtype, optional
        fill value data type; defaults to None (in this case
        the data type is inherited from the parent field)

    Returns
    -------
    xarray.DataArray
        new field located at the parent field horizontal coordinates, the target
        coordinates in the vertical and filled with the given value

    """
    # TODO: test that vertical dim of parent is named "generalVerticalLayer"
    # or take vertical dim to replace as argument
    #       be aware that vcoord contains also xr.DataArray GRIB attributes;
    #  one should separate these from coordinate properties
    #       in the interface
    # attrs
    attrs = parent.attrs

    # Metadata handling is not yet implemented
    # attrs = parent.attrs | metadata.override(
    #     parent.metadata, typeOfLevel=vcoord.type_of_level
    # )
    # dims
    sizes = dict(parent.sizes.items()) | {"z": vcoord.size}
    # coords
    # ... inherit all except for the vertical coordinates
    coords = {c: v for c, v in parent.coords.items() if c != "z"}
    # ... initialize the vertical target coordinates
    coords["z"] = xr.IndexVariable("z", vcoord.values)
    # dtype
    if dtype is None:
        dtype = parent.data.dtype

    return xr.DataArray(
        name=parent.name,
        data=np.full(tuple(sizes.values()), fill_value, dtype),
        dims=tuple(sizes.keys()),
        coords=coords,
        attrs=attrs,
    )
