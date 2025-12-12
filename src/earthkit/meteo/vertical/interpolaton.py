import dataclasses as dc
from typing import Any, Literal, Sequence

import numpy as np
import xarray as xr


__all__ = [
    "TargetCoordinates",
    "interpolate_monotonic",
    "interpolate_to_pressure_levels",
    "interpolate_to_theta_levels",
    "interpolate_to_any",
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
    target_data: xr.DataArray,
    target_coord: TargetCoordinates,
    interpolation: Literal["linear", "log", "nearest"] = "linear",
) -> xr.DataArray:

    # Initializations
    if interpolation not in {"linear", "log", "nearest"}:
        raise ValueError(f"Unknown interpolation: {interpolation}")

    # ... determine direction of target field
    dtdz = target_data.diff("z")
    positive = np.all(dtdz > 0)

    if not positive and not np.all(dtdz < 0):
        raise ValueError("target data is not monotonic in the vertical dimension")

    # Prepare output field field_on_target on target coordinates
    field_on_target = _init_field_with_vcoord(
        data.broadcast_like(target_data), target_coord, np.nan,
    )

    # Interpolate
    # ... prepare interpolation
    tkm1 = target_data.shift(z=1)
    fkm1 = data.shift(z=1)

    # ... loop through target values
    for target_idx, t0 in enumerate(target_coord):
        # ... find the 3d field where pressure is > p0 on level k
        #     and was <= p0 on level k-1
        # ... extract the index k of the vertical layer at which p2 adopts its minimum
        #     (corresponds to search from top of atmosphere to bottom)
        # ... note that if the condition above is not fulfilled, minind will
        #     be set to k_top
        if positive:
            t2 = target_data.where((target_data < t0) & (tkm1 >= t0))
        else:
            t2 = target_data.where((target_data > t0) & (tkm1 <= t0))

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
    p_data: xr.DataArray,
    p_target: Sequence[float],
    p_target_units: Literal["Pa", "hPa"] = "Pa",
    interpolation: Literal["linear", "log", "nearest"] = "linear",
) -> xr.DataArray:
    """Interpolate a field from model (k) levels to pressure coordinates.

    Example for vertical interpolation to isosurfaces of a target field,
    which is strictly monotonically decreasing with height.

    Parameters
    ----------
    data : xarray.DataArray
        field to interpolate
    p_data : xarray.DataArray
        pressure field in Pa
    p_target : list of float
        pressure target coordinate values
    p_target_units : str
        pressure target coordinate units
    interpolation : str
        interpolation algorithm, one of {"linear", "log", "nearest"}

    Returns
    -------
    field_on_target : xarray.DataArray
        field on target (i.e., pressure) coordinates

    """
    # TODO: check that p_data is the pressure field, given in Pa (can only be done
    #       if attributes are consequently set)
    #       print warn message if result contains missing values

    # Initializations
    # ... supported target units and corresponding conversion factors to Pa
    p_target_unit_conversions = dict(Pa=1.0, hPa=100.0)
    if p_target_units not in p_target_unit_conversions.keys():
        raise ValueError(
            f"unsupported value of p_target_units: {p_target_units}"
        )
    # ... supported range of pressure target values (in Pa)
    p_target_min = 1.0
    p_target_max = 120000.0

    # Define vertical target coordinates (target)
    target_factor = p_target_unit_conversions[p_target_units]
    target_values = np.array(sorted(p_target)) * target_factor
    if np.any((target_values < p_target_min) | (target_values > p_target_max)):
        raise ValueError(
            "target coordinate value out of range "
            f"(must be in interval [{p_target_min}, {p_target_max}]Pa)"
        )
    target = TargetCoordinates(
        type_of_level="isobaricInPa",
        values=target_values.tolist(),
    )

    return interpolate_monotonic(data, p_data, target, interpolation)


def interpolate_to_any(
    data: xr.DataArray,
    target_data: xr.DataArray,
    target: TargetCoordinates,
    h_data: xr.DataArray,
    folding_mode: Literal["low_fold", "high_fold", "undef_fold"] = "undef_fold",
) -> xr.DataArray:
    """Interpolate a field from model levels to coordinates w.r.t. an arbitrary field.

    Example for vertical interpolation to isosurfaces of a target field
    that is no monotonic function of height.

    Parameters
    ----------
    data : xarray.DataArray
        field to interpolate (e.g. on model levels)
    target_data : xarray.DataArray
        target field on same levels as data field
    target : TargetCoordinates
        target coordinate definition
    h_data : xarray.DataArray
        height on same levels as data field
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
    field_on_target = _init_field_with_vcoord(data.broadcast_like(target_data), target, np.nan)

    # Interpolate
    # ... prepare interpolation
    tkm1 = target_data.shift(z=1)
    fkm1 = data.shift(z=1)

    # ... loop through tc values
    for t_idx, t0 in enumerate(target.values):
        folding_coord_exception = xr.full_like(h_data[{"z": 0}], False)
        # ... find the height field where target is >= t0 on level k and was <= t0
        #     on level k-1 or where theta is <= th0 on level k
        #     and was >= th0 on level k-1
        h = h_data.where(
            ((target_data >= t0) & (tkm1 <= t0)) | ((target_data <= t0) & (tkm1 >= t0))
        )
        if folding_mode == "undef_fold":
            # ... find condition where more than one interval is found, which
            # contains the target coordinate value
            tmp = xr.where(h.notnull(), 1, 0).sum(dim=["z"])
            folding_coord_exception = tmp.where(tmp > 1).notnull()
        if folding_mode in ("low_fold", "undef_fold"):
            # ... extract the index k of the smallest height at which
            # the condition is fulfilled
            tcind = h.fillna(h_max).argmin(dim="z")
        if folding_mode == "high_fold":
            # ... extract the index k of the largest height at which the condition
            # is fulfilled
            tcind = h.fillna(h_min).argmax(dim="z")

        # ... extract theta and field at level k
        t2 = target_data[{"z": tcind}]
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


def interpolate_to_theta_levels(
    data: xr.DataArray,
    t_data: xr.DataArray,
    t_target: Sequence[float],
    h_data: xr.DataArray,
    t_target_units: Literal["K", "cK"] = "K",
    folding_mode: Literal["low_fold", "high_fold", "undef_fold"] = "undef_fold",
) -> xr.DataArray:
    """Interpolate a field from model levels to potential temperature coordinates.

       Example for vertical interpolation to isosurfaces of a target field
       that is no monotonic function of height.

    Parameters
    ----------
    data : xarray.DataArray
        field to interpolate
    t_data : xarray.DataArray
        potential temperature theta in K
    t_target : list of float
        target coordinate values
    h_data : xarray.DataArray
        height
    t_target_units : str
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
    if t_target_units not in th_tc_unit_conversions.keys():
        raise ValueError(
            f"unsupported value of th_tc_units: {t_target_units}"
        )
    # ... supported range of tc values (in K)
    th_tc_min = 1.0
    th_tc_max = 1000.0

    # Define vertical target coordinates
    # Sorting cannot be exploited for optimizations, since theta is
    # not monotonous wrt to height tc values are stored in K
    tc_values = np.array(t_target) * th_tc_unit_conversions[t_target_units]
    if np.any((tc_values < th_tc_min) | (tc_values > th_tc_max)):
        raise ValueError(
            "target coordinate value "
            f"out of range (must be in interval [{th_tc_min}, {th_tc_max}]K)"
        )
    tc = TargetCoordinates(
        type_of_level="theta",
        values=tc_values.tolist(),
    )

    return interpolate_to_any(data, t_data, tc, h_data, folding_mode)


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
    coords["z"] = xr.IndexVariable("z", vcoord.values, attrs=dc.asdict(vcoord.attrs))
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
