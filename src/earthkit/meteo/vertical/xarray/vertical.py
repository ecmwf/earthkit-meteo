# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import numpy as np
import xarray as xr

from earthkit.meteo import constants
from earthkit.meteo.utils.decorators import xarray_ufunc

from .. import array
from ..array.vertical import _geopotential_below_surface, _temperature_below_surface


def geopotential_height_from_geopotential(z: xr.DataArray) -> xr.DataArray:
    r"""Compute geopotential height from geopotential.

    Parameters
    ----------
    z : xr.DataArray
        Geopotential (m2/s2)

    Returns
    -------
    xr.DataArray
        Geopotential height (m)


    The computation is based on the following definition:

    .. math::

        gh = \frac{z}{g}

    where :math:`g` is the gravitational acceleration on the surface of
    the Earth (see :py:attr:`earthkit.meteo.constants.g`)
    """
    return xarray_ufunc(array.geopotential_height_from_geopotential, z).assign_attrs({
        "standard_name": "geopotential_height",
        "units": "m",
    })


def geopotential_from_geopotential_height(gh: xr.DataArray) -> xr.DataArray:
    r"""Compute geopotential height from geopotential.

    Parameters
    ----------
    gh : xr.DataArray
        Geopotential height (m)

    Returns
    -------
    xr.DataArray
        Geopotential height (m)


    The computation is based on the following definition:

    .. math::

        z = gh  g

    where :math:`g` is the gravitational acceleration on the surface of
    the Earth (see :py:attr:`earthkit.meteo.constants.g`)
    """
    return xarray_ufunc(array.geopotential_from_geopotential_height, gh).assign_attrs({
        "standard_name": "geopotential",
        "units": "m2 s-2",
    })


def geopotential_height_from_geometric_height(h: xr.DataArray, R_earth: float = constants.R_earth) -> xr.DataArray:
    r"""Compute the geopotential height from geometric height.

    Parameters
    ----------
    h : xr.DataArray
        Geometric height with respect to the sea level (m)
    R_earth : float
        Average radius of the Earth (m)

    Returns
    -------
    xr.DataArray
        Geopotential height (m)


    The computation is based on the following formula:

    .. math::

        gh = \frac{h  R_{earth}}{R_{earth} + h}

    where :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`earthkit.meteo.constants.R_earth`)
    """
    return xarray_ufunc(array.geopotential_height_from_geometric_height, h, R_earth).assign_attrs({
        "standard_name": "geopotential_height",
        "units": "m",
    })


def geopotential_from_geometric_height(h: xr.DataArray, R_earth: float = constants.R_earth) -> xr.DataArray:
    r"""Compute the geopotential from geometric height.

    Parameters
    ----------
    h : xr.DataArray
        Geometric height with respect to the sea level (m)
    R_earth : float
        Average radius of the Earth (m)

    Returns
    -------
    xr.DataArray
        Geopotential (m2/s2)


    The computation is based on the following formula:

    .. math::

        z = \frac{h  g  R_{earth}}{R_{earth} + h}

    where

        * :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`earthkit.meteo.constants.R_earth`)
        * :math:`g` is the gravitational acceleration on the surface of
          the Earth (see :py:attr:`earthkit.meteo.constants.g`)
    """
    return xarray_ufunc(array.geopotential_from_geometric_height, h, R_earth).assign_attrs({
        "standard_name": "geopotential",
        "units": "m2 s-2",
    })


def geometric_height_from_geopotential_height(gh: xr.DataArray, R_earth: float = constants.R_earth) -> xr.DataArray:
    r"""Compute the geometric height from geopotential height.

    Parameters
    ----------
    gh : xr.DataArray
        Geopotential height (m)
    R_earth : float
        Average radius of the Earth (m)

    Returns
    -------
    xr.DataArray
        Geometric height (m)


    The computation is based on the following formula:

    .. math::

        h = \frac{R_{earth}  gh}{R_{earth} - gh}

    where :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`earthkit.meteo.constants.R_earth`)
    """
    return xarray_ufunc(array.geometric_height_from_geopotential_height, gh, R_earth).assign_attrs({
        "standard_name": "geometric_height",
        "units": "m",
    })


def geometric_height_from_geopotential(z: xr.DataArray, R_earth: float = constants.R_earth) -> xr.DataArray:
    r"""Compute the geometric height from geopotential.

    Parameters
    ----------
    z : xr.DataArray
        Geopotential (m2/s2)
    R_earth : float
        Average radius of the Earth (m)

    Returns
    -------
    xr.DataArray
        Geometric height (m)


    The computation is based on the following formula:

    .. math::

        h = \frac{R_{earth} \frac{z}{g}}{R_{earth} - \frac{z}{g}}

    where

        * :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`earthkit.meteo.constants.R_earth`)
        * :math:`g` is the gravitational acceleration on the surface of
          the Earth (see :py:attr:`earthkit.meteo.constants.g`)
    """
    return xarray_ufunc(array.geometric_height_from_geopotential, z, R_earth).assign_attrs({
        "standard_name": "geometric_height",
        "units": "m",
    })


_PRESSURE_UNITS = {"Pa": 1.0, "hPa": 100.0}
_HEIGHT_UNITS = {"m": 1.0, "km": 1000.0}


def _to_si(coord: xr.DataArray, name: str, units: dict[str, float] | None) -> xr.DataArray:
    """Convert ``coord`` to SI units with its ``units`` attribute.

    Missing units are taken to be the SI unit of ``name``. ``units`` is the mapping of the
    supported units to their conversion factors; when None no conversion is done.
    """
    if units is None:
        return coord

    value = coord.attrs.get("units")
    if value is None or value == "":
        return coord
    if not isinstance(value, str) or value not in units:
        raise ValueError(
            f'"{name}" has unsupported units: {value!r}. Its "units" attribute must be missing '
            f"or one of {sorted(units)}."
        )
    factor = units[value]
    return coord if factor == 1.0 else coord * factor


def _vertical_coordinate(
    data: xr.DataArray,
    coord: xr.DataArray | None,
    vertical_dim: str,
    name: str,
    units: dict[str, float] | None,
    interfaces: bool,
) -> tuple[xr.DataArray, str]:
    """Resolve the vertical coordinate of a vertical integral.

    ``coord`` is the coordinate given by the user (its name is ``name``) or None, in which
    case the ``vertical_dim`` coordinate of ``data`` is used. It is converted to SI units
    with its ``units`` attribute (``units`` is the mapping of the supported units, None
    means no conversion) and checked. The DataArray and the name of the dimension it is
    defined on are returned.

    The coordinate is defined on the ``N`` data levels when it has the ``vertical_dim``
    dimension of ``data`` and on the ``N + 1`` layer interfaces when it has exactly one
    dimension that is not in ``data`` (only when ``interfaces`` is True). Its other
    dimensions must be dimensions of ``data``; the ones that are missing are broadcast over
    the data (see :func:`_add_leading_axes`). Its vertical coordinate is dropped and the
    values are matched with the data by position along it, while the shared horizontal
    dimensions are matched by label (see :func:`_match_horizontal`).
    """
    nlev = data.sizes[vertical_dim]

    if coord is None:
        if vertical_dim not in data.coords:
            raise ValueError(
                f'"{name}" was not specified and "data" has no coordinate along the vertical '
                f'dimension "{vertical_dim}".'
            )
        coord = data[vertical_dim]
    elif not isinstance(coord, xr.DataArray):
        raise TypeError(f'"{name}" must be an xarray.DataArray or None. Found {type(coord)}.')

    coord = _to_si(coord, name, units)

    outside = tuple(d for d in coord.dims if d not in data.dims)
    if vertical_dim in coord.dims:
        if outside:
            raise ValueError(
                f'"{name}" with dimensions {coord.dims} mixes the vertical dimension '
                f'"{vertical_dim}" of "data" with the dimension(s) {outside} that are not in "data".'
            )
        if coord.sizes[vertical_dim] != nlev:
            hint = (
                f' Define "{name}" on "{vertical_dim}" for the {nlev} data levels or on its own '
                f"dimension with {nlev + 1} values for the layer interfaces."
                if interfaces
                else f' It must be defined on the {nlev} data levels of "data".'
            )
            raise ValueError(
                f'"{name}" has {coord.sizes[vertical_dim]} value(s) along the vertical dimension '
                f'"{vertical_dim}", but "data" has {nlev} level(s).{hint}'
            )
        core_dim = vertical_dim
    else:
        if not interfaces:
            raise ValueError(
                f'"{name}" must be defined on the vertical dimension "{vertical_dim}" of "data". '
                f"Found dimensions {coord.dims}."
            )
        if len(outside) != 1:
            raise ValueError(
                f'"{name}" must have either the vertical dimension "{vertical_dim}" of "data" (for the '
                f'{nlev} data levels) or exactly one dimension that is not in "data" (for the '
                f"{nlev + 1} layer interfaces). Found dimensions {coord.dims}."
            )
        core_dim = outside[0]
        if coord.sizes[core_dim] != nlev + 1:
            raise ValueError(
                f'"{name}" is defined on its own dimension "{core_dim}" with {coord.sizes[core_dim]} '
                f'value(s), but the layer interfaces of the {nlev} levels of "data" must be '
                f'{nlev + 1}. Define "{name}" on "{vertical_dim}" for the data levels instead.'
            )

    if coord.ndim > 1:
        order = tuple(d for d in data.dims if d in coord.dims) + tuple(d for d in coord.dims if d not in data.dims)
        coord = coord.transpose(*order)

    # The coordinate along the vertical dimension of the argument is irrelevant for the
    # integral and its values may even be in a different unit than the ones of "data": it is
    # removed to avoid the alignment of the inputs along it. Its vertical values are matched
    # with the data by position. The other, e.g. horizontal, coordinates are kept and are
    # matched with the ones of the data by label (see "_match_horizontal").
    coord = coord.drop_vars([c for c in coord.coords if core_dim in coord.coords[c].dims])
    coord = _match_horizontal(data, coord, name, core_dim)

    return coord, core_dim


def _match_horizontal(data: xr.DataArray, coord: xr.DataArray, name: str, core_dim: str) -> xr.DataArray:
    """Match ``coord`` with ``data`` by label along their shared indexed dimensions.

    ``coord`` is the ``name`` coordinate argument of a vertical integral, defined on the
    ``core_dim`` vertical dimension. On the dimensions other than ``core_dim`` that are also
    dimensions of ``data`` and carry an index in both objects the values are matched by label:
    the same labels listed in another order reorder ``coord``, different labels raise a
    ValueError. A shared dimension without an index on either side keeps being matched by
    position, the sizes being checked by :func:`xarray.apply_ufunc`.
    """
    for dim in coord.dims:
        if dim == core_dim or dim not in data.dims:
            continue
        if coord.indexes.get(dim) is None or data.indexes.get(dim) is None:
            continue

        coord_labels = coord.coords[dim].values
        data_labels = data.coords[dim].values
        if np.array_equal(coord_labels, data_labels):
            continue
        try:
            same_labels = np.array_equal(np.sort(coord_labels), np.sort(data_labels))
        except TypeError:
            same_labels = False
        if not same_labels:
            raise ValueError(
                f'"{name}" and "data" have different coordinate labels along the dimension "{dim}": '
                f"{_preview_labels(coord_labels)} and {_preview_labels(data_labels)}. The labels must be the "
                f'same when both objects define a coordinate on it; remove the "{dim}" coordinate of '
                f'"{name}" to match its values with the data by position.'
            )
        # the same labels in a different order: reorder the values to the order of the data
        coord = coord.sel({dim: data.coords[dim]})

    return coord


def _preview_labels(labels, count: int = 4) -> str:
    """Format the first values of a coordinate label array for an error message."""
    values = np.asarray(labels)[:count].tolist()
    if len(labels) > count:
        values.append("...")
    return str(values)


def _add_leading_axes(values, ndim: int):
    """Add the leading singleton axes needed to give ``values`` ``ndim`` dimensions.

    The ufunc inputs are passed to the array functions with their core (vertical) dimension
    last and their remaining dimensions in the order of the dimensions of the data. Only the
    leading dimensions the coordinate is missing have to be added: :func:`xarray.apply_ufunc`
    inserts a singleton for every other dimension it does not have.
    """
    extra = ndim - values.ndim
    return values if extra <= 0 else values.reshape((1,) * extra + values.shape)


def _integral_pressure(data, p):
    """Call :func:`earthkit.meteo.vertical.array.vertical_integral_pressure` on the ufunc blocks."""
    return array.vertical_integral_pressure(data, _add_leading_axes(p, data.ndim), vertical_dim=-1)


def _integral_height(data, h, rho=None):
    """Call :func:`earthkit.meteo.vertical.array.vertical_integral_height` on the ufunc blocks."""
    if rho is not None:
        rho = _add_leading_axes(rho, data.ndim)
    return array.vertical_integral_height(data, _add_leading_axes(h, data.ndim), rho=rho, vertical_dim=-1)


def _integral_attrs(data: xr.DataArray, units: str) -> dict:
    """Build the attributes of a vertical integral from the ones of ``data``.

    The ``standard_name`` of the data does not apply to its vertically integrated value and
    is dropped. ``units`` is the unit multiplying the units of the data.
    """
    attrs = dict(data.attrs)

    attrs.pop("standard_name", None)

    label = attrs.pop("long_name", None) or data.name
    if label:
        attrs["long_name"] = f"vertical integral of {label}"

    base = attrs.pop("units", None)
    if base:
        attrs["units"] = f"{base} {units}"

    return attrs


def _vertical_integral(
    func,
    data: xr.DataArray,
    coord: xr.DataArray,
    coord_dim: str,
    vertical_dim: str,
    rho: xr.DataArray | None = None,
    rho_dim: str | None = None,
) -> xr.DataArray:
    """Apply ``func`` to the blocks of the inputs, reducing the ``vertical_dim`` dimension of ``data``.

    ``func`` is one of :func:`_integral_pressure` or :func:`_integral_height`, ``coord`` is
    defined on ``coord_dim`` and ``rho`` on ``rho_dim``.
    """
    args = [data, coord]
    core_dims = [[vertical_dim], [coord_dim]]

    if rho is not None:
        args.append(rho)
        core_dims.append([rho_dim])

    return xarray_ufunc(
        func,
        *args,
        xarray_ufunc_kwargs={
            "input_core_dims": core_dims,
            "output_core_dims": [[]],
            # dask needs a single chunk along the core dimensions
            "dask_gufunc_kwargs": {"allow_rechunk": True},
        },
    )


def vertical_integral_pressure(
    data: xr.DataArray, p: xr.DataArray | None = None, vertical_dim: str = "z"
) -> xr.DataArray:
    r"""Integrate data over pressure.

    *New in version 1.4.0*

    Computes the integral of ``data`` along the ``vertical_dim`` dimension:

    .. math::

        I = \frac{1}{g} \int_{p_{top}}^{p_{bot}} f \, dp

    where :math:`f` are the data values, :math:`p` is the pressure (Pa) and :math:`g` is
    the gravitational acceleration on the surface of the Earth (see
    :py:attr:`earthkit.meteo.constants.g`). See
    :func:`earthkit.meteo.vertical.array.vertical_integral_pressure` for the integration
    schemes used for the level and interface coordinates.

    Parameters
    ----------
    data : xr.DataArray
        Data to be integrated. Its dimension corresponding to the vertical coordinate is
        defined by ``vertical_dim``.
    p : xr.DataArray | None
        Pressure (Pa), either on the :math:`N` data levels or on the :math:`N + 1` layer
        interfaces, as described in the notes of
        :func:`earthkit.meteo.vertical.array.vertical_integral_pressure`. When None
        (default), the ``vertical_dim`` coordinate of ``data`` is used. Its ``units``
        attribute may be "Pa" (default) or "hPa"; any other unit raises a ValueError. The
        dimension of ``p`` along the vertical is either ``vertical_dim``, holding the
        :math:`N` values of the data levels, or a dimension of :math:`N + 1` values that is
        not in ``data``, holding the layer interfaces. Its other dimensions must be
        dimensions of ``data``: the missing ones are broadcast over the data. When ``data``
        and ``p`` both define a coordinate along one of these dimensions, the values of ``p``
        are matched with the data by its labels. The coordinate along the vertical dimension
        is ignored: the vertical values are always matched with the data by position.
    vertical_dim : str
        Name of the vertical dimension in ``data``. Default is "z".

    Returns
    -------
    xr.DataArray
        The vertically integrated data, with the dimensions of ``data`` without
        ``vertical_dim``. Its name is the name of ``data``, its ``long_name`` is "vertical
        integral of <long_name or name of data>" and its ``units`` are the ones of ``data``
        multiplied by "kg m-2" when ``data`` has units. The ``standard_name`` of the data is
        dropped. NaN values in the input propagate to the output.

    Raises
    ------
    ValueError
        If ``vertical_dim`` is not a dimension of ``data``, if ``p`` is None and ``data``
        has no ``vertical_dim`` coordinate, if the units of ``p`` are unsupported, if the
        number of vertical values of ``p`` is neither :math:`N` nor :math:`N + 1` or if
        ``p`` and ``data`` have different coordinate labels along a shared dimension.
    TypeError
        If ``p`` is given and it is not an xarray DataArray.

    See Also
    --------
    vertical_integral_height
    earthkit.meteo.vertical.array.vertical_integral_pressure
    earthkit.meteo.vertical.array.pressure_on_hybrid_levels
    """
    if vertical_dim not in data.dims:
        raise ValueError(f'Vertical dimension "{vertical_dim}" is not in the dimensions of "data": {data.dims}.')

    pressure, p_dim = _vertical_coordinate(data, p, vertical_dim, "p", _PRESSURE_UNITS, interfaces=True)
    result = _vertical_integral(_integral_pressure, data, pressure, p_dim, vertical_dim)
    result.attrs = _integral_attrs(data, "kg m-2")
    return result


def vertical_integral_height(
    data: xr.DataArray,
    h: xr.DataArray | None = None,
    rho: xr.DataArray | None = None,
    vertical_dim: str = "z",
) -> xr.DataArray:
    r"""Integrate data over geometric height.

    *New in version 1.4.0*

    Computes the integral of ``data`` along the ``vertical_dim`` dimension. Without ``rho``:

    .. math::

        I = \int_{z_{bot}}^{z_{top}} f \, dz

    and with ``rho`` the mass-weighted integral:

    .. math::

        I = \int_{z_{bot}}^{z_{top}} \rho f \, dz

    where :math:`f` are the data values, :math:`z` is the geometric height (m) and
    :math:`\rho` is the air density (kg m-3). See
    :func:`earthkit.meteo.vertical.array.vertical_integral_height` for the integration
    schemes used for the level and interface coordinates.

    Parameters
    ----------
    data : xr.DataArray
        Data to be integrated. Its dimension corresponding to the vertical coordinate is
        defined by ``vertical_dim``.
    h : xr.DataArray | None
        Geometric height (m), either on the :math:`N` data levels or on the :math:`N + 1`
        layer interfaces, as described in the notes of
        :func:`earthkit.meteo.vertical.array.vertical_integral_height`. When None (default),
        the ``vertical_dim`` coordinate of ``data`` is used. Its ``units`` attribute may be
        "m" (default) or "km"; any other unit raises a ValueError. The dimension of ``h``
        along the vertical is either ``vertical_dim``, holding the :math:`N` values of the
        data levels, or a dimension of :math:`N + 1` values that is not in ``data``, holding
        the layer interfaces. Its other dimensions must be dimensions of ``data``: the
        missing ones are broadcast over the data. When ``data`` and ``h`` both define a
        coordinate along one of these dimensions, the values of ``h`` are matched with the
        data by its labels. The coordinate along the vertical dimension is ignored: the
        vertical values are always matched with the data by position.
    rho : xr.DataArray | None
        Air density (kg m-3) on the :math:`N` data levels, i.e. defined on ``vertical_dim``.
        It is assumed to be in kg m-3 and not converted. Its values are matched with the
        data by the labels of its coordinates other than ``vertical_dim``, as for ``h``.
        When None (default) the plain integral :math:`\int f dz` is computed.
    vertical_dim : str
        Name of the vertical dimension in ``data``. Default is "z".

    Returns
    -------
    xr.DataArray
        The vertically integrated data, with the dimensions of ``data`` without
        ``vertical_dim``. Its name is the name of ``data``, its ``long_name`` is "vertical
        integral of <long_name or name of data>" and its ``units`` are the ones of ``data``
        multiplied by "m" (without ``rho``) or "kg m-2" (with ``rho``) when ``data`` has
        units. The ``standard_name`` of the data is dropped. NaN values in the input
        propagate to the output.

    Raises
    ------
    ValueError
        If ``vertical_dim`` is not a dimension of ``data``, if ``h`` is None and ``data``
        has no ``vertical_dim`` coordinate, if the units of ``h`` are unsupported, if the
        number of vertical values of ``h`` is neither :math:`N` nor :math:`N + 1`, if
        ``rho`` is not defined on the :math:`N` data levels or if ``h``/``rho`` and ``data``
        have different coordinate labels along a shared dimension.
    TypeError
        If ``h`` or ``rho`` is given and it is not an xarray DataArray.

    See Also
    --------
    vertical_integral_pressure
    earthkit.meteo.vertical.array.vertical_integral_height
    """
    if vertical_dim not in data.dims:
        raise ValueError(f'Vertical dimension "{vertical_dim}" is not in the dimensions of "data": {data.dims}.')

    height, h_dim = _vertical_coordinate(data, h, vertical_dim, "h", _HEIGHT_UNITS, interfaces=True)

    density = None
    density_dim = None
    if rho is not None:
        density, density_dim = _vertical_coordinate(data, rho, vertical_dim, "rho", None, interfaces=False)

    result = _vertical_integral(
        _integral_height,
        data,
        height,
        h_dim,
        vertical_dim,
        rho=density,
        rho_dim=density_dim,
    )
    result.attrs = _integral_attrs(data, "kg m-2" if density is not None else "m")
    return result


def extrapolate_temperature_to_pressure_levels_below_surface(
    t_sfc: xr.DataArray, h_sfc: xr.DataArray, p_sfc: xr.DataArray, target_p: xr.DataArray | float
) -> xr.DataArray:
    r"""Extrapolate temperature from the surface to pressure levels below the surface.

    Parameters
    ----------
    t_sfc : xr.DataArray
        Surface temperature (K)
    h_sfc : xr.DataArray
        Surface height above sea level (m)
    p_sfc : xr.DataArray
        Surface pressure (Pa)
    target_p : xr.DataArray | float
        Target pressure (Pa). Broadcast against the other inputs following xarray rules,
        e.g. ``xr.DataArray([85000.0, 100000.0], dims="z")`` adds a ``z`` dimension.

    Returns
    -------
    xr.DataArray
        Temperature (K)

    See :func:`earthkit.meteo.vertical.array.extrapolate_temperature_to_pressure_levels_below_surface`
    for the algorithm.
    """
    return xarray_ufunc(_temperature_below_surface, t_sfc, h_sfc, p_sfc, target_p).assign_attrs({
        "standard_name": "air_temperature",
        "units": "K",
    })


def extrapolate_geopotential_to_pressure_levels_below_surface(
    t_sfc: xr.DataArray, h_sfc: xr.DataArray, p_sfc: xr.DataArray, target_p: xr.DataArray | float
) -> xr.DataArray:
    r"""Extrapolate geopotential from the surface to pressure levels below the surface.

    Parameters
    ----------
    t_sfc : xr.DataArray
        Surface temperature (K)
    h_sfc : xr.DataArray
        Surface height above sea level (m)
    p_sfc : xr.DataArray
        Surface pressure (Pa)
    target_p : xr.DataArray | float
        Target pressure (Pa). Broadcast against the other inputs following xarray rules,
        e.g. ``xr.DataArray([85000.0, 100000.0], dims="z")`` adds a ``z`` dimension.

    Returns
    -------
    xr.DataArray
        Geopotential (m2/s2)

    See :func:`earthkit.meteo.vertical.array.extrapolate_geopotential_to_pressure_levels_below_surface`
    for the algorithm.
    """
    return xarray_ufunc(_geopotential_below_surface, t_sfc, h_sfc, p_sfc, target_p).assign_attrs({
        "standard_name": "geopotential",
        "units": "m2 s-2",
    })
