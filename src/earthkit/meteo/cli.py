# (C) Copyright 2022 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import click


@click.command("potential_temperature")
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option(
    "--temperature-param",
    default="t",
    show_default=True,
    help="Param/shortName (GRIB) or variable name (NetCDF) of the temperature field(s).",
)
@click.option(
    "--pressure-param",
    default=None,
    help="Param/shortName (GRIB) or variable name (NetCDF) of the pressure field(s). If not given, "
    "the pressure is inferred from the temperature field's metadata (GRIB) or from a pressure-level "
    "coordinate (NetCDF).",
)
def potential_temperature(input, output, temperature_param, pressure_param):
    """Compute the potential temperature from temperature in a GRIB or NetCDF file.

    INPUT is the path to a GRIB or NetCDF file containing the temperature field(s) (and, optionally,
    the pressure field(s)). OUTPUT is the path the resulting file is written to, in the same
    representation (FieldList/GRIB or xarray/NetCDF) as INPUT.
    """
    import earthkit.data as ekd

    ds = ekd.from_source("file", input)
    kind = _preferred_kind(ds, input)

    if kind == "fieldlist":
        _potential_temperature_fieldlist(ds.to_fieldlist(), output, temperature_param, pressure_param, input)
    else:
        _potential_temperature_xarray(ds.to_xarray(), output, temperature_param, pressure_param, input)


def _preferred_kind(ds, input):
    """Decide whether to convert ``ds`` to a FieldList or an xarray.Dataset.

    Whichever of ``"fieldlist"`` or ``"xarray"`` comes first in ``ds.available_types`` wins.
    """
    types = list(ds.available_types)
    candidates = [kind for kind in ("fieldlist", "xarray") if kind in types]
    if not candidates:
        raise click.ClickException(
            f"Cannot convert {input!r} to a FieldList or an xarray.Dataset (available types: {', '.join(types)})"
        )
    return min(candidates, key=types.index)


def _potential_temperature_fieldlist(ds, output, temperature_param, pressure_param, input):
    from earthkit.meteo import thermo

    t = ds.sel(**{"parameter.variable": temperature_param})
    if len(t) == 0:
        raise click.ClickException(f"No fields found for temperature param {temperature_param!r} in {input!r}")

    p = ds.sel(**{"parameter.variable": pressure_param}) if pressure_param is not None else None
    if pressure_param is not None and len(p) == 0:
        raise click.ClickException(f"No fields found for pressure param {pressure_param!r} in {input!r}")

    theta = thermo.potential_temperature(t, p)
    theta.to_target("file", output)


def _potential_temperature_xarray(ds, output, temperature_param, pressure_param, input):
    from earthkit.meteo import thermo

    if temperature_param not in ds:
        raise click.ClickException(f"No variable {temperature_param!r} found in {input!r}")
    t = ds[temperature_param]

    if pressure_param is not None:
        if pressure_param not in ds:
            raise click.ClickException(f"No variable {pressure_param!r} found in {input!r}")
        p = ds[pressure_param]
    else:
        p = _pressure_from_coords(t, input)

    theta = thermo.potential_temperature(t, p).rename("pt")
    theta.attrs = {
        "standard_name": "air_potential_temperature",
        "long_name": "Potential temperature",
        "units": "K",
    }
    theta.to_dataset().to_netcdf(output)


def _pressure_from_coords(da, input):
    """Build a pressure (Pa) DataArray from a pressure-level coordinate of ``da``."""
    for name, coord in da.coords.items():
        if coord.attrs.get("standard_name") == "air_pressure":
            units = coord.attrs.get("units", "Pa")
            factor = {"pa": 1.0, "hpa": 100.0}.get(units.lower())
            if factor is None:
                raise click.ClickException(f"Unsupported pressure units {units!r} for coordinate {name!r} in {input!r}")
            return coord * factor

    raise click.ClickException(
        f"Could not infer pressure from the metadata of {input!r}. Use --pressure-param to specify "
        "the pressure variable explicitly."
    )


COMMANDS = {
    "potential_temperature": potential_temperature,
}
