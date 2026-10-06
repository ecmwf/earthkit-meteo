# (C) Copyright 2022 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import click

"""
Here's how to use it from the shell:

**Basic case — pressure-level GRIB (pressure inferred from the level metadata):**
```bash
earthkit potential_temperature docs/experimental/tuv_pl.grib theta.grib
```
This picks out the `t` fields (default `--temperature-param t`), and since no `--pressure-param` is
given, the pressure is derived from each field's own level metadata.

**When pressure is a separate field in the file (e.g. model-level data with a `sp`/`pres` field):**
```bash
earthkit potential_temperature model_level_data.grib theta.grib \
    --temperature-param t \
    --pressure-param pres
```

**Custom temperature param name** (e.g. if your file uses a different shortName):
```bash
earthkit potential_temperature input.grib output.grib --temperature-param 2t
```

Output is a GRIB file containing the potential-temperature fields (`shortName=pt`, `paramId=3`),
one per input temperature field.
"""


@click.command("potential_temperature")
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option(
    "--temperature-param",
    default="t",
    show_default=True,
    help="GRIB param/shortName of the temperature field(s).",
)
@click.option(
    "--pressure-param",
    default=None,
    help="GRIB param/shortName of the pressure field(s). If not given, the pressure is inferred "
    "from the temperature field's metadata (e.g. from the pressure level).",
)
def potential_temperature(input, output, temperature_param, pressure_param):
    """Compute the potential temperature from temperature in a GRIB file.

    INPUT is the path to a GRIB file containing the temperature field(s) (and, optionally, the
    pressure field(s)). OUTPUT is the path the resulting GRIB file is written to.
    """
    import earthkit.data as ekd

    from earthkit.meteo import thermo

    ds = ekd.from_source("file", input).to_fieldlist()

    t = ds.sel(**{"parameter.variable": temperature_param})
    if len(t) == 0:
        raise click.ClickException(f"No fields found for temperature param {temperature_param!r} in {input!r}")

    p = ds.sel(**{"parameter.variable": pressure_param}) if pressure_param is not None else None
    if pressure_param is not None and len(p) == 0:
        raise click.ClickException(f"No fields found for pressure param {pressure_param!r} in {input!r}")

    theta = thermo.potential_temperature(t, p)
    theta.to_target("file", output)


COMMANDS = {
    "potential_temperature": potential_temperature,
}
