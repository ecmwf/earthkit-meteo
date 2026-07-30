# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

# define skip rules for autoapi
def _skip_api_items(app, what, name, obj, skip, options):
    # print(f"{what=} {name=}")

    # if (
    #     what == "module"
    #     and ".array" not in name
    #     and name not in ["earthkit.meteo.solar", "earthkit.meteo.solar.array"]
    # ):
    #     skip = True
    if name in [
        "earthkit.meteo.version",
        "earthkit.meteo.utils",
        "earthkit.meteo.vertical.array.monotonic",
    ]:
        skip = True
    elif what == "module" and ".array." in name:
        skip = True
    elif what == "module" and len(name.split(".")) > 3:
        skip = True
    elif what == "function" and name.endswith("dispatch"):
        skip = True
    elif name.endswith("ArrayLike"):
        skip = True

    # if not skip:
    #     print(f"{what} {name}")
    return skip
