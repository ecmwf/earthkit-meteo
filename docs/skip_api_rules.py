# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


# define skip rules for autoapi
def _skip_api_items(app, what, name, obj, skip, options):
    # print(f"{what=} {name=}")

    if (
        what == "module"
        and ".array" not in name
        and name not in ["earthkit.meteo.solar", "earthkit.meteo.solar.array"]
    ):
        skip = True
    elif what == "package" and ".array" not in name and len(name.split(".")) > 2:
        skip = True
    elif what == "function" and ".array" not in name:
        skip = True

    # if not skip:
    #     print(f"{what} {name}")
    return skip
