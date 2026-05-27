# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

_FIELD_PARAMS = {
    "temperature": {"variable": "t", "units": "K"},
    "geopotential": {"variable": "z", "units": "m2/s2"},
    "geopotential_height": {"variable": "z", "units": "m"},
    "height": {"variable": "h", "units": "m"},
    "pressure": {"variable": "pres", "units": "Pa"},
    "relative_humidity": {"variable": "r", "units": "%"},
    "pressure_full_level": {"variable": "pres", "units": "Pa"},
    "pressure_half_level": {"variable": "pres_half", "units": "Pa"},
    "hybrid_alpha": {"variable": "hybrid_alpha", "units": "1"},
    "hybrid_delta": {"variable": "hybrid_delta", "units": "1"},
    "relative_geopotential_thickness": {"variable": "rel_geopot_thick", "units": "m2/s2"},
}


class _ParamDB:
    def __init__(self):
        self.params = _FIELD_PARAMS

    def get(self, name):
        return self.params[name]


FIELD_PARAMS = _ParamDB()
