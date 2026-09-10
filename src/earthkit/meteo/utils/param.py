# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""
Define earthkit-data Field parameter metadata.

Used ffor the various meteorological parameters earthkit-meteo computes. This metadata includes:

- "variable": define the Field.parameter.variable
- "units": define the Field.parameter.units
- "paramId": define the ecCodes GRIB parameter ID (if applicable)
- "edition": define the ecCodes GRIB edition (if applicable)

The keys of the dictionary are the names of the parameters used only internally by
various earthkit-meteo functions.
"""

_FIELD_PARAMS = {
    "10m_wind_direction": {"variable": "10wdir", "units": "degrees", "paramId": 260260, "edition": ("2",)},
    "10m_wind_speed": {
        "variable": "10si",
        "units": "m/s",
        "paramId": 207,
        "edition": (
            "1",
            "2",
        ),
    },
    "100m_wind_speed": {
        "variable": "100si",
        "units": "m/s",
        "paramId": 228249,
        "edition": (
            "1",
            "2",
        ),
    },
    "200m_wind_speed": {
        "variable": "200si",
        "units": "m/s",
        "paramId": 228241,
        "edition": (
            "1",
            "2",
        ),
    },
    "2m_dewpoint": {
        "variable": "2d",
        "units": "K",
        "paramId": 168,
        "edition": (
            "1",
            "2",
        ),
    },
    "2m_relative_humidity": {
        "variable": "2r",
        "units": "%",
        "paramId": 260242,
        "edition": (
            "1",
            "2",
        ),
    },
    "2m_specific_humidity": {
        "variable": "2sh",
        "units": "kg/kg",
        "paramId": 174096,
        "edition": (
            "1",
            "2",
        ),
    },
    "2m_temperature": {
        "variable": "2t",
        "units": "K",
        "paramId": 167,
        "edition": (
            "1",
            "2",
        ),
    },
    "coriolis": {"variable": "fc", "units": "1/s"},
    "cos_solar_zenith_angle": {
        "variable": "cossza",
        "units": "1",
        "paramId": 214001,
        "edition": (
            "1",
            "2",
        ),
    },
    "cos_solar_zenith_angle_integrated": {"variable": "cossza_integrated", "units": "1"},
    "dewpoint": {"variable": "td", "units": "K"},
    "eqpt": {"variable": "eqpt", "units": "K", "paramId": 4, "edition": ("1", "2")},
    "geometric_height_above_ground": {"variable": "h", "units": "m", "paramId": 3008, "edition": ("1", "2")},
    "geometric_height_above_sea": {"variable": "h", "units": "m"},
    "geometric_vertical_velocity": {"variable": "wz", "units": "m/s", "paramId": 260238, "edition": ("2")},
    "geopotential": {"variable": "z", "units": "m2/s2", "paramId": 129, "edition": ("1", "2")},
    "geopotential_height": {"variable": "gh", "units": "gpm", "paramId": 156, "edition": ("1", "2")},
    "hybrid_alpha": {"variable": "hybrid_alpha", "units": "1"},
    "hybrid_delta": {"variable": "hybrid_delta", "units": "1"},
    "lcl_pressure": {"variable": "p_lcl", "units": "Pa"},
    "lcl_temperature": {"variable": "t_lcl", "units": "K"},
    "mixing_ratio": {"variable": "mass_mixrat", "units": "kg/kg", "paramId": 402000, "edition": ("2")},
    "potential_temperature": {"variable": "pt", "units": "K", "paramId": 3, "edition": ("1", "2")},
    "pressure": {"variable": "pres", "units": "Pa", "paramId": 54, "edition": ("1", "2")},
    "pressure_full_level": {"variable": "pres", "units": "Pa"},
    "pressure_half_level": {"variable": "pres_half", "units": "Pa"},
    "relative_geopotential_thickness": {"variable": "rel_geopot_thick", "units": "m2/s2"},
    "relative_humidity": {"variable": "r", "units": "%", "paramId": 157, "edition": ("1", "2")},
    "saturation_mixing_ratio": {"variable": "ws", "units": "kg/kg"},
    "saturation_mixing_ratio_slope": {"variable": "ws_slope", "units": "kg/kg/K"},
    "saturation_specific_humidity": {"variable": "sqw", "units": "kg/kg"},
    "saturation_specific_humidity_slope": {"variable": "sqw_slope", "units": "kg/kg/K"},
    "saturation_vapour_pressure": {"variable": "swvp", "units": "Pa", "paramId": 261021, "edition": ("2")},
    "saturation_vapour_pressure_slope": {"variable": "swvp_slope", "units": "Pa/K"},
    "sept": {"variable": "sept", "units": "K", "paramId": 5, "edition": ("1", "2")},
    "specific_gas_constant": {"variable": "R", "units": "J/kg/K"},
    "specific_humidity": {"variable": "q", "units": "kg/kg", "paramId": 133, "edition": ("1", "2")},
    "surface_downward_shortwave_radiation": {
        "variable": "avg_sdswrf",
        "units": "W/m2",
        "paramId": 235035,
    },
    "surface_downward_longwave_flux": {
        "variable": "avg_sdlwrf",
        "units": "W/m2",
        "paramId": 235036,
    },
    "temperature": {"variable": "t", "units": "K", "paramId": 130, "edition": ("1", "2")},
    "toa_incident_solar_radiation": {
        "variable": "toa_incident_solar_radiation",
        "units": "W/m2",
    },
    "vapour_pressure": {"variable": "vapp", "units": "Pa", "paramId": 260008, "edition": ("2")},
    "virtual_potential_temperature": {"variable": "vptmp", "units": "K", "paramId": 3012, "edition": ("2")},
    "virtual_temperature": {"variable": "vtmp", "units": "K", "paramId": 300012, "edition": ("1", "2")},
    "wet_bulb_potential_temperature": {"variable": "wbpt", "units": "K", "paramId": 261022, "edition": ("2")},
    "wet_bulb_temperature": {"variable": "wbgt", "units": "K", "paramId": 261014, "edition": ("2")},
    "wind_direction": {"variable": "wdir", "units": "degrees", "paramId": 3031, "edition": ("1", "2")},
    "wind_speed": {"variable": "ws", "units": "m/s", "paramId": 10, "edition": ("1", "2")},
}


class _ParamDB:
    def __init__(self):
        self.params = _FIELD_PARAMS

    def get(self, name):
        return self.params[name]

    def field_parameter_metadata(self, name):
        param_item = self.get(name)
        return {"parameter.variable": param_item["variable"], "parameter.units": param_item["units"]}


FIELD_PARAMS = _ParamDB()
