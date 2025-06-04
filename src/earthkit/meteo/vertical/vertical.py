# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from . import array


def pressure_at_model_levels(*args, **kwargs):
    return array.pressure_at_model_levels(*args, **kwargs)


def relative_geopotential_thickness(*arg, **kwargs):
    return array.relative_geopotential_thickness(*arg, **kwargs)


def pressure_at_height_levels(*args, **kwargs):
    return array.pressure_at_height_levels(*args, **kwargs)


def geopotential_height_from_geopotential(*args, **kwargs):
    return array.geopotential_height_from_geopotential(*args, **kwargs)


def geopotential_from_geopotential_height(*args, **kwargs):
    return array.geopotential_from_geopotential_height(*args, **kwargs)


def geopotential_height_from_geometric_height(*args, **kwargs):
    return array.geopotential_height_from_geometric_height(*args, **kwargs)


def geopotential_from_geometric_height(*args, **kwargs):
    return array.geopotential_from_geometric_height(*args, **kwargs)


def geometric_height_from_geopotential_height(*args, **kwargs):
    return array.geometric_height_from_geopotential_height(*args, **kwargs)


def geometric_height_from_geopotential(*args, **kwargs):
    return array.geometric_height_from_geopotential(*args, **kwargs)
