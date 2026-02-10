# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


def project(field, patterns, weights, **patterns_extra_coords):
    from .xarray import project

    return project(field, patterns, weights, **patterns_extra_coords)


def standardise(projections, mean, std):
    from .xarray import standardise

    return standardise(projections, mean, std)
