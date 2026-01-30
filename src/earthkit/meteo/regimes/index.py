# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from . import array


def project(field, patterns, weights, **patterns_kwargs):
    return array.project(field, patterns, weights, **patterns_kwargs)


def standardise(projections, mean, std):
    """Regime index by standardisation of regime projections.

    Parameters
    ----------
    projections : TODO
        Projections onto regime patterns.
    mean : TODO
    std : TODO

    Returns
    -------
    TODO
        ``(projection - mean) / std``
    """
    return (projections - mean) / std
