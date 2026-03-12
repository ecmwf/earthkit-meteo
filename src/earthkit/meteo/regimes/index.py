# (C) Copyright 2026- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

from ..utils.decorators import dispatch

__all__ = ["project", "regime_index"]


def project(field, patterns, weights, **patterns_extra_coords):
    """Project onto the given patterns.

    Parameters
    ----------
    field : xarray.DataArray | array_like
        Input field(s) to project. The patterns are projected onto the trailing
        dimensions of the input fields.
    patterns : earthkit.meteo.regimes.Patterns
        Patterns to project on.
    weights : xarray.DataArray | array_like
        Weights for the summation in the projection. Weights are normalised
        before application so the sum of weights over the domain equals 1.
    **patterns_coords : dict[str,Any], optional
        Coordinates for the pattern generation function.

    Returns
    -------
    xarray.DataArray | array_like
        The projection(s) for each pattern.


    .. admonition:: Implementations

        Depending on the type of argument `field`, this function calls:

        - :py:func:`earthkit.meteo.regimes.xarray.project` for ``xarray.DataArray``
        - :py:func:`earthkit.meteo.regimes.array.project` for ``array_like``
    """
    dispatched = dispatch(project, xarray=True, array=True)
    return dispatched(field, patterns, weights, **patterns_extra_coords)


def regime_index(projections, mean, std):
    """Regime index by standardisation of projections onto patterns.

    Parameters
    ----------
    projections : xarray.DataArray
        Projections onto regime patterns.
    mean : xarray.DataArray
    std : xarray.DataArray

    Returns
    -------
    xarray.DataArray
        ``(projection - mean) / std``


    .. admonition:: Implementations

        Depending on the type of argument `projections`, this function calls:

        - :py:func:`earthkit.meteo.regimes.xarray.regime_index` for ``xarray.DataArray``
    """
    # Array-variant of project returns a dict with array values, can't dispatch on that
    dispatched = dispatch(regime_index, xarray=True, array=False)
    return dispatched(projections, mean, std)
