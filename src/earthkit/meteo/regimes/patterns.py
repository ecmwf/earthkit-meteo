# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import abc
import collections.abc
import functools
import operator

from earthkit.utils.array import array_namespace


class RegimePatterns(abc.ABC):
    """Collection of weather regime patterns.

    Parameters
    ----------
    regimes : Iterable[str]
        Names of the regimes. The ordering here determines the ordering of
        regimes in all outputs.
    grid : dict
        The grid on which the regime patterns live.
    """

    def __init__(self, regimes, grid):
        self._regimes = tuple(regimes)
        self._grid = grid

    @property
    def regimes(self):
        """Names of the regime patterns."""
        return self._regimes

    @property
    def grid(self):
        """Grid specification of the regime patterns."""
        return self._grid

    @property
    def shape(self):
        """Shape of a regime pattern."""
        # TODO placeholder until this functionality is available from earthkit-geo
        lat0, lon0, lat1, lon1 = self.grid["area"]
        dlat, dlon = self.grid["grid"]
        return (int(abs(lat0 - lat1) / dlat) + 1, int(abs(lon0 - lon1) / dlon) + 1)

    def size(self):
        """Number of grid points of a regime pattern."""
        return functools.reduce(operator.mul, self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    @abc.abstractmethod
    def patterns(self, **patterns_extra_coords) -> collections.abc.Mapping:
        """Patterns for all regimes."""

    # While it would be nice to expose the this to the user, keep it internal
    # for now until a more elegant solution is found. Ideally, this function
    # would only take **pattern_extra_coords, but the ordering of the
    # dimensions matters and the dimensions/coordinates of the pattern itself
    # need to be added in. We might be able to get the latter from the gridspec
    # at some point. For now, everything is taken from a reference dataset to
    # ensure that dimension order and naming matches.
    def _patterns_iterxr(self, reference_da, patterns_extra_coords):
        """Patterns for all regimes as xarray DataArrays.

        Parameters
        ----------
        reference_da : xr.DataArray
            Reference dataarray to take coordinates and dimension orders from.
        patterns_extra_coords : Mapping[str, str]
            Mapping of extra coordinates argument names (as given to .patterns)
            to DataArray coordinate names (as used in reference_da).
        """
        import numpy as np
        import xarray as xr

        # Extra coordinate dims, in order of reference dims
        extra_dims = [dim for dim in reference_da.dims if dim in patterns_extra_coords.values()]
        # Output dimensions and coordinates of the patterns
        dims = [*extra_dims, *reference_da.dims[-self.ndim :]]
        coords = {dim: reference_da.coords[dim] for dim in dims}
        # Cartesian product of coordinates for patterns generator
        extra_coords_arrs = dict(zip(extra_dims, np.meshgrid(*(coords[dim] for dim in extra_dims))))
        # Rearrange to match provided kwarg-coord mapping
        extra_coords = {
            kwarg: extra_coords_arrs[patterns_extra_coords[kwarg]] for kwarg in patterns_extra_coords
        }
        # Delegate the pattern generation and DataArray-ify the patterns
        for regime, patterns in self.patterns(**extra_coords).items():
            yield regime, xr.DataArray(patterns, coords=coords, dims=dims)

    def __repr__(self):
        return f"{self.__class__.__name__}{self.regimes}"


class DeferredRegimePatternsDict(collections.abc.Mapping):
    """Mapping that evaluates regime patterns on access.

    Parameters
    ----------
    regimes : Iterable[str]
        Regime names (keys).
    getter : Callable[[str], array_like]
        Function that returns the patterns for a given regime.
    """

    def __init__(self, regimes, getter):
        self._regimes = tuple(regimes)
        assert callable(getter)
        self._getter = getter

    def __getitem__(self, key):
        return self._getter(key)

    def __iter__(self):
        yield from self._regimes

    def __len__(self):
        return len(self._regimes)


class ConstantRegimePatterns(RegimePatterns):
    """Constant regime patterns.

    Parameters
    ----------
    regimes : Iterable[str]
        Regime labels.
    grid : dict
        Grid specification of the patterns.
    patterns : array_like
        Regime patterns.
    """

    def __init__(self, regimes, grid, patterns):
        super().__init__(regimes, grid)
        self._xp = array_namespace(patterns)
        self._patterns = self._xp.asarray(patterns)
        if self._patterns.ndim != 1 + len(self.shape):
            raise ValueError("must have exactly one regime dimension in the patterns")
        if len(self.regimes) != self._patterns.shape[0]:
            raise ValueError("number of regimes does not match number of patterns")

    def patterns(self):
        """Regime patterns.

        Returns
        -------
        dict[str, array_like]
            Regime patterns.
        """
        return dict(zip(self._regimes, self._patterns))


class ModulatedRegimePatterns(RegimePatterns):
    """Regime patterns modulated by a custom scalar function.

    Parameters
    ----------
    regimes : Iterable[str]
        Regime labels.
    grid : dict
        Grid specification of the patterns.
    patterns : array_like
        Base regime patterns.
    modulator : Callable[Any, array_like]
        Scalar function to modulate the base patterns.
    """

    def __init__(self, regimes, grid, patterns, modulator):
        super().__init__(regimes, grid)
        self._xp = array_namespace(patterns)
        self._base_patterns = self._xp.asarray(patterns)
        # Pattern verification
        if self._base_patterns.ndim != 1 + len(self.shape):
            raise ValueError("must have exactly one regime dimension in the patterns")
        if len(self.regimes) != self._base_patterns.shape[0]:
            raise ValueError("number of regimes does not match number of patterns")
        self.modulator = modulator
        if not callable(self.modulator):
            raise ValueError("modulator must be callable")

    def _base_pattern(self, regime):
        return self._base_patterns[self._regimes.index(regime)]

    def patterns(self, **patterns_extra_coords):
        """Regime patterns for a given input to the modulator function.

        Parameters
        ----------
        **patterns_extra_coords : dict[str, Any], optional
            Keyword arguments for the modulator function.

        Returns
        -------
        dict[str, array_like]
            Modulated regime patterns.
        """
        xp = self._xp
        modulator = xp.asarray(self.modulator(**patterns_extra_coords))
        # Adapt to shape of regime patterns
        modulator = modulator[(..., *((xp.newaxis,) * len(self.shape)))]
        return DeferredRegimePatternsDict(
            self._regimes, lambda regime: modulator * self._base_pattern(regime)
        )
