# (C) Copyright 2025- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import abc
import collections.abc
import functools
import operator

from earthkit.utils.array import array_namespace


class Patterns(abc.ABC):
    """Collection of patterns.

    Parameters
    ----------
    labels : Iterable[str]
        Labels for the patterns. The ordering determines the ordering of all
        outputs.
    grid : dict
        The grid on which the patterns live.
    xp : array_namespace, optional
        Array namespace of the generated patterns.
    """

    def __init__(self, labels, grid, xp=None):
        self._labels = tuple(labels)
        self._grid = grid
        self._xp = xp

    @property
    def labels(self):
        """Labels of the patterns."""
        return self._labels

    @property
    def grid(self):
        """The grid on which the patterns live."""
        return self._grid

    @property
    def shape(self):
        """Shape of a single pattern."""
        # TODO placeholder until this functionality is available from earthkit-geo
        lat0, lon0, lat1, lon1 = self.grid["area"]
        dlat, dlon = self.grid["grid"]
        return (int(abs(lat0 - lat1) / dlat) + 1, int(abs(lon0 - lon1) / dlon) + 1)

    @property
    def size(self):
        """Number of grid points in a single pattern."""
        return functools.reduce(operator.mul, self.shape)

    @property
    def ndim(self):
        """Number of dimensions/axes in a single pattern."""
        return len(self.shape)

    @property
    def xp(self):
        """Array namespace of the generated patterns."""
        if self._xp is None:
            import numpy as np

            return np
        return self._xp

    @abc.abstractmethod
    def patterns(self, **patterns_extra_coords) -> collections.abc.Mapping:
        """Patterns evaluated for the given coords (if any)."""

    # While it would be nice to expose the this to the user, keep it internal
    # for now until a more elegant solution is found. Ideally, this function
    # would only take **pattern_extra_coords, but the ordering of the
    # dimensions matters and the dimensions/coordinates of the pattern itself
    # need to be added in. We might be able to get the latter from the gridspec
    # at some point. For now, everything is taken from a reference dataset to
    # ensure that dimension order and naming matches.
    def _patterns_iterxr(self, reference_da, patterns_extra_coords):
        """Patterns evaluated for the given coords (if any) as xr.DataArrays.

        Parameters
        ----------
        reference_da : xr.DataArray
            Reference dataarray to take coordinates and dimension orders from.
        patterns_extra_coords : Mapping[str, str]
            Mapping of extra coordinates argument names (as given to .patterns)
            to DataArray coordinate names (as used in reference_da).
        """
        import xarray as xr

        # Extra coordinate dims, in order of reference dims
        extra_dims = [dim for dim in reference_da.dims if dim in patterns_extra_coords.values()]
        # Output dimensions and coordinates of the patterns
        dims = [*extra_dims, *reference_da.dims[-self.ndim :]]
        coords = {dim: reference_da.coords[dim] for dim in dims}
        # Cartesian product of coordinates for patterns generator
        extra_coords_arrs = dict(
            zip(extra_dims, self.xp.meshgrid(*(coords[dim] for dim in extra_dims), indexing="ij"))
        )
        # Rearrange to match provided kwarg-coord mapping
        extra_coords = {
            kwarg: extra_coords_arrs[patterns_extra_coords[kwarg]] for kwarg in patterns_extra_coords
        }
        # Delegate the pattern generation and package the patterns as DataArrays
        for name, patterns in self.patterns(**extra_coords).items():
            yield name, xr.DataArray(patterns, coords=coords, dims=dims)

    def __repr__(self):
        return f"{self.__class__.__name__}{self.labels}"


class DeferredPatternsDict(collections.abc.Mapping):
    """Mapping that evaluates patterns on access.

    Parameters
    ----------
    labels : Iterable[str]
        Labels for the patterns (keys).
    getter : Callable[[str], array_like]
        Function that returns evaluated patterns for a given label.
    """

    def __init__(self, labels, getter):
        self._labels = tuple(labels)
        assert callable(getter)
        self._getter = getter

    def __getitem__(self, key):
        return self._getter(key)

    def __iter__(self):
        yield from self._labels

    def __len__(self):
        return len(self._labels)


class ConstantPatterns(Patterns):
    """Constant patterns.

    Parameters
    ----------
    labels : Iterable[str]
        Labels for the patterns.
    grid : dict
        The grid on which the patterns live.
    patterns : array_like
        The patterns with the outermost dimension corresponding to the labels.
    """

    def __init__(self, labels, grid, patterns):
        super().__init__(labels, grid, xp=array_namespace(patterns))
        self._patterns = self._xp.asarray(patterns)
        if self._patterns.ndim != 1 + len(self.shape):
            raise ValueError("must have exactly one label axis in the patterns")
        if len(self.labels) != self._patterns.shape[0]:
            raise ValueError("number of labels does not match number of patterns")

    def patterns(self):
        """All patterns.

        Returns
        -------
        dict[str, array_like]
            Mapping from labels to patterns.
        """
        return dict(zip(self._labels, self._patterns))


class ModulatedPatterns(Patterns):
    """Patterns modulated by a custom scalar function.

    Parameters
    ----------
    labels : Iterable[str]
        Labels for the patterns.
    grid : dict
        Grid specification of the patterns.
    patterns : array_like
        Base patterns.
    modulator : Callable[Any, array_like]
        Scalar function to modulate the base patterns.
    """

    def __init__(self, labels, grid, patterns, modulator):
        super().__init__(labels, grid, xp=array_namespace(patterns))
        self._base_patterns = self.xp.asarray(patterns)
        # Pattern verification
        if self._base_patterns.ndim != 1 + len(self.shape):
            raise ValueError("must have exactly one label axis in the patterns")
        if len(self.labels) != self._base_patterns.shape[0]:
            raise ValueError("number of labels does not match number of patterns")
        self.modulator = modulator
        if not callable(self.modulator):
            raise ValueError("modulator must be callable")

    def _base_pattern(self, label):
        return self._base_patterns[self._labels.index(label)]

    def patterns(self, **patterns_extra_coords):
        """Evaluated patterns for a given input to the modulator function.

        Parameters
        ----------
        **patterns_extra_coords : dict[str, Any], optional
            Keyword arguments for the modulator function.

        Returns
        -------
        dict[str, array_like]
            Modulated patterns.
        """
        modulator = self.xp.asarray(self.modulator(**patterns_extra_coords))
        # Adapt to shape of patterns
        modulator = modulator[(..., *((self.xp.newaxis,) * len(self.shape)))]
        return DeferredPatternsDict(self._labels, lambda label: modulator * self._base_pattern(label))
