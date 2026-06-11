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
    """Collection/Generator of patterns.

    Parameters
    ----------
    labels : Iterable[str]
        Labels for the patterns. The ordering determines the ordering of all
        outputs.
    grid : dict
        Specification of the grid on which the patterns live.
    xp : array_namespace, optional
        Array namespace of the generated patterns.
    """

    def __init__(self, labels, grid, xp):
        self._labels = tuple(labels)
        self._grid = grid
        self._xp = xp

    @property
    def labels(self):
        """Labels of the patterns."""
        return self._labels

    @property
    def grid(self) -> dict:
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
    def size(self) -> int:
        """Number of grid points in a single pattern."""
        return functools.reduce(operator.mul, self.shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions/axes in a single pattern."""
        return len(self.shape)

    @property
    def xp(self):
        """Array namespace of the generated patterns."""
        return self._xp

    @abc.abstractmethod
    def patterns(self, **patterns_extra_coords) -> collections.abc.Mapping:
        """Patterns evaluated for the given coords (if any)."""

    def __repr__(self):
        return f"{self.__class__.__name__}{self.labels}"

    def __len__(self):
        return len(self._labels)


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
    """Collection of constant/fixed patterns.

    Parameters
    ----------
    labels : Iterable[str]
        Labels for the patterns.
    grid : dict
        Specification of the grid on which the patterns live.
    patterns : array_like
        The patterns (one for each label, stacked into a single array).
    xp : array_namespace, optional
        The array namespace used for the patterns and their generation. By
        default, it is inferred from the type of `base_patterns`.
    """

    def __init__(self, labels, grid, patterns, xp=None):
        if xp is None:
            xp = array_namespace(patterns)
        super().__init__(labels, grid, xp)
        self._patterns = self._xp.asarray(patterns)
        if self._patterns.ndim != 1 + len(self.shape):
            raise ValueError("must have exactly one label axis in the patterns")
        if len(self.labels) != self._patterns.shape[0]:
            raise ValueError("number of labels does not match number of patterns")

    def patterns(self):
        """All patterns.

        Returns
        -------
        dict[str,array_like]
            Mapping from labels to patterns.
        """
        return self._patterns


class ModulatedPatterns(Patterns):
    """Patterns generated from a set of base patterns and a custom scalar function.

    Parameters
    ----------
    labels : Iterable[str]
        Labels for the patterns.
    grid : dict
        Specification of the grid on which the patterns live.
    base_patterns : array_like
        Base patterns (one for each label, stacked into a single array).
    modulator : Callable[Any,array_like]
        Scalar function to modulate the base patterns. The parameters required
        to evaluate this function must be provided when projecting as
        `patterns_extra_coords` kwargs.
    xp : array_namespace, optional
        The array namespace used for the patterns and their generation. By
        default, it is inferred from the type of `base_patterns`.
    """

    def __init__(self, labels, grid, base_patterns, modulator, xp=None):
        if xp is None:
            xp = array_namespace(base_patterns)
        super().__init__(labels, grid, xp)
        self._base_patterns = self.xp.asarray(base_patterns)
        # Pattern verification
        if self._base_patterns.ndim != 1 + len(self.shape):
            raise ValueError("must have exactly one label axis in the patterns")
        if len(self.labels) != self._base_patterns.shape[0]:
            raise ValueError("number of labels does not match number of patterns")
        self._modulator = modulator
        if not callable(self._modulator):
            raise ValueError("modulator must be callable")

    def patterns(self, **patterns_extra_coords):
        """Evaluated patterns for a given input to the modulator function.

        Parameters
        ----------
        **patterns_extra_coords : dict[str,Any], optional
            Keyword arguments for the modulator function.

        Returns
        -------
        Mapping[str,array_like]
            Modulated patterns.
        """
        modulator = self.xp.asarray(self._modulator(**patterns_extra_coords))
        # Adapt to shape of patterns
        modulator = modulator[(..., *((self.xp.newaxis,) * len(self.shape)))]
        return modulator * self._base_patterns
