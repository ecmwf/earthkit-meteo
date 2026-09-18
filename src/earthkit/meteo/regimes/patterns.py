# (C) Copyright 2025- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import abc
import functools
import operator

from earthkit.utils.array import array_namespace


def _from_grid_spec(grid):
    from earthkit.geo.grids import Grid

    return Grid(grid)


class Patterns(abc.ABC):
    """Collection/Generator of patterns.

    Parameters
    ----------
    labels : Iterable[str]
        Labels for the patterns. The ordering determines the ordering of all
        outputs.
    shape : tuple[int,...], optional
        The shape of a single pattern (i.e., without the label dimension).
    grid : earthkit.geo.grids.Grid | dict | str, optional
        Specification of the grid on which the patterns live. If provided, the
        pattern shape can be omitted and is inferred from the grid.
    xp : array_namespace, optional
        Array namespace of the generated patterns.
    """

    def __init__(self, labels, *, shape=None, grid=None, xp=None):
        self._labels = tuple(labels)
        self._xp = xp if xp is not None else array_namespace()
        if shape is None and grid is None:
            raise ValueError("must provide shape of a pattern or grid to determine shape")
        self._shape = None if shape is None else tuple(shape)
        if grid is not None:
            self._grid = _from_grid_spec(grid) if isinstance(grid, (str, dict)) else grid
            if self._shape is None:
                self._shape = self._grid.shape
            elif self.shape != self.grid.shape:
                raise ValueError(f"specified shape {self.shape} does not match grid shape {self.grid.shape}")

    @property
    def labels(self):
        """Labels of the patterns."""
        return self._labels

    @property
    def grid(self):
        """The grid on which the patterns live."""
        return self._grid

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of a single pattern."""
        return self._shape if self._shape is not None else self.grid.shape

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
    def patterns(self, **patterns_coords):
        """Patterns evaluated for the given coords (if any)."""

    def __repr__(self):
        return f"{self.__class__.__name__}{self.labels}"

    def __len__(self):
        return len(self._labels)


class ConstantPatterns(Patterns):
    """Collection of constant/fixed patterns.

    Parameters
    ----------
    labels : Iterable[str]
        Labels for the patterns.
    patterns : array_like
        The patterns (one for each label, stacked into a single array).
    grid : dict | str | earthkit.geo.grid.Grid, optional
        Specification of the grid on which the patterns live.
    xp : array_namespace, optional
        The array namespace used for the patterns and their generation. By
        default, it is inferred from the type of `patterns`.
    """

    def __init__(self, labels, patterns, *, xp=None, grid=None):
        if xp is None:
            xp = array_namespace(patterns)
        self._patterns = xp.asarray(patterns)
        shape = self._patterns.shape[1:]  # set shape explicitly from patterns
        super().__init__(labels, xp=xp, shape=shape, grid=grid)
        if len(self.labels) != self._patterns.shape[0]:
            raise ValueError("number of labels does not match number of patterns")

    def patterns(self):
        """Patterns.

        Returns
        -------
        array_like
        """
        return self._patterns


class ModulatedPatterns(Patterns):
    """Patterns generated from base patterns and a custom scalar function.

    The base patterns are multiplied with the return values of the modulation
    function to generate the patterns.

    Parameters
    ----------
    labels : Iterable[str]
        Labels for the patterns.
    base_patterns : array_like
        Base patterns (one for each label, stacked into a single array).
    modulator : Callable[Any,array_like]
        Scalar function to modulate the base patterns. The parameters required
        to evaluate this function must be provided when projecting as
        `patterns_extra_coords` kwargs.
    grid : dict | str | earthkit.geo.grid.Grid, optional
        Specification of the grid on which the patterns live.
    xp : array_namespace, optional
        The array namespace used for the patterns and their generation. By
        default, it is inferred from the type of `base_patterns`.
    """

    def __init__(self, labels, base_patterns, modulator, *, grid=None, xp=None):
        if xp is None:
            xp = array_namespace(base_patterns)
        self._base_patterns = xp.asarray(base_patterns)
        shape = self._base_patterns.shape[1:]  # set shape explicitly from patterns
        super().__init__(labels, shape=shape, grid=grid, xp=xp)
        if len(self.labels) != self._base_patterns.shape[0]:
            raise ValueError("number of labels does not match number of patterns")
        self._modulator = modulator
        if not callable(self._modulator):
            raise ValueError("modulator must be callable")

    def patterns(self, **patterns_coords):
        """Evaluated patterns for a given input to the modulator function.

        Parameters
        ----------
        **patterns_coords : dict[str,Any], optional
            Keyword arguments for the modulator function.

        Returns
        -------
        array_like
            Modulated patterns.
        """
        modulator = self.xp.asarray(self._modulator(**patterns_coords))
        # Adapt to shape of patterns, include patterns as dim
        modulator = modulator[(..., *((self.xp.newaxis,) * (1 + self.ndim)))]
        return modulator * self._base_patterns
