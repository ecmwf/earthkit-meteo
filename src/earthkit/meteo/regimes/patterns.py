# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import abc
import collections.abc

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
        """Shape of the regime patterns."""
        # TODO placeholder until this functionality is available from earthkit-geo
        lat0, lon0, lat1, lon1 = self.grid["area"]
        dlat, dlon = self.grid["grid"]
        return (int(abs(lat0 - lat1) / dlat) + 1, int(abs(lon0 - lon1) / dlon) + 1)

    @abc.abstractmethod
    def patterns(self, **kwargs) -> collections.abc.Mapping:
        """Patterns for all regimes."""

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

    def patterns(self, **kwargs):
        """Regime patterns for a given input to the modulator function.

        Parameters
        ----------
        **kwargs : dict[str, Any], optional
            Keyword arguments for the modulator function.

        Returns
        -------
        dict[str, array_like]
            Modulated regime patterns.
        """
        xp = self._xp
        modulator = xp.asarray(self.modulator(**kwargs))
        # Adapt to shape of regime patterns
        modulator = modulator[(..., *((xp.newaxis,) * len(self.shape)))]
        return DeferredRegimePatternsDict(
            self._regimes, lambda regime: modulator * self._base_pattern(regime)
        )
