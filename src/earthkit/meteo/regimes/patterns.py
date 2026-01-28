# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import abc

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
    def get(self, regime, **kwargs):
        """Patterns for a regime."""

    def iter(self, **kwargs):
        """Patterns for all regimes."""
        for regime in self.regimes:
            yield regime, self.get(regime, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}{self.regimes}"


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

    def get(self, regime):
        """Regime patterns.

        Parameters
        ----------
        regime : str
            Name of the regime to provide patterns for.

        Returns
        -------
        dict[str, array_like]
            Regime patterns.
        """
        # Not fast, but the number of regimes is unlikely to be very large
        i = self._regimes.index(regime)
        return self._patterns[i]


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

    @property
    def _base_patterns_dict(self):
        return dict(zip(self._regimes, self._base_patterns))

    def get(self, regime, **kwargs):
        """Regime patterns for a given input to the modulator function.

        Parameters
        ----------
        regime : str
            Name of the regime to provide patterns for.
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
        base_pattern = self._base_patterns_dict[regime]
        return modulator * base_pattern
