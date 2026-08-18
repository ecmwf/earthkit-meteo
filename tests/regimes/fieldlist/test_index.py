# (C) Copyright 2026- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import numpy as np
import pandas as pd
import pytest

ekd = pytest.importorskip("earthkit.data")
ekg = pytest.importorskip("earthkit.geo", reason="regimes require earthkit.geo")

from earthkit.meteo.regimes import ConstantPatterns, ModulatedPatterns, array, fieldlist

GRID_SPEC = {"grid": [1.0, 1.0], "area": [45.0, 0.0, 45.0, 1.0]}


@pytest.fixture
def fields():
    return ekd.FieldList.from_fields([
        ekd.Field.from_components(
            values=np.asarray([[1.0, 1.0]]),
            time={"base_datetime": "2020-01-01T00:00", "step": 0},
            geography={"grid_spec": GRID_SPEC},
        ),
        ekd.Field.from_components(
            values=np.asarray([[1.0, 6.0]]),
            time={"base_datetime": "2020-01-01T00:00", "step": 3},
            geography={"grid_spec": GRID_SPEC},
        ),
        ekd.Field.from_components(
            values=np.asarray([[6.0, 1.0]]),
            time={"base_datetime": "2020-01-01T00:00", "step": 6},
            geography={"grid_spec": GRID_SPEC},
        ),
        ekd.Field.from_components(
            values=np.asarray([[6.0, 6.0]]),
            time={"base_datetime": "2020-01-01T00:00", "step": 9},
            geography={"grid_spec": GRID_SPEC},
        ),
    ])


@pytest.mark.parametrize("weights", [None, np.asarray([[1.0, 1.0]]), np.asarray([[0.2, 0.8]])])
def test_project_with_constant_patterns(fields, weights):
    patterns = ConstantPatterns(labels=["foo", "bar"], patterns=[[[1.0, 1.0]], [[0.1, 0.9]]], grid=GRID_SPEC)
    result = fieldlist.project(fields, patterns, weights)
    reference = array.project(fields.data(keys="value", flatten=False), patterns, weights)
    assert result.shape == (4, 2)
    np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("weights", [None, np.asarray([[1.0, 1.0]]), np.asarray([[0.2, 0.8]])])
def test_project_with_valid_time_dependent_patterns(fields, weights):
    patterns = ModulatedPatterns(
        labels=["foo", "bar"],
        base_patterns=[[[1.0, 1.0]], [[0.1, 0.9]]],
        modulator=lambda t: pd.to_datetime(t).hour,
        grid=GRID_SPEC,
    )
    result = fieldlist.project(fields, patterns, weights, patterns_coords={"t": "time.valid_datetime"})
    reference = array.project(
        fields.data(keys="value", flatten=False),
        patterns,
        weights,
        patterns_coords={"t": fields.get("time.valid_datetime")},
    )
    assert result.shape == (4, 2)
    np.testing.assert_array_equal(result, reference)


def test_project_with_regrid_to_pattern():
    field = ekd.Field.from_components(
        values=np.asarray([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
        geography={"grid_spec": {"grid": [1.0, 1.0], "area": [46.0, -1.0, 44.0, 2.0]}},
    )
    patterns = ConstantPatterns(
        labels=["foo", "bar"],
        grid={"grid": [1.0, 1.0], "area": [45.0, 0.0, 44.0, 1.0]},
        patterns=np.asarray([
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ]),
    )
    result = fieldlist.project(field, patterns, weights=np.ones((2, 2)), regrid_to_pattern=True)
    np.testing.assert_allclose(result, [[1.0, 2.0]])
