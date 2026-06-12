# (C) Copyright 2026- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import numpy as np
import pandas as pd
import pytest

from earthkit.meteo.utils.testing import NO_EKD

pytestmark = pytest.mark.skipif(NO_EKD, reason="EKD is not installed")

from earthkit.meteo.regimes import ConstantPatterns, ModulatedPatterns, array, fieldlist

GRIDSPEC = {"grid": [1.0, 1.0], "area": [45.0, 0.0, 45.0, 1.0]}


@pytest.fixture
def fields():
    from earthkit.data import Field, FieldList

    return FieldList.from_fields([
        Field.from_components(values=np.asarray([[1.0, 1.0]]), time={"base_datetime": "2020-01-01T00:00", "step": 0}),
        Field.from_components(values=np.asarray([[1.0, 6.0]]), time={"base_datetime": "2020-01-01T00:00", "step": 3}),
        Field.from_components(values=np.asarray([[6.0, 1.0]]), time={"base_datetime": "2020-01-01T00:00", "step": 6}),
        Field.from_components(values=np.asarray([[6.0, 6.0]]), time={"base_datetime": "2020-01-01T00:00", "step": 9}),
    ])


@pytest.mark.parametrize("weights", [np.asarray([[1.0, 1.0]]), np.asarray([[0.2, 0.8]])])
def test_project_with_constant_patterns(fields, weights):
    patterns = ConstantPatterns(labels=["foo", "bar"], grid=GRIDSPEC, patterns=[[[1.0, 1.0]], [[0.1, 0.9]]])
    result = fieldlist.project(fields, patterns, weights)
    reference = array.project(fields.values[:, None, :], patterns, weights)

    assert result.shape == (4, 2)
    np.testing.assert_allclose(result["foo"], reference["foo"])
    np.testing.assert_allclose(result["bar"], reference["bar"])


@pytest.mark.parametrize("weights", [np.asarray([[1.0, 1.0]]), np.asarray([[0.2, 0.8]])])
def test_project_with_valid_time_dependent_patterns(fields, weights):
    patterns = ModulatedPatterns(
        labels=["foo", "bar"],
        grid=GRIDSPEC,
        base_patterns=[[[1.0, 1.0]], [[0.1, 0.9]]],
        modulator=lambda t: pd.to_datetime(t).hour,
    )
    result = fieldlist.project(fields, patterns, weights, t="time.valid_datetime")
    reference = array.project(fields.values[:, None, :], patterns, weights, t=fields.get("time.valid_datetime"))

    assert result.shape == (4, 2)
    np.testing.assert_allclose(result["foo"], reference["foo"])
    np.testing.assert_allclose(result["bar"], reference["bar"])
