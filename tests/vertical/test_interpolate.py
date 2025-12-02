# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo import vertical

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})

NUMPY = [x for x in NAMESPACE_DEVICES if x[0]._earthkit_array_namespace_name == "numpy"]


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "value,pres,target,mode,expected_value",
    [
        (
            [1012.0, 1000.0, 990.0],
            [1012.0, 1000.0, 990.0],
            [1022.0, 1009.0, 995.0, 987.0],
            "linear",
            [np.nan, 1009, 995, np.nan],
        ),
        (
            [[1020.0, 1010.0, 1000.0], [920.0, 910.0, 900.0], [820, 810.0, 800.0]],
            [[1020.0, 1010.0, 1000.0], [920.0, 910.0, 900.0], [820, 810.0, 800.0]],
            [1030.0, 1018.0, 1005.0, 950.0, 914.0, 905.0, 850.0, 814.0, 805.0, 790.0],
            "linear",
            [
                [np.nan, np.nan, np.nan],
                [1018.0, np.nan, np.nan],
                [1005.0, 1005.0, np.nan],
                [950.0, 950.0, 950.0],
                [914.0, 914.0, 914.0],
                [905.0, 905.0, 905.0],
                [850.0, 850.0, 850.0],
                [np.nan, 814.0, 814.0],
                [np.nan, np.nan, 805.0],
                [np.nan, np.nan, np.nan],
            ],
        ),
    ],
)
def test_to_pressure(value, pres, target, mode, expected_value, xp, device):
    value = xp.asarray(value, device=device)
    pres = xp.asarray(pres, device=device)
    target = xp.asarray(target, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.to_pressure(value, pres, target, mode)
    assert xp.allclose(r, expected_value, equal_nan=True)
