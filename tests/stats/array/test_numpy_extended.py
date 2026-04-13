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

from earthkit.meteo import stats

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "data,weights,kwargs,v_ref",
    [
        (
            [[[4, 2], [4, np.nan]], [[4, 4], [np.nan, 2]]],
            None,
            dict(axis=-1),
            [[3.0, 4.0], [4.0, 2.0]],
        ),
        (
            [[[4, 2], [4, np.nan]], [[4, 4], [np.nan, 2]]],
            [1, 0.25],
            dict(axis=-1),
            [[3.6, 4.0], [4.0, 2.0]],
        ),
    ],
)
def test_nanaverage(xp, device, data, weights, v_ref, kwargs):
    data, v_ref = xp.asarray(data, device=device), xp.asarray(v_ref, device=device)

    if weights is not None:
        weights = xp.asarray(weights, device=device)

    r = stats.nanaverage(data, weights=weights, **kwargs)
    assert xp.allclose(r, v_ref)

    # NOTE: we used the following numpy code to compute the reference values!
    # when weight is None:
    #    v_ref = np.nanmean(data, axis=-1)
    # when weight is not None:
    #   v_ref = np.average(data, axis=-1, weights=weights)
    #   v_ref[:, 1] = np.nansum(data, axis=-1)[:, 1]

    # v_ref = np.nanmean(data, axis=-1)

    # v_ref = np.average(data, axis=-1, weights=weights)
    # # replace nan values
    # v_ref[:, 1] = np.nansum(data, axis=-1)[:, 1]
