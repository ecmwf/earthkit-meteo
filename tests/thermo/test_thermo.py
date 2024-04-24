# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np

from earthkit.meteo import thermo


def test_high_level_celsius_to_kelvin():
    t = np.array([-10, 23.6])
    v = thermo.celsius_to_kelvin(t)
    v_ref = np.array([263.16, 296.76])
    np.testing.assert_allclose(v, v_ref)
