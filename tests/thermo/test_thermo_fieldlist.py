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

import earthkit.meteo.thermo.array as array_thermo
from earthkit.meteo import thermo
from earthkit.meteo.utils.testing import NO_EKD

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})


@pytest.mark.skipif(NO_EKD, reason="earthkit.data is not installed")
def test_fieldlist_thermo_specific_humidity_from_mixing_ratio():
    import earthkit.data as ekd
    # TODO find sample grib file for mixing ratio
    # ds = ekd.from_source("sample", "tuv_pl.grib")

