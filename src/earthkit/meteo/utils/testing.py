# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

# A collection of functions to support pytest testing

import os

from earthkit.utils.testing import get_array_backend

ARRAY_BACKENDS = get_array_backend(["numpy", "torch", "cupy"], raise_on_missing=False)
NUMPY_BACKEND = get_array_backend("numpy")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if not os.path.exists(os.path.join(ROOT_DIR, "tests")):
    ROOT_DIR = "./"


def test_data_path(filename: str) -> str:
    return os.path.join(ROOT_DIR, filename)
