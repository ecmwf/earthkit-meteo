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
from importlib import import_module

from earthkit.meteo.utils.download import simple_download

_REMOTE_ROOT_URL = "https://sites.ecmwf.int/repository/earthkit-meteo/"
_REMOTE_TEST_DATA_URL = "https://sites.ecmwf.int/repository/earthkit-meteo/test-data/"
_REMOTE_EXAMPLES_URL = "https://sites.ecmwf.int/repository/earthkit-meteo/examples/"

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
if not os.path.exists(os.path.join(_ROOT_DIR, "tests")):
    _ROOT_DIR = "./"


def earthkit_path(*args) -> str:
    return os.path.join(_ROOT_DIR, *args)


def earthkit_test_data_path(name):
    return earthkit_path("tests", "data", name)


def earthkit_remote_path(*args):
    return os.path.join(_REMOTE_ROOT_URL, *args)


def earthkit_remote_test_data_path(*args):
    return os.path.join(_REMOTE_TEST_DATA_URL, *args)


def earthkit_remote_examples_path(*args):
    return os.path.join(_REMOTE_EXAMPLES_URL, *args)


def get_test_data(filename, subfolder):
    if not isinstance(filename, list):
        filename = [filename]

    res = []
    for fn in filename:
        d_path = earthkit_test_data_path(subfolder)
        os.makedirs(d_path, exist_ok=True)
        f_path = os.path.join(d_path, fn)
        if not os.path.exists(f_path):
            simple_download(url=f"{_REMOTE_ROOT_URL}/{subfolder}/{fn}", target=f_path)
        res.append(f_path)

    if len(res) == 1:
        return res[0]
    else:
        return res


def read_data_file(path):
    import numpy as np

    d = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
    )
    return d


def read_test_data_file(path):
    return read_data_file(earthkit_test_data_path(path))


def save_test_data_reference(file_name, data):
    """Helper function to save test reference data into csv"""
    import numpy as np

    np.savetxt(
        earthkit_test_data_path(file_name),
        np.column_stack(tuple(data.values())),
        delimiter=",",
        header=",".join(list(data.keys())),
    )


def modules_installed(*modules) -> bool:
    for module in modules:
        try:
            import_module(module)
        except (ImportError, RuntimeError, SyntaxError):
            return False
    return True


NO_XARRAY = not modules_installed("xarray")
