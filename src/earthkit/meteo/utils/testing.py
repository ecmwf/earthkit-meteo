# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

# A collection of functions to support pytest testing

from abc import ABCMeta
from abc import abstractmethod
from importlib import import_module

import numpy as np


def modules_installed(*modules):
    for module in modules:
        try:
            import_module(module)
        except ImportError:
            return False
    return True


def is_scalar(data):
    return isinstance(data, (int, float)) or data is not data


class ArrayBackend(metaclass=ABCMeta):
    name = None
    xp = None

    def asarray(self, *data, **kwargs):
        res = [self.xp.asarray(d, **kwargs) for d in data]
        r = res if len(res) > 1 else res[0]
        return r

    def allclose(self, *args, **kwargs):
        if is_scalar(args[0]):
            v = [self.asarray(a, dtype=self.dtype) for a in args]
        else:
            v = args
        return self.xp.allclose(*v, **kwargs)

    @staticmethod
    @abstractmethod
    def available():
        return True


class NumpyBackend(ArrayBackend):
    name = "numpy"

    def __init__(self):
        self.xp = np
        self.dtype = np.float64

    @staticmethod
    def available():
        return True


class PytorchBackend(ArrayBackend):
    name = "torch"

    def __init__(self):
        import torch

        self.xp = torch
        self.dtype = torch.float64

    @staticmethod
    def available():
        return modules_installed("torch")


class CupyBackend(ArrayBackend):
    name = "cupy"

    def __init__(self):
        import cupy

        self.xp = cupy
        self.dtype = cupy.float64

    @staticmethod
    def available():
        if modules_installed("cupy"):
            try:
                import cupy as cp

                cp.ones(2)
                return True
            except Exception:
                return False

        return False


_ARRAY_BACKENDS = {}
for b in {NumpyBackend, PytorchBackend, CupyBackend}:
    if b.available():
        _ARRAY_BACKENDS[b.name] = b()


ARRAY_BACKENDS = list(_ARRAY_BACKENDS.keys())


def get_array_backend(backend):
    if backend is None:
        backend = "numpy"

    if isinstance(backend, str):
        return _ARRAY_BACKENDS[backend]

    return backend
