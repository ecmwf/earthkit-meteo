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
from functools import cached_property
from importlib import import_module


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

    @abstractmethod
    def _make_sample(self):
        return None

    @cached_property
    def namespace(self):
        """The array-api-compat namespace associated with the backend"""
        from .array import array_namespace

        return array_namespace(self._make_sample())

    def asarray(self, *data, **kwargs):
        # TODO: add support for dtype
        res = [self.namespace.asarray(d, **kwargs) for d in data]
        # if "dtype" not in kwargs:
        #     dtype = res[0].dtype
        #     for i in range(1, len(res)):
        #         res[i] = self.namespace.asarray(res[i], dtype=dtype)

        r = res if len(res) > 1 else res[0]
        return r

    def allclose(self, *args, **kwargs):
        if is_scalar(args[0]):
            dtype = self.float64
            v = [self.asarray(a, dtype=dtype) for a in args]
        else:
            v = args
        return self.namespace.allclose(*v, **kwargs)

    def isclose(self, *args, **kwargs):
        if is_scalar(args[0]):
            dtype = self.float64
            v = [self.asarray(a, dtype=dtype) for a in args]
        else:
            v = args
        return self.namespace.isclose(*v, **kwargs)

    def astype(self, *args, **kwargs):
        return self.namespace.astype(*args, **kwargs)

    @cached_property
    def float64(self):
        return self.dtypes.get("float64")

    @cached_property
    def float32(self):
        return self.dtypes.get("float32")

    @property
    @abstractmethod
    def dtypes(self):
        pass

    @staticmethod
    @abstractmethod
    def available():
        return True


class NumpyBackend(ArrayBackend):
    name = "numpy"

    def _make_sample(self):
        import numpy as np

        return np.ones(2)

    @cached_property
    def dtypes(self):
        import numpy

        return {"float64": numpy.float64, "float32": numpy.float32}

    @staticmethod
    def available():
        return True


class PytorchBackend(ArrayBackend):
    name = "torch"

    def _make_sample(self):
        import torch

        return torch.ones(2)

    @cached_property
    def dtypes(self):
        import torch

        return {"float64": torch.float64, "float32": torch.float32}

    @staticmethod
    def available():
        return modules_installed("torch")


class CupyBackend(ArrayBackend):
    name = "cupy"

    def _make_sample(self):
        import cupy

        return cupy.ones(2)

    @cached_property
    def dtypes(self):
        import cupy as cp

        return {"float64": cp.float64, "float32": cp.float32}

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


class JaxBackend(ArrayBackend):
    name = "jax"

    def _make_sample(self):
        import jax.numpy as jarray

        return jarray.ones(2)

    @staticmethod
    def available():
        return modules_installed("jax")

    @cached_property
    def dtypes(self):
        import jax.numpy as jarray

        return {"float64": jarray.float64, "float32": jarray.float32}


# TODO: add support for jax and cupy

_ARRAY_BACKENDS = {}
for b in {NumpyBackend, PytorchBackend}:
    if b.available():
        _ARRAY_BACKENDS[b.name] = b()


ARRAY_BACKENDS = list(_ARRAY_BACKENDS.values())


def get_array_backend(backend):
    if backend is None:
        backend = "numpy"

    if isinstance(backend, list):
        return [get_array_backend(b) for b in backend]

    if isinstance(backend, str):
        return _ARRAY_BACKENDS[backend]

    return backend
