# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from importlib import import_module

import numpy as np


def modules_installed(*modules):
    for module in modules:
        try:
            import_module(module)
        except ImportError:
            return False
    return True


NO_PYTORCH = not modules_installed("torch")
NO_CUPY = not modules_installed("cupy")
if not NO_CUPY:
    try:
        import cupy as cp

        a = cp.ones(2)
    except Exception:
        NO_CUPY = True


def is_scalar(data):
    return isinstance(data, (int, float)) or data is not data


class ArrayBackend:
    ns = None

    def asarray(self, *data, **kwargs):
        res = [self.ns.asarray(d, **kwargs) for d in data]
        r = res if len(res) > 1 else res[0]
        return r

    def allclose(self, *args, **kwargs):
        if is_scalar(args[0]):
            v = [self.asarray(a, dtype=self.dtype) for a in args]
        else:
            v = args
        return self.ns.allclose(*v, **kwargs)


class NumpyBackend(ArrayBackend):
    def __init__(self):
        self.ns = np
        self.dtype = np.float64


class PytorchBackend(ArrayBackend):
    def __init__(self):
        import torch

        self.ns = torch
        self.dtype = torch.float64


class CupyBackend(ArrayBackend):
    def __init__(self):
        import cupy

        self.ns = cupy
        self.dtype = cupy.float64


ARRAY_BACKENDS = [NumpyBackend()]
if not NO_PYTORCH:
    ARRAY_BACKENDS.append(PytorchBackend())

if not NO_CUPY:
    ARRAY_BACKENDS.append(CupyBackend())
