# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from functools import partial

import array_api_compat


class ArrayNamespace:
    pass


class NumpyNamespace(ArrayNamespace):
    def match(self, xp):
        return array_api_compat.is_numpy_namespace(xp)

    def __call__(self):
        """Return the patched version of the array-api-compat numpy namespace."""
        import earthkit.meteo.utils.namespace.numpy as xp

        return xp


class TorchNamespace(ArrayNamespace):
    def match(self, xp):
        return array_api_compat.is_torch_namespace(xp)

    def __call__(self):
        """Return the patched version of the array-api-compat numpy namespace."""
        import earthkit.meteo.utils.namespace.torch as xp

        return xp


class CupyNamespace(ArrayNamespace):
    def match(self, xp):
        return array_api_compat.is_cupy_namespace(xp)

    def __call__(self):
        """Return the patched version of the array-api-compat numpy namespace."""
        import earthkit.meteo.utils.namespace.cupy as xp

        return xp


NAMESPACES = [NumpyNamespace(), TorchNamespace(), CupyNamespace()]
DEFAULT_NAMESPACE = NAMESPACES[0]


# TODO: maybe this is not necessary
def other_namespace(xp):
    """Return the patched version of an array-api-compat namespace."""
    if not hasattr(xp, "histogram2d"):
        from .compute import histogram2d

        xp.histogram2d = partial(histogram2d, xp)
    if not hasattr(xp, "polyval"):
        from .compute import polyval

        xp.polyval = partial(polyval, xp)
    if not hasattr(xp, "percentile"):
        from .compute import percentile

        xp.percentile = partial(percentile, xp)

    if not hasattr(xp, "seterr"):
        from .compute import seterr

        xp.seterr = partial(seterr, xp)

    return xp


def array_namespace(*args):
    """Return the array namespace of the arguments.

    Parameters
    ----------
    *args: tuple
        Scalar or array-like arguments.

    Returns
    -------
    xp: module
        The array-api-compat namespace of the arguments. The namespace
        returned from array_api_compat.array_namespace(*args) is patched with
        extra/modified methods. When only a scalar is passed, the numpy namespace
        is returned.

    Notes
    -----
    The array namespace is extended with the following methods when necessary:
        - polyval: evaluate a polynomial (available in numpy)
        - percentile: compute the nth percentile of the data along the
          specified axis (available in numpy)
        - histogram2d: compute a 2D histogram (available in numpy)
    Some other methods may be reimplemented for a given namespace to ensure correct
    behaviour. E.g. sign() for torch.
    """
    arrays = [a for a in args if hasattr(a, "shape")]
    if not arrays:
        return DEFAULT_NAMESPACE
    else:
        xp = array_api_compat.array_namespace(*arrays)
        for ns in NAMESPACES:
            if ns.match(xp):
                return ns()

        return other_namespace(xp)
