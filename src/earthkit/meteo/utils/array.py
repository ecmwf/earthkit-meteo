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


def numpy_namespace():
    """Return the patched version of the array-api-compat numpy namespace."""
    import earthkit.meteo.utils.namespace.numpy as xp

    return xp


def torch_namespace():
    """Return the patched version of the array-api-compat torch namespace."""
    import earthkit.meteo.utils.namespace.torch as xp

    return xp


def other_namespace(xp):
    """Return the patched version of an array-api-compat namespace."""
    if not hasattr(xp, "polyval"):
        from .compute import polyval

        xp.polyval = partial(polyval, xp)
    if not hasattr(xp, "percentile"):
        from .compute import percentile

        xp.percentile = partial(percentile, xp)

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
    Some other methods may be reimplemented for a given namespace to ensure correct
    behaviour. E.g. sign() for torch.
    """
    arrays = [a for a in args if hasattr(a, "shape")]
    if not arrays:
        return numpy_namespace()
    else:
        xp = array_api_compat.array_namespace(*arrays)
        if array_api_compat.is_numpy_namespace(xp):
            return numpy_namespace()
        elif array_api_compat.is_torch_namespace(xp):
            return torch_namespace()
        else:
            return other_namespace(xp)
