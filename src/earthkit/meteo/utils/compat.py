# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from .array import array_namespace


def polyval(x, c):
    """Evaluation of a polynomial using Horner's scheme.

    If ``c`` is of length ``n + 1``, this function returns the value

    .. math:: p(x) = c_0 + c_1 * x + ... + c_n * x^n


    Parameters
    ----------
    x: array-like
        The values(s) at which to evaluate the polynomial. Its elements must
        support addition and multiplication with with themselves and with
        the elements of ``c``.
    c: array0like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n].

    Returns
    -------
    values : array-like
        The value(s) of the polynomial at the given point(s).


    Comments
    --------
    Based on the ``numpy.polynomal.polinomial.polyval`` function.
    """
    c0 = c[-1] + x * 0
    for i in range(2, len(c) + 1):
        c0 = c[-i] + c0 * x
    return c0


def sign(x, *args, **kwargs):
    xp = array_namespace(x)
    x = xp.asarray(x)
    r = xp.sign(x, *args, **kwargs)
    if not xp.isnan(xp.sign(xp.asarray(xp.nan))):
        r[xp.isnan(x)] = xp.nan
    return r
