# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Additional methods made available in an the array-api-compat namespace.
"""


def polyval(xp, x, c):
    """Evaluation of a polynomial using Horner's scheme.

    If ``c`` is of length ``n + 1``, this function returns the value

    .. math:: p(x) = c_0 + c_1 * x + ... + c_n * x^n


    Parameters
    ----------
    xp: array namespace
        The array namespace to use.
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
    Based on the ``numpy.polynomal.polynomial.polyval`` function.
    """
    c0 = c[-1] + x * 0
    for i in range(2, len(c) + 1):
        c0 = c[-i] + c0 * x
    return c0


def percentile(xp, a, q, **kwargs):
    """Compute percentiles by calling the quantile function.

    Parameters
    ----------
    xp: array namespace
        The array namespace to use.
    """
    return xp.quantile(a, q / 100, **kwargs)


def histogram2d(xp, x, y, *args, **kwargs):
    """Compute a 2D histogram.

    Parameters
    ----------
    xp: array namespace
        The array namespace to use.
    x: array-like
        An array containing the x coordinates of the points to be histogrammed.
    y: array-like
        An array containing the y coordinates of the points to be histogrammed.
    """
    return xp.histogramdd(xp.stack([x, y]).T, *args, **kwargs)


def seterr(xp, *args, **kwargs):
    """Set how floating-point errors are handled.

    Just a placeholder for the numpy function.

    Parameters
    ----------
    xp: array namespace
        The array namespace to use.
    """
    return dict()
