# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from earthkit.utils.array import array_namespace


# Replacement for scipy.stats.lmoment from scipy v0.15
def lmoment(sample, order=(1, 2), axis=0):
    """Compute first 2 L-moments of a dataset along the first axis."""
    if len(order) != 2 or order[0] != 1 or order[1] != 2:
        raise NotImplementedError
    if axis != 0:
        raise NotImplementedError

    nmoments = 3

    xp = array_namespace(sample)
    sample = xp.asarray(sample)
    # At least four values needed to make a sample L-moments estimation
    nvalues, *rest_shape = sample.shape
    if nvalues < 4:
        raise ValueError("Insufficient number of values to perform sample L-moments estimation")

    sample = xp.sort(sample, axis=0)  # ascending order

    sums = xp.zeros_like(sample, shape=(nmoments, *rest_shape))

    for i in range(1, nvalues + 1):
        z = i
        term = sample[i - 1]
        sums[0] = sums[0] + term
        for j in range(1, nmoments):
            z -= 1
            term = term * z
            sums[j] = sums[j] + term

    y = float(nvalues)
    z = float(nvalues)
    sums[0] = sums[0] / z
    for j in range(1, nmoments):
        y = y - 1.0
        z = z * y
        sums[j] = sums[j] / z

    k = nmoments
    p0 = -1.0
    for _ in range(2):
        ak = float(k)
        p0 = -p0
        p = p0
        temp = p * sums[0]
        for i in range(1, k):
            ai = i
            p = -p * (ak + ai - 1.0) * (ak - ai) / (ai * ai)
            temp = temp + (p * sums[i])
        sums[k - 1] = temp
        k = k - 1

    return xp.stack([sums[0], sums[1]])
