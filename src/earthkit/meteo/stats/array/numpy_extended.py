# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from earthkit.utils.array import array_namespace


def nanaverage(data, weights=None, **kwargs):
    """A merge of the functionality of np.nanmean and np.average.

    Parameters
    ----------
    data: array-like
        Data to average.
    weights: array-like, None
        Weights to apply to the data for averaging.
        Weights will be normalised and must correspond to the
        shape of the numpy data array and axis/axes that is/are
        averaged over.
    axis: axis/axes to compute the nanaverage over.
    kwargs: any other np.nansum kwargs

    Returns
    -------
    array-like
        Mean of data (along axis) where nan-values are ignored
        and weights applied if provided.
    """
    xp = array_namespace(data)
    data = xp.asarray(data)

    if weights is not None:
        weights = xp.asarray(weights)
        # set weights to nan where data is nan:
        this_weights = xp.ones(data.shape) * weights
        this_weights[xp.isnan(data)] = xp.nan
        # Weights must be scaled to the sum of valid
        #  weights for each relevant axis:
        this_denom = xp.nansum(this_weights, **kwargs)
        # If averaging over an axis then we must add dummy
        # dimension[s] to the denominator to make compatible
        # with the weights.
        if kwargs.get("axis", None) is not None:
            reshape = list(this_weights.shape)
            reshape[kwargs.get("axis")] = 1
            this_denom = xp.reshape(this_denom, reshape)

        # Scale weights to mean of valid weights:
        this_weights = this_weights / this_denom
        # Apply weights to data:
        nanaverage = xp.nansum(data * this_weights, **kwargs)
    else:
        # If no weights, then nanmean will suffice
        nanaverage = xp.nanmean(data, **kwargs)

    return nanaverage
