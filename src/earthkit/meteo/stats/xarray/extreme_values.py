# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .. import GumbelDistribution as GumbelDistributionBase


class GumbelDistribution(GumbelDistributionBase):

    def __init__(self, mu, sigma, time_dim=None):
        # freq = mu.coords[time_dim]  # TODO
        super().__init__(mu, sigma, freq=None)  # TODO freq

    @classmethod
    def fit(cls, sample, time_dim=None):
        return NotImplementedError
