# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""
Extreme index functions operating on numpy arrays.
"""

from .cpf import cpf  # noqa
from .efi import efi  # noqa
from .sot import sot  # noqa
from .sot import sot_func  # noqa
from .sot import sot_unsorted  # noqa

__all__ = [
    "cpf",
    "efi",
    "sot",
    "sot_unsorted",
    "sot_func",
]
