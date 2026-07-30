"""
Extreme index functions operating on xarray objects.
"""

from .cpf import cpf  # noqa
from .efi import efi  # noqa
from .sot import sot  # noqa
from .sot import sot_unsorted  # noqa

__all__ = [
    "cpf",
    "efi",
    "sot",
    "sot_unsorted",
]
