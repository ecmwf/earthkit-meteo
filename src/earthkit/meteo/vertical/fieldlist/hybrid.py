# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from earthkit.data import Field, FieldList  # type: ignore[import]

from earthkit.meteo.utils.fieldlist import get_hybrid_level_parameters
from earthkit.meteo.utils.param import FIELD_PARAMS


class _HybridInput:
    NAMES = {
        "t": "Temperature",
        "q": "Specific humidity",
        "alpha": "Alpha",
        "delta": "Delta",
        "sp": "Surface pressure",
        "zs": "Surface geopotential",
    }

    def __init__(self):
        self.sp = None
        self.A = None
        self.B = None
        self.zs = None
        self.t = None
        self.q = None
        self.alpha = None
        self.delta = None
        self.levels = None

    def add_zs(self, zs):
        if isinstance(zs, FieldList):
            if len(zs) != 1:
                raise ValueError(f"Expected exactly one surface geopotential field, but found {len(zs)}.")
            zs = zs[0]
        if not isinstance(zs, Field):
            raise ValueError("Surface geopotential must be a Field or a FieldList containing exactly one Field.")
        self.zs = zs

    def add_sp(self, sp):
        if isinstance(sp, FieldList):
            if len(sp) != 1:
                raise ValueError(f"Expected exactly one surface pressure field, but found {len(sp)}.")
            sp = sp[0]
        if not isinstance(sp, Field):
            raise ValueError("Surface pressure must be a Field or a FieldList containing exactly one Field.")
        self.sp = sp

    def add_t(self, t):
        self.t = self.add_profile(t, self.NAMES["t"])

    def add_q(self, q):
        self.q = self.add_profile(q, self.NAMES["q"])

    def add_alpha(self, alpha):
        self.alpha = self.add_profile(alpha, self.NAMES["alpha"])

    def add_delta(self, delta):
        self.delta = self.add_profile(delta, self.NAMES["delta"])

    def add_profile(self, fl, name):
        if not isinstance(fl, FieldList):
            raise ValueError(f"{name} must be a FieldList.")
        if len(fl) == 0:
            raise ValueError(f"{name} must contain at least one field.")

        fl = fl.order_by("vertical.level")
        u = fl.unique(["vertical.level", "vertical.level_type"])
        if u["vertical.level_type"] != ("hybrid",):
            raise ValueError(f"{name} fields must have 'hybrid' as their vertical level type.")

        if len(u["vertical.level"]) != len(fl):
            raise ValueError(f"Multiple fields with the same vertical level found in {name} FieldList.")

        setattr(self, name, fl)

        return fl

    def check_levels(self):
        profiles = {"t": self.t, "q": self.q, "alpha": self.alpha, "delta": self.delta}
        self.levels = None
        for key, fl in profiles.items():
            if fl is not None:
                if self.levels is None:
                    self.levels = fl.get("vertical.level")
                else:
                    if list(fl.get("vertical.level")) != list(self.levels):
                        raise ValueError(
                            f"All input FieldLists must have the same vertical "
                            f"  levels. Mismatch found in {self.NAMES[key]} fields."
                        )

    def generate_AB(self, A, B):
        self.A, self.B = get_hybrid_level_parameters(self.sp, A=A, B=B)
        return self.A, self.B

    def to_fieldlist(self, arr, template=None, metadata=None, param_name=""):
        metadata = metadata or {}
        metadata = {"parameter": FIELD_PARAMS.get(param_name), **metadata}
        res = []
        template = template or (self.t[0] if self.t is not None else self.q[0] if self.q is not None else self.sp)
        for v, level in zip(arr, self.levels):
            res.append(
                template.set({
                    "values": v,
                    **metadata,
                    "vertical.level": level,
                    "vertical.level_type": "hybrid",
                })
            )
        return FieldList.from_fields(res)


class MonotonicInput:
    def __init__(self):
        pass

    def add_surface(self, field):
        if isinstance(field, FieldList):
            if len(field) != 1:
                raise ValueError(f"Expected exactly one {self.name} field, but found {len(field)}.")
            field = field[0]
        if not isinstance(field, Field):
            raise ValueError(f"{self.name} must be a Field or a FieldList containing exactly one Field.")
        self.field = field

    def add_profile(self, fl, name):
        if not isinstance(fl, FieldList):
            raise ValueError(f"{name} must be a FieldList.")
        if len(fl) == 0:
            raise ValueError(f"{name} must contain at least one field.")

        fl = fl.order_by("vertical.level")
        u = fl.unique(["vertical.level", "vertical.level_type"])
        if u["vertical.level_type"] != ("hybrid",):
            raise ValueError(f"{name} fields must have 'hybrid' as their vertical level type.")

        if len(u["vertical.level"]) != len(fl):
            raise ValueError(f"Multiple fields with the same vertical level found in {name} FieldList.")

        setattr(self, name, fl)

        return fl


def to_fieldlist(arr, template=None, levels=None, vertical=None, metadata=None, param_name=""):
    metadata = metadata or {}
    if param_name:
        metadata = {"parameter": FIELD_PARAMS.get(param_name), **metadata}

    res = []
    assert template is not None, "A template Field must be provided to convert the array to a FieldList."
    assert levels is not None, "Levels must be provided to convert the array to a FieldList."
    assert vertical is not None, "Vertical information must be provided to convert the array to a FieldList."
    assert len(levels) == len(arr), "The number of levels must match the number of array elements."
    for v, level in zip(arr, levels):
        vertical_metadata = dict(vertical)
        vertical_metadata["level"] = level
        res.append(
            template.set({
                "values": v,
                **metadata,
                "vertical": vertical_metadata,
            })
        )
    return FieldList.from_fields(res)
