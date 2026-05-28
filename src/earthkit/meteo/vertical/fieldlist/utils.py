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

"""Helper classes for handling inputs to vertical computation functions when using FieldList objects."""


def number_to_field(value, template=None, keep_template_shape=False):
    if isinstance(value, (float, int)):
        import numpy as np

        if isinstance(template, FieldList):
            template = template[0]

        if isinstance(template, Field):
            if keep_template_shape:
                values = np.full(template.shape, value)
                fl = template.set({"values": values})
            else:
                values = np.asarray([value])
                fl = Field.from_dict({"values": values, "vertical": template.vertical})
            return fl
        elif template is None:
            return Field.from_dict({"values": np.asarray([value]), "vertical": {"level_type": "unknown", "level": 0}})
        else:
            raise ValueError("Invalid template. Template must be None, a Field, or a FieldList")
    else:
        raise ValueError(f"Invalid input for single field. Must be a scalar value. Got {type(value)}.")


def array_to_fieldlist(arr, template, keep_template_shape=False):
    if isinstance(arr, (int, float)):
        arr = [arr]
    if hasattr(arr, "__len__") and len(arr) > 0:
        if isinstance(template, Field):
            template = FieldList.from_fields([template])

        if isinstance(template, FieldList):
            if len(template) != len(arr):
                raise ValueError(
                    f"Length of provided array does not match the number of fields in the template FieldList.\n"
                    f"  {len(arr)} values provided, but template has {len(template)} fields."
                )

            r = []
            for f, t_f in zip(arr, template):
                value = f if isinstance(f, (int, float)) else f[0]
                r.append(number_to_field(value, template=t_f, keep_template_shape=keep_template_shape))
            return FieldList.from_fields(r)
        elif template is None:
            r = []
            for f in arr:
                level = f if isinstance(f, (int, float)) else f[0]
                r.append(number_to_field(level, template=None))
            return FieldList.from_fields(r)

    else:
        raise ValueError(
            f"Invalid input for target coordinate. Must be a FieldList or an array-like of "
            f"numeric values. Got {type(arr)} with length {len(arr) if hasattr(arr, '__len__') else 'N/A'}."
        )


def coord_fieldlist_from_template(fl_template=None, keep_template_shape=False):
    if fl_template is None:
        raise ValueError("A template FieldList must be provided for coordinate fields.")
    if not isinstance(fl_template, FieldList):
        raise ValueError("Template must be a FieldList.")

    r = []
    level_type = fl_template[0].get("vertical.level_type")
    for f in fl_template:
        level = f.get("vertical.level")
        if level_type == "pressure":
            level = level * 100.0
        r.append(number_to_field(level, template=f, keep_template_shape=keep_template_shape))
    return FieldList.from_fields(r)


def to_resulting_fieldlist(arr, template=None, levels=None, vertical=None, metadata=None, param_name=""):
    metadata = metadata or {}
    if param_name:
        metadata = {"parameter": FIELD_PARAMS.get(param_name), **metadata}

    if "vertical" in metadata:
        raise ValueError(
            "Vertical metadata should not be provided in 'metadata' argument, but in the 'vertical' argument."
        )

    res = []

    assert template is not None, "A template Field must be provided to convert the array to a FieldList."
    assert levels is not None, "Levels must be provided to convert the array to a FieldList."

    level_type = None
    if vertical is None:
        vertical = template[0].vertical
        level_type = vertical.get("level_type")
    elif isinstance(vertical, dict):
        level_type = vertical.get("level_type")

    if level_type is None:
        raise ValueError(
            "Level type could not be determined from input. Please provide 'vertical' metadata with a 'level_type' key."
        )

    assert vertical is not None, "Vertical information must be provided to convert the array to a FieldList."
    assert len(levels) == len(arr), "The number of levels must match the number of array elements."
    for v, level in zip(arr, levels):
        if level_type == "pressure":
            # level is in Pa in the Field vertical metadata needs hPa
            level = level / 100.0

        if isinstance(vertical, dict):
            vertical_metadata = dict(vertical)
            vertical_metadata["level"] = level
        else:
            vertical_metadata = vertical.set({"level": level})

        res.append(
            template.set({
                "values": v,
                **metadata,
                "vertical": vertical_metadata,
            })
        )
    return FieldList.from_fields(res)


class Variable:
    def __init__(self, key: str = None, name: str = None, fl=None):
        self.key = key
        if name is None:
            name = key
        self.name = name
        self.fl = fl
        self._levels = None

    def levels(self):
        if self._levels is None:
            if self.fl is not None:
                self._levels = self.fl.get("vertical.level")
            else:
                raise ValueError(f"{self.name} does not have an associated FieldList to infer levels from.")
        return self._levels


class SingleVariable(Variable):
    def __init__(self, key: str = None, name: str = None, fl=None):
        super().__init__(key, name, fl)

        if isinstance(self.fl, FieldList):
            if len(self.fl) != 1:
                raise ValueError(f"Expected exactly one {name} field, but found {len(self.fl)}.")
            self.fl = self.fl[0]
        if not isinstance(self.fl, Field):
            raise ValueError(f"{name} must be a Field or a FieldList containing exactly one Field.")

    @classmethod
    def build(cls, key=None, name=None, fl=None, fl_template=None, keep_template_shape=False) -> None:
        assert key, "Key must be provided for single field."

        if isinstance(fl, (float, int)):
            fl = number_to_field(fl, fl_template)

        return SingleVariable(key=key, name=name, fl=fl)


class ProfileVariable(Variable):
    def __init__(
        self,
        key: str = None,
        name: str = None,
        fl=None,
        level_type=None,
        sort="ascending",
        unique_levels=True,
    ):
        super().__init__(key, name, fl)

        self.sort = sort
        self._levels = None
        self.level_type = level_type
        self.unique_levels = unique_levels

        if not isinstance(fl, FieldList):
            raise ValueError(f"{name} must be a FieldList.")
        if len(fl) == 0:
            raise ValueError(f"{name} must contain at least one field.")

        if isinstance(self.sort, str):
            self.fl = self.fl.order_by({"vertical.level": self.sort})

        if self.unique_levels or self.level_type is not None:
            u = fl.unique(["vertical.level", "vertical.level_type"])

            if self.level_type is not None:
                if u["vertical.level_type"] != (self.level_type,):
                    raise ValueError(f"{name} fields must have '{self.level_type}' as their vertical level type.")

            if self.unique_levels:
                if len(u["vertical.level"]) != len(fl):
                    raise ValueError(f"Multiple fields with the same vertical level found in {name} FieldList. ")

    @classmethod
    def build(
        cls,
        key=None,
        name=None,
        fl=None,
        fl_template=None,
        keep_template_shape=False,
        sort="ascending",
        unique_levels=True,
    ) -> None:
        assert key, "Key must be provided for profile field."

        if isinstance(fl, Field):
            fl = FieldList.from_fields([fl])

        if not isinstance(fl, FieldList):
            fl = array_to_fieldlist(fl, template=fl_template, keep_template_shape=keep_template_shape)

        if not isinstance(fl, FieldList):
            raise ValueError(f"Profile field '{key}' must be a FieldList or convertible to a FieldList.")

        return ProfileVariable(key=key, name=name, fl=fl, sort=sort, unique_levels=unique_levels)

    def first_field_values(self):
        r = []
        for f in self.fl:
            values = f.to_numpy(copy=False, flatten=True)[0]
            r.append(values)
        return r


class CoordinateVariable(ProfileVariable):
    @classmethod
    def build(
        cls,
        key=None,
        name=None,
        fl=None,
        fl_template=None,
        keep_template_shape=False,
        sort="ascending",
        unique_levels=True,
    ) -> None:
        assert key, "Key must be provided for coordinate profile field."

        if isinstance(fl, Field):
            fl = FieldList.from_fields([fl])

        if not isinstance(fl, FieldList):
            if fl is not None:
                fl = array_to_fieldlist(fl, template=fl_template, keep_template_shape=keep_template_shape)
            else:
                fl = coord_fieldlist_from_template(fl_template=fl_template, keep_template_shape=keep_template_shape)

        if not isinstance(fl, FieldList):
            raise ValueError(f"Coordinate profile field '{key}' must be a FieldList or convertible to a FieldList.")

        return CoordinateVariable(key=key, name=name, fl=fl, sort=sort, unique_levels=unique_levels)


class TargetVariable(ProfileVariable):
    def __init__(self, **kwargs):
        raise NotImplementedError(
            "Use factory method TargetVariable.build() to create TargetVariable instances, as the input"
            " handling is more complex than for regular ProfileVariables."
        )

    @classmethod
    def build(
        cls,
        key: str = None,
        name: str = None,
        fl=None,
    ):
        return ProfileVariable.build(
            key=key,
            name=name,
            fl=fl,
            fl_template=None,
            keep_template_shape=False,
            sort=False,
            unique_levels=False,
        )


class MonotonicData:
    def __init__(self, level_type=None, sort="ascending"):
        self.level_type = level_type
        self.sort = sort
        assert self.sort in (
            None,
            "ascending",
            "descending",
        ), "sort must be None, 'ascending' or 'descending'."

        self.single = {}
        self.profile = {}

    def add_single(self, key=None, name=None, fl=None, fl_template=None, keep_template_shape=False) -> None:
        assert key, "Key must be provided for single field."
        if key in self.single:
            raise ValueError(f"Single field with key '{key}' already exists.")

        item = SingleVariable.build(
            key=key, name=name, fl=fl, fl_template=fl_template, keep_template_shape=keep_template_shape
        )
        setattr(self, key, item.fl)
        self.single[key] = item

    def add_profile(
        self,
        key=None,
        name=None,
        fl=None,
        fl_template=None,
        keep_template_shape=False,
        coord=False,
        unique_levels=True,
    ) -> None:
        assert key, "Key must be provided for profile field."
        if key in self.profile:
            raise ValueError(f"Profile field with key '{key}' already exists.")

        if coord:
            item = CoordinateVariable.build(
                key=key,
                name=name,
                fl=fl,
                fl_template=fl_template,
                keep_template_shape=keep_template_shape,
                sort=self.sort,
                unique_levels=unique_levels,
            )
        else:
            item = ProfileVariable.build(
                key=key,
                name=name,
                fl=fl,
                fl_template=fl_template,
                keep_template_shape=keep_template_shape,
                sort=self.sort,
                unique_levels=unique_levels,
            )
        setattr(self, key, item.fl)
        self.profile[key] = item

    def check_levels(self):
        ref = None
        for item in self.profile.values():
            if ref is None:
                ref = item
            else:
                fl = item.fl
                if fl is not None:
                    if list(ref.levels()) != list(item.levels()):
                        raise ValueError(
                            f"All FieldList profiles must have the same vertical "
                            f"  levels. Mismatch found in '{item.name}' compared to '{ref.name}'. "
                            f" Levels in '{item.name}': {list(item.levels())}, "
                            f" levels in '{ref.name}': {list(ref.levels())}"
                        )

    def levels(self):
        for item in self.profile.values():
            return item.levels()
        raise ValueError("No profile fields added to infer levels from.")

    def to_fieldlist(self, arr, template=None, levels=None, vertical=None, metadata=None, param_name=""):
        return to_resulting_fieldlist(
            arr=arr,
            template=template,
            levels=levels,
            vertical=vertical,
            metadata=metadata,
            param_name=param_name,
        )


class HybridData(MonotonicData):
    NAMES = {
        "t": "Temperature",
        "q": "Specific humidity",
        "alpha": "Alpha",
        "delta": "Delta",
        "sp": "Surface pressure",
        "zs": "Surface geopotential",
    }
    LEVEL_TYPE = "hybrid"

    def __init__(self):
        super().__init__(level_type=self.LEVEL_TYPE)
        self.sp = None
        self.A = None
        self.B = None
        self.zs = None
        self.t = None
        self.q = None
        self.alpha = None
        self.delta = None

    def add_zs(self, zs):
        self.add_single(key="zs", name="Surface geopotential", fl=zs)

    def add_sp(self, sp):
        self.add_single(key="sp", name="Surface pressure", fl=sp)

    def add_t(self, t):
        self.add_profile(key="t", name="Temperature", fl=t)

    def add_q(self, q):
        self.add_profile(key="q", name="Specific humidity", fl=q)

    def add_alpha(self, alpha):
        self.add_profile(key="alpha", name="Alpha", fl=alpha)

    def add_delta(self, delta):
        self.add_profile(key="delta", name="Delta", fl=delta)

    def generate_AB(self, A, B):
        self.A, self.B = get_hybrid_level_parameters(self.sp, A=A, B=B)
        return self.A, self.B

    def to_fieldlist(self, arr, template=None, levels=None, vertical=None, metadata=None, param_name=None):
        metadata = metadata or {}

        if vertical is None:
            vertical = {"level_type": self.LEVEL_TYPE}
        if levels is None:
            levels = self.levels()

        return super().to_fieldlist(
            arr,
            template=template,
            levels=levels,
            vertical=vertical,
            metadata=metadata,
            param_name=param_name,
        )
