# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from earthkit.data import Field, FieldList  # type: ignore[import]

from earthkit.meteo.utils.fieldlist import field_pressure_in_pa, get_hybrid_level_parameters
from earthkit.meteo.utils.param import FIELD_PARAMS


class Item:
    def __init__(self, key: str, name: str, fl=None):
        self.key = key
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


class SurfaceItem(Item):
    def __init__(self, key: str, name: str, fl=None):
        super().__init__(key, name, fl)

        if isinstance(self.fl, FieldList):
            if len(self.fl) != 1:
                raise ValueError(f"Expected exactly one {name} field, but found {len(self.fl)}.")
            self.fl = self.fl[0]
        if not isinstance(self.fl, Field):
            raise ValueError(f"{name} must be a Field or a FieldList containing exactly one Field.")


class ProfileItem(Item):
    def __init__(self, key: str, name: str, fl=None, level_type=None, sort_direction="ascending", sort=True):
        self.sort_direction = sort_direction
        super().__init__(key, name, fl)
        self._levels = None
        self.level_type = level_type

        if not isinstance(fl, FieldList):
            raise ValueError(f"{name} must be a FieldList.")
        if len(fl) == 0:
            raise ValueError(f"{name} must contain at least one field.")

        if sort:
            self.fl = self.fl.order_by({"vertical.level": self.sort_direction})
        u = fl.unique(["vertical.level", "vertical.level_type"])

        if self.level_type is not None:
            if u["vertical.level_type"] != (self.level_type,):
                raise ValueError(f"{name} fields must have '{self.level_type}' as their vertical level type.")

        if len(u["vertical.level"]) != len(fl):
            raise ValueError(f"Multiple fields with the same vertical level found in {name} FieldList.")


class MonotonicInputHandler:
    def __init__(self, level_type=None, sort_direction="ascending"):
        self.level_type = level_type
        self.sort_direction = sort_direction
        assert self.sort_direction in (
            "ascending",
            "descending",
        ), "sort_direction must be 'ascending' or 'descending'."

        self.surface = {}
        self.profile = {}
        self.non_field_input = {}

    def add_surface(self, fl, key: str, name: str) -> None:
        assert key, "Key must be provided for surface field."
        assert name, "Name must be provided for surface field."
        if key in self.surface:
            raise ValueError(f"Surface field with key '{key}' already exists.")
        item = SurfaceItem(key, name, fl)
        setattr(self, key, item.fl)
        self.surface[key] = item

    def add_profile(self, fl, key: str, name: str, sort=True) -> None:
        assert key, "Key must be provided for profile field."
        assert name, "Name must be provided for profile field."
        if key in self.profile:
            raise ValueError(f"Profile field with key '{key}' already exists.")
        item = ProfileItem(key, name, fl, level_type=self.level_type, sort_direction=self.sort_direction, sort=sort)
        setattr(self, key, item.fl)
        self.profile[key] = item

    def add_coord(self, fl, key: str, name: str, source=None, level_type=None, sort=True) -> None:
        import numpy as np

        assert key, "Key must be provided for profile field."
        assert name, "Name must be provided for profile field."
        if isinstance(fl, Field):
            fl = FieldList.from_fields([fl])

        if isinstance(fl, FieldList):
            self.add_profile(fl, key, name)
        elif fl is not None:
            if source is not None:
                if len(fl) == len(source):
                    r = []
                    for f, s in zip(fl, source):
                        r.append(Field.from_dict({"values": f, "vertical": s.vertical}))
                    self.add_profile(FieldList.from_fields(r), key, name, sort=sort)
                else:
                    raise ValueError(
                        "Length of provided coordinate array does not match the number of levels "
                        "in the source FieldList."
                    )
            elif level_type is not None:
                if isinstance(fl, (int, float)):
                    fl = [fl]

                r = []
                for f in fl:
                    level = f if isinstance(f, (int, float)) else f[0]
                    values = np.asarray(f)
                    r.append(
                        Field.from_dict({"values": values, "vertical": {"level_type": level_type, "level": level}})
                    )
                self.add_profile(FieldList.from_fields(r), key, name, sort=sort)
                self.non_field_input[key] = self.profile[key]
            else:
                raise ValueError(
                    "When providing a coordinate array that is not a FieldList, either 'source' or "
                    "'level_type' must be provided to infer the vertical metadata."
                )
        else:
            r = []
            level_type = source[0].get("vertical.level_type")
            for s in source:
                level = s.get("vertical.level")
                if level_type == "pressure":
                    level = field_pressure_in_pa(s)
                r.append(Field.from_dict({"values": np.asarray([level]), "vertical": s.vertical}))
            self.add_profile(FieldList.from_fields(r), key, name, sort=sort)

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

    def to_fieldlist(self, arr, template=None, levels=None, vertical=None, metadata=None, param_name=""):
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

    def levels(self):
        for item in self.profile.values():
            return item.levels()
        raise ValueError("No profile fields added to infer levels from.")


class HybridInputHandler(MonotonicInputHandler):
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
        self.add_surface(zs, "zs", "Surface geopotential")

    def add_sp(self, sp):
        self.add_surface(sp, "sp", "Surface pressure")

    def add_t(self, t):
        self.add_profile(t, "t", "Temperature")

    def add_q(self, q):
        self.add_profile(q, "q", "Specific humidity")

    def add_alpha(self, alpha):
        self.add_profile(alpha, "alpha", "Alpha")

    def add_delta(self, delta):
        self.add_profile(delta, "delta", "Delta")

    # def add_profile(self, fl, name):
    #     item = ProfileItem(name, self.NAMES.get(name, name), fl, level_type=self.LEVEL_TYPE)
    #     setattr(self, name, fl)

    # def check_levels(self):
    #     profiles = {"t": self.t, "q": self.q, "alpha": self.alpha, "delta": self.delta}
    #     self.levels = None
    #     for key, fl in profiles.items():
    #         if fl is not None:
    #             if self.levels is None:
    #                 self.levels = fl.get("vertical.level")
    #             else:
    #                 if list(fl.get("vertical.level")) != list(self.levels):
    #                     raise ValueError(
    #                         f"All input FieldLists must have the same vertical "
    #                         f"  levels. Mismatch found in {self.NAMES[key]} fields."
    #                     )

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


class CoordInputHandler(MonotonicInputHandler):
    def __init__(self, data, coord, sort_direction="ascending"):
        if isinstance(coord, FieldList):
            self.coord = coord
        elif coord is not None:
            self.coord = coord
        else:
            self.level_type = data[0].get("vertical.level_type")
            self.levels = data[0].get("vertical.level")

    def add_coord(self, coord):
        self.add_profile(coord, "coord", "Coordinate")

    def add_target_coord(self, target_coord):
        self.add_profile(target_coord, "target_coord", "Target coordinate")

    def data_arr(self):
        return self.data.to_numpy(copy=False)

    def coord_arr(self):
        return self.coord.to_numpy(copy=False)


class TargetCoordItem(Item):
    def __init__(self, fl, key: str, name: str, source_data=None, source_coord=None, level_type=None) -> None:
        import numpy as np

        assert key, "Key must be provided for profile field."
        assert name, "Name must be provided for profile field."

        if isinstance(fl, Field):
            fl = FieldList.from_fields([fl])

        # determine level type
        if level_type is None:
            if isinstance(fl, FieldList):
                level_type = fl[0].get("vertical.level_type")
            elif isinstance(source_coord, FieldList):
                level_type = source_coord[0].get("vertical.level_type")
            elif isinstance(source_data, FieldList):
                level_type = source_data[0].get("vertical.level_type")
            else:
                raise ValueError(
                    "Unable to determine level type for target coordinate. Please provide 'level_type' or "
                    "a source FieldList with vertical metadata."
                )

        if isinstance(fl, FieldList):
            super().__init__(key, name, fl)
        else:
            if isinstance(fl, (int, float)):
                fl = [fl]
            if hasattr(fl, "__len__") and len(fl) > 0:
                r = []
                for f in fl:
                    level = f if isinstance(f, (int, float)) else f[0]
                    values = np.asarray(f)
                    r.append(
                        Field.from_dict({"values": values, "vertical": {"level_type": level_type, "level": level}})
                    )
                super().__init__(key, name, FieldList.from_fields(r))
            else:
                raise ValueError(
                    f"Invalid input for target coordinate. Must be a FieldList or an array-like of "
                    f"numeric values. Got {type(fl)} with length {len(fl) if hasattr(fl, '__len__') else 'N/A'}."
                )


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
