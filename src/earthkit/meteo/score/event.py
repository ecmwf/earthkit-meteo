from typing import Literal

import xarray as xr

QUANTILE_DIM_NAME = "quantile_"
QUANTILE_MAP = {
    100: "percentile",
    3: "tercile",
    4: "quartile",
    5: "quintile",
    10: "decile",
}
QUANTILE_NAMES = list(QUANTILE_MAP.values())
QUANTILE_QS = {v: k for k, v in QUANTILE_MAP.items()}

operators = {
    ">": lambda x, y: x > y,
    "<": lambda x, y: x < y,
    ">=": lambda x, y: x >= y,
    "<=": lambda x, y: x <= y,
}


def _common_valid_mask(*arrays, dim=None):
    # return a np bool array of occurrences of all values valid across dim
    # return xr.concat(arrays, dim=dim).notnull().all(dim=dim)
    mask = None
    for array in arrays:
        if dim is not None and dim in array.dims:
            nmask = array.notnull().all(dim=dim)
        else:
            nmask = array.notnull()
        if mask is None:
            mask = nmask
        else:
            mask = mask & nmask
    return mask


def event_field(event, field, statistics=None, dim=None):
    # dim solely indicates which dimension is the ensemble when `field` is an ensemble
    # TODO: check if this actually works for "abs" when field has coordinates along `dim` dimension?
    bev = BinaryEvent(**event)
    if bev.requires_climatology and statistics is None:
        raise ValueError("Binary event %s requires statistics/climatology but none is present" % bev)
    if bev["type"] == "abs":
        if dim is not None and dim in field.dims:
            threshold = xr.full_like(field[{dim: 0}], bev["value"])
        else:
            threshold = xr.full_like(field, bev["value"])
    elif bev["type"] == "stdev":
        threshold = bev["value"] * statistics["stdev"]
    else:
        threshold = bev.select_quantile_array(statistics["quantile"])
    if bev["is_anomaly"]:
        threshold = threshold + statistics["mean"]
    valid_mask = _common_valid_mask(field, threshold, dim=dim)
    return bev.operator()(field, threshold), valid_mask


class BinaryEvent(dict):
    """A binary event. It describes the event's type, value, operator and whether it is
    anomaly-based or not.

    Args:
        operator (str): Threshold operator, one of ([">", ">=", "<", "<=").
        value (float, int): Threshold value.
        type (str): Threshold value type, one of ("abs", "stdev", "percentile", "tercile").
        is_anomaly (bool): Whether the threshold value is an anomaly or not.

    Examples:

        * wind speed greater than 10 m/s:
            BinaryEvent(operator=">",value=10,type="abs",is_anomaly=False)
        * temperature anomaly less than -4 K:
            BinaryEvent(operator="<",value=-4,type="abs",is_anomaly=True)
        * geopotential anomaly greater than 1.5*standard deviation:
            BinaryEvent(operator=">",value=1.5,type="stdev",is_anomaly=True)
        * precipitation greater than 90-th percentile:
            BinaryEvent(operator=">",value=90,type="percentile",is_anomaly=False)

    """

    def __init__(
        self,
        operator: Literal[">", ">=", "<", "<="],
        value: float,
        type: Literal["abs", "stdev", "percentile", "tercile"],
        is_anomaly: bool,
    ):
        super().__init__(operator=operator, value=value, type=type, is_anomaly=is_anomaly)

    def __str__(self):
        def ord_str(n):
            """generate and ordinal string from an integer number"""
            dd = {1: "st", 2: "nd", 3: "rd"}
            nn = int(n) % 10
            try:
                return "{:d}{}".format(int(n), dd[nn])
            except KeyError:
                return "{:d}th".format(int(n))

        if self["type"] in QUANTILE_NAMES:
            val_str = ord_str(int(self["value"] + 0.5))
        else:
            val_str = str(self["value"])
        if self["is_anomaly"]:
            name = "anomaly"
        else:
            name = "field"

        name += self["operator"] + val_str
        if self["type"] != "abs":
            name += " " + self["type"]
        return name

    @property
    def signature(self):
        # used for event header key assigned by FieldMetrics
        match self["type"]:
            case "abs":
                suffix = ""
            case "stdev":
                suffix = "*stdev"
            case _:
                suffix = "_" + self["type"]
        return "{}{}{:g}{}".format(
            "anom" if self["is_anomaly"] else "value", self["operator"], self["value"], suffix
        )

    def operator(self):
        return operators[self["operator"]]

    @property
    def requires_climatology(self):
        return self["is_anomaly"] or self["type"] != "abs"

    def select_quantile_array(self, clim_ds, quantile=None, quantile_dim=QUANTILE_DIM_NAME):
        if quantile is None:
            quantile = self["value"]
        name_in_clim_ds = self["type"]
        return clim_ds.loc[{quantile_dim: "%d:%d" % (quantile, QUANTILE_QS[name_in_clim_ds])}].drop_vars(
            quantile_dim
        )

    # TODO: this is Quaver specific, to be deleted after quaver/3.3.3
    @classmethod
    def from_dict(cls, kwargs):
        mapping = {
            "threshold_operator": "operator",
            "threshold_value": "value",
            "threshold_type": "type",
            "anomaly": "is_anomaly",
        }
        ev = {}
        for key, item in kwargs.items():
            try:
                ev[mapping[key]] = item
            except KeyError:
                ev[key] = item
        return cls(**ev)
