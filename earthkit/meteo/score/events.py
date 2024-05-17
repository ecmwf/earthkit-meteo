
import numpy as np
import scipy
from typing import Literal

QUANTILE_MAP = {
    100: "percentile",
    3: "tercile",
    4: "quartile",
    5: "quintile",
    10: "decile",
}
QUANTILE_NAMES = list(QUANTILE_MAP.values())


# MOVE THIS TO A SERIES OF CLASSES
# def event_field(event, field, statistics=None, dim=None):
#     # dim solely indicates which dimension is the ensemble when `field` is an ensemble
#     # TODO: check if this actually works for "abs" when field has coordinates along `dim` dimension?
#     bev = BinaryEvent(**event)
#     if bev.requires_climatology and statistics is None:
#         raise ValueError(
#             "Binary event %s requires statistics/climatology but none is present" % bev
#         )
#     if bev["type"] == "abs":
#         if dim is not None and dim in field.dims:
#             threshold = xarray.full_like(field[{dim: 0}], bev["value"])
#         else:
#             threshold = xarray.full_like(field, bev["value"])
#     elif bev["type"] == "stdev":
#         threshold = bev["value"] * statistics["stdev"]
#     else:
#         threshold = bev.select_quantile_array(statistics["quantile"])
#     if bev["is_anomaly"]:
#         threshold = threshold + statistics["mean"]
#     valid_mask = _common_valid_mask(field, threshold, dim=dim)
#     return bev.operator()(field, threshold), valid_mask


operators = {
    ">": lambda x, y: x > y,
    "<": lambda x, y: x < y,
    ">=": lambda x, y: x >= y,
    "<=": lambda x, y: x <= y,
}


class Event(dict):
    """A binary event. It describes the event's type, value, operator and whether it is
    anomaly-based or not.

    Args:
        operator (str): Threshold operator, one of (">", ">=", "<", "<=").
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
        operator:Literal[">", ">=", "<", "<="],
        value:float,
        type:Literal["abs", "stdev", "percentile", "tercile"],
        is_anomaly:bool,
        **statistics,
    ):
        
        self.greater_event = operator in (">", ">=")

        super().__init__(
            operator=operator, value=value, type=type, is_anomaly=is_anomaly, **statistics
        )

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

    def operator(self):
        return operators[self["operator"]]

    @property
    def requires_climatology(self):
        return self["is_anomaly"] or self["type"] != "abs"

    def select_quantile_array(
        self, clim_ds, quantile=None, quantile_dim="quantile_"
    ):
        if quantile is None:
            quantile = self["value"]
        name_in_clim_ds = self["type"]
        return clim_ds.loc[
            {quantile_dim: "%d:%d" % (quantile, QUANTILE_MAP[name_in_clim_ds])}
        ].drop_vars(quantile_dim)


    def threshold_array(self):
        raise NotImplementedError

    def event_array(self, array):
        """Event occurrence in the array.
        """
        return self.operator()(array, self.threshold_array()).astype(float)

    def _gaussian_threshold(self):
        raise NotImplementedError

    def probability_gaussian(self):
        threshold = self._gaussian_threshold()
        with np.errstate(invalid="ignore"):
            prob = scipy.stats.norm().cdf(threshold)
        if self.greater_event:
            prob = 1.0 - prob
        return prob


class AbsEvent(Event):

    def __init__(self, value: float, operator: Literal[">", ">=", "<", "<="], **statistics):
        super().__init__(value=value, operator=operator, type="abs", is_anomaly=False, **statistics)

    def threshold_array(self):
        return self["value"]

    def _gaussian_threshold(self):
        return (self["value"] - self["mean"]) / self["stdev"]


class AbsAnomalyEvent(AbsEvent):

    def __init__(self, value: float, operator: Literal[">", ">=", "<", "<="], **statistics):
        # NB: the granny Event has still is_anomaly=False!
        super().__init__(value=value, operator=operator, **statistics)

    def threshold_array(self):
        return self["value"] + self["mean"]

    def _gaussian_threshold(self):
        return self["value"] / self["stdev"]


class StdevEvent(Event):

    def __init__(self, value: float, operator: Literal[">", ">=", "<", "<="], **statistics):
        super().__init__(value=value, operator=operator, type="stdev", is_anomaly=False, **statistics)

    def threshold_array(self):
        return self["value"] * self["stdev"]

    def _gaussian_threshold(self):
        return self["value"] *self["stdev"] - self["mean"]


class StdevAnomalyEvent(StdevEvent):

    def __init__(self, value: float, operator: Literal[">", ">=", "<", "<="], **statistics):
        # NB: the granny Event has still is_anomaly=False!
        super().__init__(value=value, operator=operator, **statistics)

    def threshold_array(self):
        return self["value"] * self["stdev"] + self["mean"]

    def _gaussian_threshold(self):
        return self["value"] + np.full_like(self["stdev"],0.)


# TODO
class PercentileEvent(Event):
    # TODO: how about tercile, quartile, decile, etc.

    def __init__(self, value: float, operator: Literal[">", ">=", "<", "<="], **statistics):
        super().__init__(value=value, operator=operator, type="percentile", is_anomaly=False, **statistics)

    def compute(self, array, percentile):
        binary = self.operator()(array, percentile)



###############################################################################################

def brier_score(probability, event_occurrence):
    """Compute local Brier score.

    Args:
        probability (np.array):
        event_occurrence (np.array):

    Returns:

    """
    return ((probability - event_occurrence) ** 2)


def brier_gaussian(event, array):
    """Compute Brier score of a forecast described by the Gaussian PDF.
    """
    event_occurance = event.event_array(array)
    prob_array = event.probability_gaussian()
    return brier_score(event_occurance, prob_array)

# TODO: not yet tested
def contingency_table(forecast_occurrence, observation_occurrence):
    # TODO: implement these two options
    # if method == "members":
    #     nens = len(e)
    #     rnk0 = sum(e)
    # elif method == "probabilities":
    #     nens = nmem
    #     rnk0 = e*nens
    nens, ngp = forecast_occurrence.shape
    rnk0 = np.nansum(forecast_occurrence, axis=1)
    # construct indices (rounding to cater for potential grib packing)
    ind_rnk=np.nanmin(np.nanmax(rnk0.round().astype(int),0),nens)
    where_observed = np.isclose(observation_occurrence[0],1.0)
    # set contingency table
    ct = np.zeros((2 * (nens + 1), ngp), float)
    cocc = ct[slice(0, nens + 1),:]
    cnoc = ct[slice(nens + 1, None),:]
    cocc[ind_rnk[where_observed], where_observed] = 1.0
    cnoc[ind_rnk[~where_observed], ~where_observed] = 1.0
    # TODO: apply joined nan mask of input arrays
    return cocc, cnoc

if __name__=="__main__":
    mean = np.array([3.9,4.,4.1,4.])
    stdev = np.array([0.2,0.3,0.3,0.3])
    event = AbsAnomalyEvent(.2,">", mean=mean, stdev=stdev)
    observations = np.array([3.3,3.8,4.2,4.])
    bsgaus = brier_gaussian(event, observations)
    print(bsgaus)