import numpy as np
import scipy
from typing import Literal, Union, Unpack

QUANTILE_TYPES = Literal["percentile","tercile"]

operators = {
    ">": lambda x, y: x > y,
    "<": lambda x, y: x < y,
    ">=": lambda x, y: x >= y,
    "<=": lambda x, y: x <= y,
}

# NB: ATM this implementation works as follows:
#       event=SomeEvent(value=val,...,mean=array,stdev=array,quantiles={val:array})
#                                     ^^^^ **statistics ^^^^^^^^^^^^^^^^^^^^^^^^^^
#       bs = brier_gaussian(event,obs_array)
#     an alternative would be:
#       event=SomeEvent(value=val,...)
#       bs = brier_gaussian(event,obs_array,mean=array,stdev=array,quantiles={val:array})
#                                           ^^^^ **statistics ^^^^^^^^^^^^^^^^^^^^^^^^^^



class _Event:
    """Binary event.

    Args:
        operator (str): Threshold operator, one of (">", ">=", "<", "<=").
        value (float, int, str): Threshold value.
        type (str): Threshold value type, one of ("abs", "stdev", QUANTILE_TYPES).
        is_anomaly (bool): Whether the threshold value is an anomaly or not.
        statistics (dict): Auxiliary arrays of statistics:
            * mean (ndarray)
            * stdev (ndarray)
            * quantiles (dict of ndarray)

    Examples:

        * wind speed greater than 10 m/s:
            BinaryEvent(operator=">",value=10,type="abs",is_anomaly=False)
        * temperature anomaly less than -4 K:
            BinaryEvent(operator="<",value=-4,type="abs",is_anomaly=True)
        * geopotential anomaly greater than 1.5*standard deviation:
            BinaryEvent(operator=">",value=1.5,type="stdev",is_anomaly=True)
        * precipitation greater than 90-th percentile:
            BinaryEvent(operator=">",value="90:100",type="quantiles",is_anomaly=False)

    """

    def __init__(
        self,
        operator: Literal[">", ">=", "<", "<="],
        value: Union[float, int, str],
        type: Literal[Literal["abs", "stdev"],QUANTILE_TYPES],   # this is probably illegal
        is_anomaly: bool,
        **statistics: Unpack[Union[np.ndarray, dict]],
    ):
        self._operator = operator
        self._value = value
        self._type = type
        self._is_anomaly = is_anomaly
        self._statistics = statistics

        self.greater_event = operator in (">", ">=")

    def __str__(self):
        # TODO: implement in subclasses
        def ord_str(n):
            """generate an ordinal string from an integer number"""
            dd = {1: "st", 2: "nd", 3: "rd"}
            nn = int(n) % 10
            try:
                return "{:d}{}".format(int(n), dd[nn])
            except KeyError:
                return "{:d}th".format(int(n))

        if self._type in QUANTILE_TYPES:
            val_str = ord_str(int(self._value + 0.5))
        else:
            val_str = str(self._value)
        if self._is_anomaly:
            name = "anomaly"
        else:
            name = "field"
        name += self._operator + val_str
        if self._type != "abs":
            name += " " + self._type
        return name

    def operator(self):
        return operators[self._operator]

    def _threshold_array(self):
        raise NotImplementedError

    def _gaussian_pdf_value(self):
        raise NotImplementedError

    def event_array(self, array):
        """Event occurrence in the array."""
        return self.operator()(array, self._threshold_array()).astype(float)

    def probability_gaussian(self):
        threshold = self._gaussian_pdf_value()
        with np.errstate(invalid="ignore"):
            prob = scipy.stats.norm().cdf(threshold)
        if self.greater_event:
            prob = 1.0 - prob
        return prob


class AbsEvent(_Event):

    def __init__(
        self, value: float, operator: Literal[">", ">=", "<", "<="], **statistics
    ):
        super().__init__(
            value=value, operator=operator, type="abs", is_anomaly=False, **statistics
        )

    def _threshold_array(self):
        return self._value

    def _gaussian_pdf_value(self):
        return (self._value - self._statistics["mean"]) / self._statistics["stdev"]


class AbsAnomalyEvent(AbsEvent):

    def __init__(
        self, value: float, operator: Literal[">", ">=", "<", "<="], **statistics
    ):
        # NB: the granny Event has still is_anomaly=False!
        super().__init__(value=value, operator=operator, **statistics)

    def _threshold_array(self):
        return self._value + self._statistics["mean"]

    def _gaussian_pdf_value(self):
        return self._value / self._statistics["stdev"]


class StdevEvent(_Event):
    # NB: this is a weird case, TODO: check with ML if it makes sense at all

    def __init__(
        self, value: float, operator: Literal[">", ">=", "<", "<="], **statistics
    ):
        super().__init__(
            value=value, operator=operator, type="stdev", is_anomaly=False, **statistics
        )

    def _threshold_array(self):
        return self._value * self._statistics["stdev"]

    def _gaussian_pdf_value(self):
        return self._value * self._statistics["stdev"] - self._statistics["mean"]


class StdevAnomalyEvent(StdevEvent):

    def __init__(
        self, value: float, operator: Literal[">", ">=", "<", "<="], **statistics
    ):
        # NB: the granny Event has still is_anomaly=False!
        super().__init__(value=value, operator=operator, **statistics)

    def _threshold_array(self):
        return self._value * self._statistics["stdev"] + self._statistics["mean"]

    def _gaussian_pdf_value(self):
        return self._value + np.full_like(self._statistics["stdev"], 0.0)


class _QuantileEvent(_Event):

    def __init__(
        self,
        value: int,
        type: QUANTILE_TYPES,
        operator: Literal[">", ">=", "<", "<="],
        **statistics,
    ):
        super().__init__(
            value=value,
            operator=operator,
            type=type,
            is_anomaly=False,
            **statistics,
        )

    def _threshold_array(self):
        return self._statistics["quantiles"][self._value]

    def _gaussian_pdf_value(self):
        return (
            self._statistics["quantiles"][self._value] - self._statistics["mean"]
        ) / self._statistics["stdev"]


class PercentileEvent(_QuantileEvent):

    ncats = 100

    def __init__(
        self, value: int, operator: Literal[">", ">=", "<", "<="], **statistics
    ):
        super().__init__(
            value=value,
            operator=operator,
            type="percentile",
            **statistics,
        )


class TercileEvent(_QuantileEvent):

    ncats = 3

    def __init__(
        self, value: int, operator: Literal[">", ">=", "<", "<="], **statistics
    ):
        super().__init__(
            value=value,
            operator=operator,
            type="tercile",
            **statistics,
        )


###############################################################################################


def brier_score(probability, event_occurrence):
    """Compute local Brier score.

    Args:
        probability (np.array):
        event_occurrence (np.array):

    Returns:

    """
    return (probability - event_occurrence) ** 2


def brier_gaussian(event, array):
    """Compute Brier score of a forecast described by the Gaussian PDF."""
    event_occurance = event.event_array(array)
    prob_array = event.probability_gaussian()
    return brier_score(event_occurance, prob_array)


# TODO: not yet tested
def contingency_table(forecast_occurrence, observation_occurrence):
    # TODO: implement other options
    # if method == "members":
    #     nens = len(e)
    #     rnk0 = sum(e)
    # elif method == "probabilities":
    #     nens = nmem
    #     rnk0 = e*nens
    nens, ngp = forecast_occurrence.shape
    rnk0 = np.nansum(forecast_occurrence, axis=1)
    # construct indices (rounding to cater for potential grib packing)
    ind_rnk = np.nanmin(np.nanmax(rnk0.round().astype(int), 0), nens)
    where_observed = np.isclose(observation_occurrence[0], 1.0)
    # set contingency table
    ct = np.zeros((2 * (nens + 1), ngp), float)
    cocc = ct[slice(0, nens + 1), :]
    cnoc = ct[slice(nens + 1, None), :]
    cocc[ind_rnk[where_observed], where_observed] = 1.0
    cnoc[ind_rnk[~where_observed], ~where_observed] = 1.0
    # TODO: apply joined nan mask of input arrays
    return cocc, cnoc


if __name__ == "__main__":
    mean = np.array([3.9, 4.0, 4.1, 4.0])
    stdev = np.array([0.2, 0.3, 0.3, 0.3])
    perc_60 = np.array([4.0, 4.2, 4.4, 4.8])
    event = AbsAnomalyEvent(0.2, ">", mean=mean, stdev=stdev)
    observations = np.array([3.3, 3.8, 4.2, 4.0])
    bsgaus = brier_gaussian(event, observations)
    print(event, bsgaus)
    event = PercentileEvent(60, ">", mean=mean, stdev=stdev, quantiles={60: perc_60})
    bsgaus = brier_gaussian(event, observations)
    print(event, bsgaus)
