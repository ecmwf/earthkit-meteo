# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from .. import array

def specific_humidity_from_mixing_ratio(w):
    r"""Compute the specific humidity from mixing ratio.

    Parameters
    ----------
    w :  Fieldlist
        Mixing ratio (kg/kg)

    Returns
    -------
    Fieldlist
        Specific humidity (kg/kg)


    The result is the specific humidity in kg/kg units. The computation is based on
    the following definition [Wallace2006]_:

    .. math::

        q = \frac {w}{1+w}

    """
    # Map known "mixing ratio" parameter IDs to "specific humidity" parameter IDs
    param_ids = {
        53: 133,      # humidity mixing ratio -> specific humidity
    }

    result = []
    for wi in w:
        q = wi.values / (1.0 + wi.values)

        param_id_w = wi.metadata("paramId", default=None)
        param_id_q = param_ids.get(param_id_w, 133)

        keys = {}
        if param_id_q is not None:
            keys["paramId"] = param_id_q

        md = wi.metadata().override(**keys)
        result.append(wi.clone(values=q, metadata=md))

    return w.from_fields(result)

def relative_humidity_from_dewpoint(t, td):
    r"""Compute the relative humidity from dew point temperature

    Parameters
    ----------
    t : "xarray.DataArray"
        Temperature (K)
    td: "xarray.DataArray"
        Dewpoint (K)


    Returns
    -------
    "xarray.DataArray"
        Relative humidity (%)


    The computation is based on the following formula:

    .. math::

        r = 100 \frac {e_{wsat}(td)}{e_{wsat}(t)}

    where :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water.

    """

def relative_humidity_from_specific_humidity(t, q, p):
    r"""Compute the relative humidity from specific humidity.

    Parameters
    ----------
    t: array-like
        Temperature (K)
    q: array-like
        Specific humidity (kg/kg)
    p: array-like
        Pressure (Pa)

    Returns
    -------
    array-like
        Relative humidity (%)


    The computation is based on the following formula:

    .. math::

        r = 100 \frac {e(q, p)}{e_{msat}(t)}

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure_from_specific_humidity`)
        * :math:`e_{msat}` is the :func:`saturation_vapour_pressure` based on the "mixed" phase

    """
