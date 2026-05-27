# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _read_conf():
    d = dict()

    from earthkit.meteo.utils.paths import earthkit_conf_file

    with open(
        earthkit_conf_file("ifs_levels_conf.json"),
        "r",
        encoding="utf-8",
    ) as f:
        import json

        data = json.load(f)
        d["ifs"] = dict()
        for n_levels, coeffs in data.items():
            d["ifs"][n_levels] = {
                "A": np.array(coeffs["A"]),
                "B": np.array(coeffs["B"]),
            }

    return d


_CONF = _read_conf()


def hybrid_level_parameters(n_levels: int, model: str = "ifs") -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Get the A and B parameters of hybrid levels for a given configuration.

    Parameters
    ----------
    n_levels : int
        Number of (full) hybrid levels. Currently, only ``n_levels`` 91 and 137 are supported.
    model : str
        Model name. Default is "ifs". Currently, only ``model="ifs"`` are supported.

    Returns
    -------
    tuple[NDArray, NDArray]
        A tuple containing the A and B parameters on the hybrid half-levels. See details below. Both are
        1D numpy arrays of length ``n_levels + 1``.


    Notes
    -----
    - The A and B parameters are not unique; in theory there can be multiple definitions for a
      given number of levels and model. :func:`hybrid_level_parameters` is merely a convenience
      method returning the most typical set of coefficients used.

    - The hybrid model levels divide the atmosphere into :math:`NLEV` layers. These layers are defined
      by the pressures at the interfaces between them for :math:`0 \leq k \leq NLEV`, which are
      the half-levels :math:`p_{k+1/2}` (indices increase from the top of the atmosphere towards
      the surface). The half levels are defined by the A and B parameters in such a way that at
      the top of the atmosphere the first half level pressure :math:`p_{+1/2}` is a constant, while
      at the surface :math:`p_{NLEV+1/2}` is the surface pressure.

      The full-level pressure :math:`p_{k}` associated with each model
      level is defined as the middle of the layer for :math:`1 \leq k \leq NLEV`.

      The level definitions can be written as:

      .. math::

        p_{k+1/2} = A_{k+1/2} + p_{s}  B_{k+1/2}

        p_{k} = \frac{1}{2}  (p_{k-1/2} + p_{k+1/2})

      where

        - :math:`p_{s}` is the surface pressure
        - :math:`p_{k+1/2}` is the pressure at the half-levels
        - :math:`p_{k}` is the pressure at the full-levels
        - :math:`A_{k+1/2}` and :math:`B_{k+1/2}` are the A- and B-coefficients defining
          the model levels.

    For more details see [IFS-CY49R1-Dynamics]_ Chapter 2, Section 2.2.1.
    """
    model = model.lower()
    n_levels = str(n_levels)
    if model in _CONF:
        if n_levels in _CONF[model]:
            c = _CONF[model][n_levels]
            return c["A"], c["B"]
        else:
            raise ValueError(f"Hybrid level parameters not available for {n_levels} levels in model '{model}'.")
    else:
        raise ValueError(f"Model '{model}' not recognised for hybrid level parameters.")


def _hybrid_level_parameters_from_fieldlist(*args) -> tuple[NDArray[Any], NDArray[Any]]:
    import earthkit.data as ekd

    for fl in args:
        if isinstance(fl, ekd.Field):
            fl = [fl]
        elif not isinstance(fl, ekd.FieldList):
            continue

        for field in fl:
            if field.get("vertical.level_type") == "hybrid":
                n_coeff = field.get("metadata.NV")
                try:
                    if n_coeff is not None and n_coeff > 2:
                        pv = field.get("metadata.pv")
                        if pv and len(pv) == n_coeff:
                            A = np.array(pv[: n_coeff + 1])
                            B = np.array(pv[n_coeff + 1 :])
                            return A, B
                except Exception:
                    pass
    return None, None
