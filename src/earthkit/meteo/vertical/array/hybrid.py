# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

_IFS = {
    137: (
        np.array(
            [
                0.00000000e00,
                2.00036502e00,
                3.10224104e00,
                4.66608381e00,
                6.82797718e00,
                9.74696636e00,
                1.36054239e01,
                1.86089306e01,
                2.49857178e01,
                3.29857101e01,
                4.28792419e01,
                5.49554634e01,
                6.95205765e01,
                8.68958817e01,
                1.07415741e02,
                1.31425507e02,
                1.59279404e02,
                1.91338562e02,
                2.27968948e02,
                2.69539581e02,
                3.16420746e02,
                3.68982361e02,
                4.27592499e02,
                4.92616028e02,
                5.64413452e02,
                6.43339905e02,
                7.29744141e02,
                8.23967834e02,
                9.26344910e02,
                1.03720117e03,
                1.15685364e03,
                1.28561035e03,
                1.42377014e03,
                1.57162292e03,
                1.72944897e03,
                1.89751929e03,
                2.07609595e03,
                2.26543164e03,
                2.46577051e03,
                2.67734814e03,
                2.90039136e03,
                3.13511938e03,
                3.38174365e03,
                3.64046826e03,
                3.91149048e03,
                4.19493066e03,
                4.49081738e03,
                4.79914941e03,
                5.11989502e03,
                5.45299072e03,
                5.79834473e03,
                6.15607422e03,
                6.52694678e03,
                6.91187061e03,
                7.31186914e03,
                7.72741211e03,
                8.15935400e03,
                8.60852539e03,
                9.07640039e03,
                9.56268262e03,
                1.00659785e04,
                1.05846318e04,
                1.11166621e04,
                1.16600674e04,
                1.22115479e04,
                1.27668730e04,
                1.33246689e04,
                1.38813311e04,
                1.44321396e04,
                1.49756152e04,
                1.55082568e04,
                1.60261152e04,
                1.65273223e04,
                1.70087891e04,
                1.74676133e04,
                1.79016211e04,
                1.83084336e04,
                1.86857188e04,
                1.90312891e04,
                1.93435117e04,
                1.96200430e04,
                1.98593906e04,
                2.00599316e04,
                2.02196641e04,
                2.03378633e04,
                2.04123086e04,
                2.04420781e04,
                2.04257188e04,
                2.03618164e04,
                2.02495117e04,
                2.00870859e04,
                1.98740254e04,
                1.96085723e04,
                1.92902266e04,
                1.89174609e04,
                1.84897070e04,
                1.80069258e04,
                1.74718398e04,
                1.68886875e04,
                1.62620469e04,
                1.55966953e04,
                1.48984531e04,
                1.41733242e04,
                1.34277695e04,
                1.26682578e04,
                1.19013398e04,
                1.11333047e04,
                1.03701758e04,
                9.61751562e03,
                8.88045312e03,
                8.16337500e03,
                7.47034375e03,
                6.80442188e03,
                6.16853125e03,
                5.56438281e03,
                4.99379688e03,
                4.45737500e03,
                3.95596094e03,
                3.48923438e03,
                3.05726562e03,
                2.65914062e03,
                2.29424219e03,
                1.96150000e03,
                1.65947656e03,
                1.38754688e03,
                1.14325000e03,
                9.26507812e02,
                7.34992188e02,
                5.68062500e02,
                4.24414062e02,
                3.02476562e02,
                2.02484375e02,
                1.22101562e02,
                6.27812500e01,
                2.28359375e01,
                3.75781298e00,
                0.00000000e00,
                0.00000000e00,
            ]
        ),
        np.array(
            [
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                3.81999996e-08,
                6.76070022e-06,
                2.43480008e-05,
                5.89219999e-05,
                1.11914298e-04,
                1.98577400e-04,
                3.40379687e-04,
                5.61555324e-04,
                8.89697927e-04,
                1.35280553e-03,
                1.99183798e-03,
                2.85712420e-03,
                3.97095364e-03,
                5.37781464e-03,
                7.13337678e-03,
                9.26146004e-03,
                1.18060224e-02,
                1.48156285e-02,
                1.83184519e-02,
                2.23548450e-02,
                2.69635208e-02,
                3.21760960e-02,
                3.80263999e-02,
                4.45479602e-02,
                5.17730154e-02,
                5.97284138e-02,
                6.84482530e-02,
                7.79583082e-02,
                8.82857367e-02,
                9.94616672e-02,
                1.11504652e-01,
                1.24448128e-01,
                1.38312891e-01,
                1.53125033e-01,
                1.68910414e-01,
                1.85689449e-01,
                2.03491211e-01,
                2.22332865e-01,
                2.42244005e-01,
                2.63241887e-01,
                2.85354018e-01,
                3.08598459e-01,
                3.32939088e-01,
                3.58254194e-01,
                3.84363323e-01,
                4.11124766e-01,
                4.38391209e-01,
                4.66003299e-01,
                4.93800312e-01,
                5.21619201e-01,
                5.49301147e-01,
                5.76692164e-01,
                6.03648067e-01,
                6.30035818e-01,
                6.55735970e-01,
                6.80643022e-01,
                7.04668999e-01,
                7.27738738e-01,
                7.49796569e-01,
                7.70797551e-01,
                7.90716767e-01,
                8.09536040e-01,
                8.27256083e-01,
                8.43881130e-01,
                8.59431803e-01,
                8.73929262e-01,
                8.87407541e-01,
                8.99900496e-01,
                9.11448181e-01,
                9.22095656e-01,
                9.31880772e-01,
                9.40859556e-01,
                9.49064434e-01,
                9.56549525e-01,
                9.63351727e-01,
                9.69513416e-01,
                9.75078404e-01,
                9.80071604e-01,
                9.84541893e-01,
                9.88499522e-01,
                9.91984010e-01,
                9.95002508e-01,
                9.97630119e-01,
                1.00000000e00,
            ]
        ),
    )
}


def hybrid_level_parameters(n_levels: int, model: str = "ifs") -> Tuple[NDArray[Any], NDArray[Any]]:
    r"""Get the A and B parameters of hybrid levels for a given configuration.

    Parameters
    -----------
        n_levels: int
            Number of (full) hybrid levels. Currently, only ``n_levels=137`` and ``model="ifs"`` are supported.
        model : str
            Model name. Default is "ifs". Currently, only ``n_levels=137`` and ``model="ifs"`` are supported.

    Returns
    -------
    NDArray, NDArray
        A tuple containing the A and B parameters on the hybrid half-levels See details below. Both are
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

        p_{k+1/2} = A_{k+1/2} + p_{s}\; B_{k+1/2}

        p_{k} = \frac{1}{2}\; (p_{k-1/2} + p_{k+1/2})

      where

        - :math:`p_{s}` is the surface pressure
        - :math:`p_{k+1/2}` is the pressure at the half-levels
        - :math:`p_{k}` is the pressure at the full-levels
        - :math:`A_{k+1/2}` and :math:`B_{k+1/2}` are the A- and B-coefficients defining
          the model levels.

    For more details see [IFS-CY47R3-Dynamics]_ Chapter 2, Section 2.2.1.
    """
    if model.lower() == "ifs":
        if n_levels in _IFS:
            a_coeffs, b_coeffs = _IFS[n_levels]
            return a_coeffs, b_coeffs
        else:
            raise ValueError(f"IFS hybrid level parameters not available for {n_levels} levels.")
    else:
        raise ValueError(f"Model '{model}' not recognized for hybrid level parameters.")
