Weather Regimes
###############

The concepts behind :py:mod:`earthkit.meteo.regimes`.


Computing a Regime Index
========================

The computation of a regime index by projection onto a regime pattern follows the approach of `Michel and Rivière (2011) <https://doi.org/10.1175/2011JAS3635.1>`_.


Step 1: Compute Anomaly
-----------------------

Regime patterns are typically defined as patterns of anomalies, e.g., anomalies of 500 hPa geopotential height or surface pressure.
The field from which a regime index is computed by projection must therefore also contain **anomalies**:

.. math::

    X' = X - \overline{X},

where :math:`\overline{X}` is a climatological reference field for field :math:`X` and the subtraction operation is carried out gridpoint-wise.

The climatological reference field is not necessarily a constant.
Its values can vary in time to accomodate seasonality.
A gridpoint-wise temporal low-pass filter is often additionally applied to the anomaly fields after computation and before projection.

.. seealso::

    :py:func:`earthkit.transforms.climatology.anomaly`


Step 2: Project onto Pattern
----------------------------

The **projection** of :math:`X'` onto a regime pattern :math:`R` is a weighted sum of their gridpoint-wise product over the domain :math:`D` of the regime pattern:

.. math::

    P_r = \frac{ \sum_{i \in D} X'_i \cdot R_i \cdot w_i }{ \sum_{i \in D} w_i },

where :math:`w` is a field of weights.
For fields given on a regular latitude-longitude grid, the weights are typically defined as :math:`w_i = cos(\phi_i)`, where :math:`\phi` is latitude.

The regime pattern :math:`R` for a regime :math:`r` is not necessarily constant.
Patterns may vary in time, e.g., to account for changes in anomaly amplitude over the coarse of the year.
Regime patterns are therefore implemented as pattern *generators* in earthkit-meteo.

.. seealso::

    :py:func:`earthkit.meteo.regimes.project`


Step 3: Standardise Projection
------------------------------

The standardised regime index for a regime :math:`r` is

.. math::

    I_r = \frac{ P_r - \overline{\mu_r} }{ \sigma_r },

where :math:`\mu_r` is the mean and :math:`\sigma_r` the standard deviation of the regime index over a reference period.

.. seealso::

    :py:func:`earthkit.meteo.regimes.regime_index`
