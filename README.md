<p align="center">
  <picture>
    <source srcset="https://github.com/ecmwf/logos/raw/refs/heads/main/logos/earthkit/earthkit-meteo-dark.svg" media="(prefers-color-scheme: dark)">
    <img src="https://github.com/ecmwf/logos/raw/refs/heads/main/logos/earthkit/earthkit-meteo-light.svg" height="120">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE/foundation_badge.svg" alt="ECMWF Software EnginE">
  </a>
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity/emerging_badge.svg" alt="Maturity Level">
  </a>
  <!-- <a href="https://codecov.io/gh/ecmwf/earthkit-hydro">
    <img src="https://codecov.io/gh/ecmwf/earthkit-hydro/branch/develop/graph/badge.svg" alt="Code Coverage">
  </a> -->
  <a href="https://opensource.org/licenses/apache-2-0">
    <img src="https://img.shields.io/badge/Licence-Apache 2.0-blue.svg" alt="Licence">
  </a>
  <a href="https://github.com/ecmwf/earthkit-meteo/releases">
    <img src="https://img.shields.io/github/v/release/ecmwf/earthkit-meteo?color=purple&label=Release" alt="Latest Release">
  </a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a>
  •
  <a href="#installation">Installation</a>
  •
  <a href="https://earthkit-meteo.readthedocs.io/en/latest/">Documentation</a>
</p>

> \[!IMPORTANT\]
> This software is **Emerging** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

**earthkit-meteo** is a Python package providing meteorological computations using **numpy** input and output. It is a component of [earthkit](https://github.com/ecmwf/earthkit).

## Quick Start

```python
from earthkit.meteo import thermo
import numpy as np

t = np.array([264.12, 261.45]) # Kelvins
p = np.array([850, 850]) * 100. # Pascals

theta = thermo.potential_temperature(t, p)
```

## Installation

Install via `pip` with:

```
$ pip install earthkit-meteo
```

More details, such as how to install any necessary binaries, can be found  at https://earthkit-meteo.readthedocs.io/en/latest/install.html.

Alternatively, install via `conda` with:

```
$ conda install earthkit-meteo -c conda-forge
```

This will bring in some necessary binary dependencies for you.

## Licence

```
Copyright 2023, European Centre for Medium Range Weather Forecasts.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

In applying this licence, ECMWF does not waive the privileges and immunities
granted to it by virtue of its status as an intergovernmental organisation
nor does it submit to any jurisdiction.
```
