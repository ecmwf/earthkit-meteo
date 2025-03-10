# earthkit-meteo

<img src="https://github.com/ecmwf/logos/raw/refs/heads/main/logos/earthkit/earthkit-meteo-light.svg" width="160">

[![PyPI version fury.io](https://badge.fury.io/py/earthkit-meteo.svg)](https://pypi.python.org/pypi/earthkit-meteo/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/earthkit-meteo.svg)](https://pypi.python.org/pypi/earthkit-meteo/)

**DISCLAIMER**
This project is **BETA** and will be **Experimental** for the foreseeable future.
Interfaces and functionality are likely to change, and the project itself may be scrapped.
**DO NOT** use this software in any project/software that is operational.

**earthkit-meteo** is a Python package providing meteorological computations using **numpy** input and output.

```python
from earthkit.meteo import thermo
import numpy as np

t = np.array([264.12, 261.45]) # Kelvins
p = np.array([850, 850]) * 100. # Pascals

theta = thermo.potential_temperature(t, p)
```

## Documentation

The documentation can be found at https://earthkit-meteo.readthedocs.io/.

## Install

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

## License

```
Copyright 2022, European Centre for Medium Range Weather Forecasts.

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
