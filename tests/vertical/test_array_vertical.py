# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import pytest
from _vertical_cases import *  # noqa: F401,F403

import earthkit.meteo.vertical.array as vertical_array


@pytest.fixture
def vertical():
    return vertical_array
