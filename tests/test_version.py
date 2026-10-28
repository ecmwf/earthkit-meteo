# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import earthkit.meteo


def test_version() -> None:
    assert earthkit.meteo.__version__ != "999"
