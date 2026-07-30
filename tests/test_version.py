import earthkit.meteo


def test_version() -> None:
    assert earthkit.meteo.__version__ != "999"
