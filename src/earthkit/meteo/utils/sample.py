# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

_CONF = {"vertical_hybrid_data": "json"}


class SampleInputData:
    """Helper class to load sample input data."""

    def __init__(self, d, xp=None, device=None):

        self._d = d
        if xp is None:
            from earthkit.utils.array import array_namespace

            xp = array_namespace("numpy")

        for k, v in d.items():
            if xp is not None:
                v = xp.asarray(v, device=device)
            setattr(self, k, v)

    @classmethod
    def from_json(cls, file_name):
        import json

        with open(file_name, "r") as f:
            data = json.load(f)
        return cls(data)

    def to_dict(self):
        return self._d


# TODO: add thread safety if needed
class Samples:
    _CACHE = {}

    def __contains__(self, name):
        return name in self._CACHE

    def get(self, name):
        if name in self._CACHE:
            return self._CACHE[name]

        import os
        import tempfile

        from earthkit.meteo.utils.download import simple_download
        from earthkit.meteo.utils.testing import earthkit_remote_path

        if name in _CONF:
            file_name = f"{name}.{_CONF[name]}"
        elif not name.endswith(".json"):
            file_name = f"{name}.json"
        else:
            file_name = name

        url = earthkit_remote_path(os.path.join("samples", file_name))

        d = None
        with tempfile.NamedTemporaryFile(delete_on_close=True) as fp:
            fp.close()
            target = fp.name
            simple_download(url=url, target=target)

            d = SampleInputData.from_json(target)
            self._CACHE[name] = d

        return d


SAMPLES = Samples()


def get_sample(name):
    try:
        return SAMPLES.get(name)
    except Exception as e:
        raise Exception(f"Sample data '{name}' not found or could not be loaded.") from e
