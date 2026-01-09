# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

_SAMPLES = {"vertical_hybrid_data": "json"}


class SampleInputData:
    """Helper class to load sample input data."""

    def __init__(self, d, xp=None, device=None):

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


def download_example_file(file_names, remote_dir="examples", force=False):
    import os
    import urllib.request

    from earthkit.meteo.utils.testing import earthkit_remote_path

    if isinstance(file_names, str):
        file_names = [file_names]

    for f_name in file_names:
        if force or not os.path.exists(f_name):
            urllib.request.urlretrieve(earthkit_remote_path(os.path.join(remote_dir, f_name)), f_name)


def remote_example_file(file_name, remote_dir="examples"):
    import os

    from earthkit.meteo.utils.testing import earthkit_remote_path

    return earthkit_remote_path(os.path.join(remote_dir, file_name))


def get_sample(name):
    import os

    from earthkit.meteo.utils.download import simple_download
    from earthkit.meteo.utils.testing import earthkit_remote_path

    if name in _SAMPLES:
        file_name = f"{name}.{_SAMPLES[name]}"
    else:
        file_name = name

    url = earthkit_remote_path(os.path.join("samples", file_name))
    target = f"_{file_name}"

    if not os.path.exists(target):
        simple_download(url=url, target=target, xp=None)

    d = SampleInputData.from_json(target)

    return d
