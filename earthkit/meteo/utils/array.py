# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import threading


class ArrayNamespace:
    def __init__(self):
        self._api = False
        self.lock = threading.Lock()

    @property
    def api(self):
        if self._api is False:
            with self.lock:
                if self._api is False:
                    try:
                        import array_api_compat

                        self._api = array_api_compat
                    except Exception:
                        self._api = None
        return self._api

    def namespace(self, *arrays):
        if not arrays:
            import numpy as np

            return np

        if self.api is not None:
            return self.api.array_namespace(*arrays)

        import numpy as np

        if isinstance(arrays[0], np.ndarray):
            return np
        else:
            raise ValueError(
                "Can't find namespace for array. Please install array_api_compat package"
            )


def array_namespace(*args):
    arrays = [a for a in args if hasattr(a, "shape")]
    return _NAMESPACE.namespace(arrays)


_NAMESPACE = ArrayNamespace()
