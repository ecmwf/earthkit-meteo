import os

import pytest

from earthkit.meteo.utils.testing import earthkit_path

SKIP = [
    "conf.py",
    "xml2rst.py",
    "xref.py",
    "skip_api_rules.py",
    "venv",
]

EXAMPLES = earthkit_path("docs")


def example_list():
    examples = []
    for root, _, files in os.walk(EXAMPLES):
        for file in files:
            path = os.path.join(root, file)
            if path.endswith(".py") and file not in SKIP:
                n = len(EXAMPLES) + 1
                examples.append(path[n:])

    return sorted(examples)


# # @pytest.mark.skipif(not IN_GITHUB, reason="Not on GITHUB")
# @pytest.mark.parametrize("path", example_list())
# def test_example(path):
#     full = os.path.join(EXAMPLES, path)
#     with open(full) as f:
#         exec(f.read(), dict(__file__=full), {})


@pytest.mark.parametrize("path", example_list())
def test_example(tmpdir, path):
    print("test_example path=", path)
    full = os.path.join(EXAMPLES, path)
    if not full.startswith("/"):
        full = os.path.join(os.getcwd(), full)
    print("  ->", full)
    with tmpdir.as_cwd():
        with open(full) as f:
            exec(f.read(), dict(__file__=full), {})


if __name__ == "__main__":
    from earthkit.meteo.utils.testing import main

    main(__file__)
