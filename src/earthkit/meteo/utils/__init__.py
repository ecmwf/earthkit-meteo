import sys


def is_module_loaded(module_name):
    return module_name in sys.modules
