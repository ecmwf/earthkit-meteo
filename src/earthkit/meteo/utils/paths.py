import logging
import os

LOG = logging.getLogger(__name__)


_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


def earthkit_conf_file(*args):
    return os.path.join(_ROOT_DIR, "conf", *args)
