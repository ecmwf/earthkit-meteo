"""
Wind related functions operating on xarray objects.
"""

from .wind import coriolis, direction, polar_to_xy, speed, w_from_omega, windrose, xy_to_polar

__all__ = [
    "coriolis",
    "direction",
    "polar_to_xy",
    "speed",
    "w_from_omega",
    "windrose",
    "xy_to_polar",
]
