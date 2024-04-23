from . import array


def speed(*args, **kwargs):
    return array.speed(*args, **kwargs)


def _direction_meteo(*args, **kwargs):
    return array._direction_meteo(*args, **kwargs)


def _direction_polar(*args, **kwargs):
    return array._direction_polar(*args, **kwargs)


def direction(*args, **kwargs):
    return array.direction(*args, **kwargs)


def xy_to_polar(*args, **kwargs):
    return array.xy_to_polar(*args, **kwargs)


def polar_to_xy(*args, **kwargs):
    return array.polar_to_xy(*args, **kwargs)


def w_from_omega(*args, **kwargs):
    return array.w_from_omega(*args, **kwargs)


def coriolis(*args, **kwargs):
    return array.coriolis(*args, **kwargs)


def windrose(*args, **kwargs):
    return array.windrose(*args, **kwargs)
