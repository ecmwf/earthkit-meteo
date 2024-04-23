from . import array


def julian_day(*args, **kwargs):
    return array.julian_day(*args, **kwargs)


def solar_declination_angle(*args, **kwargs):
    return array.solar_declination_angle(*args, **kwargs)


def cos_solar_zenith_angle(*args, **kwargs):
    return array.cos_solar_zenith_angle(*args, **kwargs)


def cos_solar_zenith_angle_integrated(*args, **kwargs):
    return array.cos_solar_zenith_angle_integrated(*args, **kwargs)


def incoming_solar_radiation(*args, **kwargs):
    return array.incoming_solar_radiation(*args, **kwargs)


def toa_incident_solar_radiation(*args, **kwargs):
    return array.toa_incident_solar_radiation(*args, **kwargs)
