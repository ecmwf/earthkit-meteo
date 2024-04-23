from . import array  # noqa


def cpf(clim, ens, sort_clim=True, sort_ens=True):
    return array.cpf(clim, ens, sort_clim, sort_ens)


def efi(clim, ens, eps=-0.1):
    return array.efi(clim, ens, eps)


def sot(clim, ens, perc, eps=-1e4):
    return array.sot(clim, ens, perc, eps)


def sot_unsorted(clim, ens, perc, eps=-1e4):
    return array.sot_unsorted(clim, ens, perc, eps)
