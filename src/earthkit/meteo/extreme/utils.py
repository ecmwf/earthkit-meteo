import xarray as xr


def wrapper(func, clim, ens, clim_dim: str = "number", ens_dim: str = "number", **kwargs):
    if clim_dim == ens_dim:
        clim_dim = f"clim_{ens_dim}"
        clim = clim.rename({ens_dim: clim_dim})
    args = []
    return_dataset = False
    for arg, dim in [(clim, clim_dim), (ens, ens_dim)]:
        if isinstance(arg, xr.Dataset):
            arg = arg.to_dataarray().squeeze(dim="variable", drop=True)
            return_dataset = True
        arg = arg.transpose(dim, ...)
        args.append(arg)
    out = xr.apply_ufunc(
        func,
        *args,
        input_core_dims=[x.dims for x in args],
        output_core_dims=[args[1][{ens_dim: 0}].dims],
        kwargs=kwargs,
    )
    if return_dataset:
        out = out.to_dataset(name=list(ens.data_vars.keys())[0])
    return out
