import functools
import random
from dataclasses import dataclass

from earthkit.utils.array import array_namespace

try:
    import xarray as xr
except ModuleNotFoundError:
    xr = None

from . import array


def resample(x, *args, **kwargs):
    if xr is not None and isinstance(x, xr.DataArray):
        n_arrays = len(args) + 1
        dim = kwargs.get("dim", None)
        if dim is None:
            raise TypeError("resample with xarray arguments requires 'dim'")
        in_dims = [[dim] for _ in range(n_arrays)]
        sample_dim = kwargs.get("sample_dim", "sample")
        out_dims = [[sample_dim] for _ in range(n_arrays)]
        return xr.apply_ufunc(
            functools.partial(array.resample, sample_axis=-1, out_axis=-1, **kwargs),
            x,
            *args,
            input_core_dims=in_dims,
            output_core_dims=out_dims,
        )

    return array.resample(x, *args, **kwargs)


def _bootstrap_xarray(
    func,
    *args,
    dim=None,
    sample_dim="sample",
    n_iter=100,
    n_samples=None,
    randrange=random.randrange,
    **kwargs,
):
    assert xr is not None
    if dim is None:
        raise TypeError("bootstrap with xarray arguments requires 'dim'")
    n_inputs = args[0].sizes[dim]
    assert all(
        arr.sizes[dim] == n_inputs for arr in args
    ), "Input arrays must have the same size along the sampling axis"
    if n_samples is None:
        n_samples = n_inputs
    results = []
    for _ in range(n_iter):
        indices = [randrange(n_inputs) for _ in range(n_samples)]
        sampled = tuple(arr.isel({dim: indices}) for arr in args)
        results.append(func(*sampled, **kwargs))
    return xr.concat(results, sample_dim)


def bootstrap(func, x, *args, **kwargs):
    if xr is not None and isinstance(x, xr.DataArray):
        return _bootstrap_xarray(func, x, *args, **kwargs)
    return array.bootstrap(func, x, *args, **kwargs)


@dataclass
class BootstrapResult:
    """Container for the result of a bootstrapping process"""

    #: number of bootstrapping iterations
    n_iter: int

    #: number of samples for each iteration
    n_samples: int

    #: return values for each iteration, first dimension is the iteration
    results: object

    def quantiles(self, threshold: float):
        """Compute quantiles associated with the given threshold

        The first dimension of the returned array has length 2 corresponding to
        the low (``threshold``) and high (``1 - threshold``) quantiles
        """
        return self._xp.quantile(self.results, [threshold, 1.0 - threshold], axis=0)

    def sig_mask(self, threshold: float, sig_value: float):
        """Compute the significance mask for the given threshold and significant value"""
        low, high = self.quantiles(threshold)
        return self._xp.logical_or(high < sig_value, low > sig_value)

    @property
    def _xp(self):
        return array_namespace(self.results)


class Bootstrappable:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def bootstrap(
        self,
        x,
        *args,
        sample_axis=0,
        n_iter=100,
        n_samples=None,
        randrange=random.randrange,
        **kwargs,
    ):
        """Run bootstrapping

        Parameters
        ----------
        x, *args: array-like
            Inputs to the wrapped function, sampled for bootstrapping. Must have
            the same size along ``sample_axis``
        sample_axis: int or list of int
            Sample along this axis (either same for all or one per argument)
        n_iter: int
            Number of bootstrapping iterations
        n_samples: int or None
            Number of samples for each iteration. If None, use the number of
            inputs (size of ``x`` along the sampling axis)
        randrange: function (int -> int)
            Random generator for integers: `randrange(n)` should return an
            integer in `range(n)`
        **kwargs
            Additional keyword arguments to the wrapped function

        Returns
        -------
        BootstrapResult
            Aggregated results of the bootstrapping process
        """
        results = bootstrap(
            self.func,
            x,
            *args,
            sample_axis=sample_axis,
            n_iter=n_iter,
            n_samples=n_samples,
            randrange=randrange,
        )
        return BootstrapResult(n_iter, n_samples, results)


def enable_bootstrap(func):
    """Enable bootstrapping on a binary scoring function

    The function will be wrapped in a callable object (see
    :class:`Bootstrappable`). Calling the object will call the function
    directly. The object will have a :meth:`Bootstrappable.bootstrap` method to
    run bootstrapping.

    Examples
    --------

    ::
        @enable_bootstrap
        def mse(x, y, axis=-1):
            return np.mean(np.square(y - x), axis=axis)

        x = ...
        y = ...
        mse(x, y)  # normal call, return MSE
        bresult = mse.bootstrap(x, y)  # bootstrapping
        bresult.quantiles(0.2)
        bresult.sig_mask(0.05, 0.0)
    """

    wrapper = Bootstrappable(func)
    functools.update_wrapper(wrapper, func)
    return wrapper
