import functools
import random
from dataclasses import dataclass

from earthkit.utils.array import array_namespace


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


def resample(x, *args, sample_axis=0, n_iter=100, n_samples=None, randrange=random.randrange):
    """Resample arrays for bootstrapping

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

    Yields
    ------
    tuple
        Resampled arrays (one for each iteration)
    """
    args = (x,) + args
    n_arrays = len(args)
    if isinstance(sample_axis, int):
        sample_axis = [sample_axis for _ in range(n_arrays)]
    else:
        assert len(sample_axis) == n_arrays, "sample_axis must have one element per input array"
    xp = array_namespace(*args)
    arrays = tuple((xp.asarray(arr), axis) for arr, axis in zip(args, sample_axis))
    n_inputs = x.shape[sample_axis[0]]
    assert all(
        y.shape[axis] == n_inputs for y, axis in arrays
    ), "Input arrays must have the same size along the sampling axis"
    if n_samples is None:
        n_samples = n_inputs
    for _ in range(n_iter):
        indices = [randrange(n_inputs) for _ in range(n_samples)]
        sampled = tuple(xp.take(y, indices=indices, axis=axis) for y, axis in arrays)
        yield sampled


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
        xp = array_namespace(x, *args)
        samples = resample(
            x,
            *args,
            sample_axis=sample_axis,
            n_iter=n_iter,
            n_samples=n_samples,
            randrange=randrange,
        )
        results = [self.func(*sampled, **kwargs) for sampled in samples]
        return BootstrapResult(n_iter, n_samples, xp.stack(results, axis=0))


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
