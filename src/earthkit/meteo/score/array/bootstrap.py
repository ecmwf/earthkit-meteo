import random

from earthkit.utils.array import array_namespace


def iter_samples(x, *args, sample_axis=0, n_iter=100, n_samples=None, randrange=random.randrange):
    """Iterate over resampled arrays for bootstrapping

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


def resample(x, *args, sample_axis=0, out_axis=0, n_iter=100, n_samples=None, randrange=random.randrange):
    """Resample arrays for bootstrapping

    Parameters
    ----------
    x, *args: array-like
        Inputs to the wrapped function, sampled for bootstrapping. Must have
        the same size along ``sample_axis``
    sample_axis: int or list of int
        Sample along this axis (either same for all or one per argument)
    out_axis: int
        Stack samples along this axis
    n_iter: int
        Number of bootstrapping iterations
    n_samples: int or None
        Number of samples for each iteration. If None, use the number of
        inputs (size of ``x`` along the sampling axis)
    randrange: function (int -> int)
        Random generator for integers: `randrange(n)` should return an
        integer in `range(n)`

    Returns
    ------
    tuple
        Resampled arrays (one for each iteration)
    """
    xp = array_namespace(x, *args)
    n_arrays = len(args) + 1
    samples = [[] for _ in range(n_arrays)]
    samples_it = iter_samples(
        x, *args, sample_axis=sample_axis, n_iter=n_iter, n_samples=n_samples, randrange=randrange
    )
    for sample in samples_it:
        for i in range(n_arrays):
            samples[i].append(sample[i])
    return tuple(xp.stack(sampled_arr, axis=out_axis) for sampled_arr in samples)


def bootstrap(
    func,
    x,
    *args,
    sample_axis=0,
    out_axis=0,
    n_iter=100,
    n_samples=None,
    randrange=random.randrange,
    **kwargs,
):
    """Run bootstrapping

    Parameters
    ----------
    func: function ((array, ..., **kwargs) -> array)
        Function to bootstrap
    x, *args: array-like
        Inputs to ``function``, sampled for bootstrapping. Must have the same
        size along ``sample_axis``
    sample_axis: int or list of int
        Sample along this axis (either same for all or one per argument)
    out_axis: int
        Stack samples along this axis
    n_iter: int
        Number of bootstrapping iterations
    n_samples: int or None
        Number of samples for each iteration. If None, use the number of
        inputs (size of ``x`` along the sampling axis)
    randrange: function (int -> int)
        Random generator for integers: `randrange(n)` should return an
        integer in `range(n)`
    **kwargs
        Additional keyword arguments to ``func``

    Returns
    -------
    array-like
        Aggregated results of the bootstrapping process
    """
    xp = array_namespace(x, *args)
    samples = iter_samples(
        x,
        *args,
        sample_axis=sample_axis,
        n_iter=n_iter,
        n_samples=n_samples,
        randrange=randrange,
    )
    return xp.stack([func(*sampled, **kwargs) for sampled in samples], axis=out_axis)
