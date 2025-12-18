# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Union

import numpy as np
from earthkit.utils.array import array_namespace
from numpy.typing import ArrayLike


class AuxBottomLayer:
    def __init__(self, aux_data, aux_coord, data, coord, xp):
        aux_data = xp.atleast_1d(aux_data)
        aux_coord = xp.atleast_1d(aux_coord)

        if aux_coord.shape != coord.shape:
            aux_coord = xp.broadcast_to(aux_coord, coord.shape)
        if aux_coord.shape != data.shape:
            aux_coord = xp.broadcast_to(aux_coord, data.shape)
        if aux_data.shape != data.shape:
            aux_data = xp.broadcast_to(aux_data, data.shape)

        mask = aux_coord > coord
        self.coord = xp.where(mask, aux_coord, coord)
        self.data = xp.where(mask, aux_data, data)


class AuxTopLayer:
    def __init__(self, aux_data, aux_coord, data, coord, xp):
        aux_data = xp.atleast_1d(aux_data)
        aux_coord = xp.atleast_1d(aux_coord)

        if aux_coord.shape != coord.shape:
            aux_coord = xp.broadcast_to(aux_coord, coord.shape)
        if aux_coord.shape != data.shape:
            aux_coord = xp.broadcast_to(aux_coord, data.shape)
        if aux_data.shape != data.shape:
            aux_data = xp.broadcast_to(aux_data, data.shape)

        mask = aux_coord < coord
        self.coord = xp.where(mask, aux_coord, coord)
        self.data = xp.where(mask, aux_data, data)

        # print("AuxTopLayer:", self.coord, self.data)


class MonotonicInterpolator:
    def __call__(
        self,
        data: ArrayLike,
        coord: Union[ArrayLike, list, tuple, float, int],
        target_coord: Union[ArrayLike, list, tuple, float, int],
        interpolation: str = "linear",
        aux_min_level_data=None,
        aux_min_level_coord=None,
        aux_max_level_data=None,
        aux_max_level_coord=None,
    ):

        if interpolation not in ["linear", "log", "nearest"]:
            raise ValueError(
                f"Unknown interpolation method '{interpolation}'. Supported methods are 'linear', 'log' and 'nearest'."
            )

        xp = array_namespace(data, coord)
        target_coord = xp.atleast_1d(target_coord)
        coord = xp.atleast_1d(coord)
        data = xp.asarray(data)

        # Ensure levels are in descending order with respect to the values in the first
        # dimension of coord
        first = [0] * xp.ndim(coord)
        last = [0] * xp.ndim(coord)
        first = tuple([0] + first[1:])
        last = tuple([-1] + last[1:])

        if coord[first] < coord[last]:
            coord = xp.flip(coord, axis=0)
            data = xp.flip(data, axis=0)

        nlev = data.shape[0]
        if nlev < 2:
            raise ValueError("At least two levels are required for interpolation.")

        if data.shape[0] != coord.shape[0]:
            raise ValueError(
                f"The first dimension of data and that of coord must match! {data.shape=} {coord.shape=} {data.shape[0]} != {coord.shape[0]}"
            )

        self.data_is_scalar = xp.ndim(data[0]) == 0
        self.coord_is_scalar = xp.ndim(coord[0]) == 0
        self.target_is_scalar = xp.ndim(target_coord[0]) == 0

        same_shape = data.shape == coord.shape
        if same_shape:
            if self.data_is_scalar and not self.target_is_scalar:
                raise ValueError("If values and p have the same shape, they cannot both be scalars.")
            if (
                not self.data_is_scalar
                and not self.target_is_scalar
                and data.shape[1:] != target_coord.shape[1:]
            ):
                raise ValueError(
                    "When values and target_p have different shapes, target_p must be a scalar or a 1D array."
                )

        if not same_shape and xp.ndim(coord) != 1:
            raise ValueError(
                f"When values and p have different shapes, p must be a scalar or a 1D array. {data.shape=} {coord.shape=} {xp.ndim(coord)}"
            )

        # initialize the output array
        res = xp.empty((len(target_coord),) + data.shape[1:], dtype=data.dtype)

        if same_shape:
            if self.data_is_scalar:
                data = xp.broadcast_to(data, (1, nlev)).T
                coord = xp.broadcast_to(coord, (1, nlev)).T
            else:
                assert not self.data_is_scalar
                assert not self.coord_is_scalar
        else:
            assert self.coord_is_scalar
            # print(f"scalar_info.target: {scalar_info.target}")
            # if scalar_info.target:
            #     return _to_level_1(data, coord, nlev, target_coord, interpolation, scalar_info, xp, res, aux_bottom, aux_top)
            # else:
            #     coord = xp.broadcast_to(coord, (nlev,) + data.shape[1:]).T

            if not self.target_is_scalar:
                coord = xp.broadcast_to(coord, (nlev,) + data.shape[1:]).T

        # assert data.shape == coord.shape, f"{data.shape=} != {coord.shape=}"

        # reevaluate shape after possible broadcasting
        same_shape = data.shape == coord.shape

        aux_bottom = None
        aux_top = None
        if aux_max_level_coord is not None and aux_max_level_data is not None:
            aux_bottom = AuxBottomLayer(aux_max_level_data, aux_max_level_coord, data[0], coord[0], xp)

        if aux_min_level_coord is not None and aux_min_level_data is not None:
            aux_top = AuxTopLayer(aux_min_level_data, aux_min_level_coord, data[-1], coord[-1], xp)

        self.xp = xp
        self.data = data
        self.coord = coord
        self.target_coord = target_coord
        self.interpolation = interpolation
        self.aux_bottom = aux_bottom
        self.aux_top = aux_top
        self.nlev = nlev

        if same_shape or not self.target_is_scalar:
            self.compute(res)
        else:
            return self.simple_compute(res)

        return res

    # values and p have the same shape
    def compute(self, res):
        xp = self.xp

        # The coordinate levels must be ordered in descending order with respect to the
        # first dimension. So index 0 has the highest coordinate values, index -1 the lowest,
        # as if it were pressure levels in the atmosphere. The algorithm below agnostic to the
        # actual meaning of the coordinate in the real atmosphere. The terms "top" and "bottom"
        # are used with respect to this coordinate ordering in mind and not related to actual
        # vertical position in the atmosphere. Of course, if the  coordinate is pressure these
        # two definitions coincide.
        for target_idx, tc in enumerate(self.target_coord):

            # find the level below the target
            idx_bottom = (self.coord > tc).sum(0)
            idx_bottom = xp.atleast_1d(idx_bottom)

            # print(f"tc: {tc} i_top: {i_top}")
            # initialise the output array
            r = xp.empty(idx_bottom.shape)

            # mask when the target is below the lowest level
            mask_bottom = idx_bottom == 0

            # mask when the target is above the highest level
            mask_top = idx_bottom == self.nlev

            # mask when the target is in the coordinate range
            mask_mid = ~(mask_bottom | mask_top)

            if xp.any(mask_bottom):
                self._compute_bottom(mask_bottom, r, tc)

            if xp.any(mask_top):
                self._compute_top(mask_top, r, tc)

            if xp.any(mask_mid):
                self._compute_mid(idx_bottom, mask_mid, r, tc)

            if self.data_is_scalar:
                r = r[0]

            res[target_idx] = r

        return res

    def _compute_bottom(self, mask, r, tc):
        xp = self.xp
        if xp.any(mask):
            if self.aux_bottom is None:
                if self.interpolation == "nearest":
                    r[mask] = self.data[0][mask]
                else:
                    r[mask] = np.nan
                    m = mask & (xp.isclose(self.coord[0], tc))
                    r[m] = self.data[0][m]
            else:
                r[mask] = np.nan
                aux_mask = mask & (self.aux_bottom.coord > self.coord[0]) & (self.aux_bottom.coord >= tc)
                if xp.any(aux_mask):
                    d_top = self.data[0][aux_mask]
                    d_bottom = self.aux_bottom.data[aux_mask]
                    c_top = self.coord[0][aux_mask]
                    c_bottom = self.aux_bottom.coord[aux_mask]

                    if not self.target_is_scalar:
                        tc = tc[aux_mask]

                    factor = self._factor(c_top, c_bottom, tc, self.interpolation, xp)
                    r[aux_mask] = (1.0 - factor) * d_bottom + factor * d_top

    def _compute_top(self, mask, r, tc):
        xp = self.xp
        if xp.any(mask):
            if self.aux_top is None:
                if self.interpolation == "nearest":
                    r[mask] = self.data[-1][mask]
                else:
                    r[mask] = np.nan
                    m = mask & (xp.isclose(self.coord[-1], tc))
                    r[m] = self.data[-1][m]
            else:
                # print("Using aux top layer")
                r[mask] = np.nan
                # print("aux top coord:", self.aux_top.coord, "self.coord[-1]:", self.coord[-1])
                aux_mask = mask & (self.aux_top.coord < self.coord[-1]) & (self.aux_top.coord <= tc)
                if xp.any(aux_mask):
                    d_top = self.aux_top.data[aux_mask]
                    d_bottom = self.data[-1][aux_mask]
                    c_top = self.aux_top.coord[aux_mask]
                    c_bottom = self.coord[-1][aux_mask]
                    # print(f"tc: {tc} c_top: {c_top} c_bottom: {c_bottom} d_top: {d_top} d_bottom: {d_bottom}")

                    if not self.target_is_scalar:
                        tc = tc[aux_mask]

                    factor = self._factor(c_top, c_bottom, tc, self.interpolation, xp)
                    r[aux_mask] = (1.0 - factor) * d_bottom + factor * d_top

    def _compute_mid(self, idx_bottom, mask, r, tc):
        xp = self.xp
        i_lev = idx_bottom
        indices = np.indices(i_lev.shape)
        masked_indices = tuple(dim[mask] for dim in indices)
        top = (idx_bottom[mask],) + masked_indices
        bottom = (idx_bottom[mask] - 1,) + masked_indices
        c_top = self.coord[top]
        c_bottom = self.coord[bottom]
        d_top = self.data[top]
        d_bottom = self.data[bottom]

        # print(f"tc: {tc} c_top: {c_top} c_bottom: {c_bottom} f_top: {f_top} f_bottom: {f_bottom}")

        if not self.target_is_scalar:
            tc = tc[mask]

        factor = self._factor(c_top, c_bottom, tc, self.interpolation, xp)
        r[mask] = (1.0 - factor) * d_bottom + factor * d_top

    # values and p have a different shape, p is 1D and target is 1D
    def simple_compute(self, res):
        xp = self.xp

        # The coordinate levels must be ordered in descending order with respect to the
        # first dimension. So index 0 has the highest coordinate values, index -1 the lowest,
        # as if it were pressure levels in the atmosphere. The algorithm below agnostic to the
        # actual meaning of the coordinate in the real atmosphere. The terms "top" and "bottom"
        # are used with respect to this coordinate ordering in mind and not related to actual
        # vertical position in the atmosphere. Of course, if the  coordinate is pressure these
        # two definitions coincide.

        # initialize the output array
        # res = xp.empty((len(target_coord),) + data.shape[1:], dtype=data.dtype)

        # print("src_coord", src_coord, compare)

        # p on a level is a number
        for target_idx, tc in enumerate(self.target_coord):
            # initialise the output array
            r = xp.empty(self.data.shape[1:])
            r = xp.atleast_1d(r)

            # find the level below the target
            idx_bottom = (self.coord > tc).sum(0)

            # print(f"tc: {tc} i_top: {i_top}", src_coord[0])
            if idx_bottom == 0:
                if self.interpolation == "nearest":
                    r = self.data[0]
                else:
                    if xp.isclose(self.coord[0], tc):
                        r = self.data[0]
                        # print(f"tc: {tc} r: {r}")
                    else:
                        r.fill(xp.nan)

            elif idx_bottom == self.nlev:
                if self.interpolation == "nearest":
                    r = self.data[-1]
                else:
                    if xp.isclose(self.coord[-1], tc):
                        r = self.data[-1]
                    else:
                        r.fill(xp.nan)
            else:
                top = idx_bottom
                bottom = idx_bottom - 1

                c_top = self.coord[top]
                c_bottom = self.coord[bottom]

                d_top = self.data[top]
                d_bottom = self.data[bottom]

                factor = self._factor(c_top, c_bottom, tc, self.interpolation, xp)
                r = (1.0 - factor) * d_bottom + factor * d_top

            res[target_idx] = r

        return res

    @staticmethod
    def _factor(c_top, c_bottom, tc, interpolation, xp):
        if interpolation == "linear":
            factor = (tc - c_bottom) / (c_top - c_bottom)
        elif interpolation == "log":
            factor = (xp.log(tc) - xp.log(c_bottom)) / (xp.log(c_top) - xp.log(c_bottom))
        elif interpolation == "nearest":
            dist_top = xp.abs(c_top - tc)
            dist_bottom = xp.abs(c_bottom - tc)
            factor = xp.where(dist_top < dist_bottom, 1.0, 0.0)
        return factor
