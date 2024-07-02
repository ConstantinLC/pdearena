# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import functools
from typing import Optional

import h5py
import torch
import torchdata.datapipes as dp

from pdearena.data.datapipes_common import build_datapipes
import numpy as np

import random

class KolmogorovDatasetOpener(dp.iter.IterDataPipe):
    """DataPipe to load Navier-Stokes dataset.

    Args:
        dp (dp.iter.IterDataPipe): List of `hdf5` files containing Navier-Stokes data.
        mode (str): Mode to load data from. Can be one of `train`, `val`, `test`.
        limit_trajectories (int, optional): Limit the number of trajectories to load from individual `hdf5` file. Defaults to None.
        usegrid (bool, optional): Whether to output spatial grid or not. Defaults to False.

    Yields:
        (Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]): Tuple containing particle scalar field, velocity vector field, and optionally buoyancy force parameter value  and spatial grid.
    """

    def __init__(self, dp, mode: str, limit_trajectories: Optional[int] = None, usegrid: bool = False, conditioned: bool = False) -> None:
        super().__init__()
        self.dp = dp
        self.mode = mode
        self.limit_trajectories = limit_trajectories
        self.usegrid = usegrid
        self.conditioned = conditioned
        
        self.time_step = 16
        self.trajectories = []
        self.resolution = "32"
        self.high_resolution = None

        for path in self.dp:
            print(path)
            f = h5py.File(path, "r")
            data_lowres_key = f"full_prefix_{self.resolution}x{self.resolution}"
            data_lowres = np.array(f[data_lowres_key])[:, 128:]
            data_lowres = torch.Tensor(data_lowres)
            data_lowres = torch.swapaxes(data_lowres, 0, 1)

            if self.high_resolution is not None:
                data_highres_key = f"full_prefix_{self.high_resolution}x{self.high_resolution}"
                data_highres = np.array(f[data_highres_key])[:, 128:]
                data_highres = torch.Tensor(data_highres)
                data_highres = torch.swapaxes(data_highres, 0, 1)
                data_full = torch.cat((data_lowres.repeat_interleave(2, axis=-1).repeat_interleave(2, axis=-2), data_highres), axis=1)
            else:
                data_full = data_lowres

            time_gaps = range(self.time_step)

            for time_gap in time_gaps:
                print(data_full[time_gap::16].shape)
                self.trajectories.append(data_full[time_gap::16])
        
        if self.mode == "train":
            random.shuffle(self.trajectories)
        

    def __iter__(self):
        
        for data_full in self.trajectories:
            yield data_full, None, None, None

def _train_filter(fname):
    return "train" in fname and "hdf5" in fname and not "seed" in fname.split("-")[-1]


def _valid_filter(fname):
    return "valid" in fname and "hdf5" in fname and not "seed" in fname.split("-")[-1]


def _test_filter(fname):
    return "test" in fname and "hdf5" in fname and "0000" in fname


train_datapipe_kg = functools.partial(
    build_datapipes,
    dataset_opener=KolmogorovDatasetOpener,
    filter_fn=_train_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="train",
)
onestep_valid_datapipe_kg = functools.partial(
    build_datapipes,
    dataset_opener=KolmogorovDatasetOpener,
    filter_fn=_valid_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,

    mode="valid",
    onestep=True,
)
trajectory_valid_datapipe_kg = functools.partial(
    build_datapipes,
    dataset_opener=KolmogorovDatasetOpener,
    filter_fn=_valid_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="valid",
    onestep=False,
)

onestep_test_datapipe_kg = functools.partial(
    build_datapipes,
    dataset_opener=KolmogorovDatasetOpener,
    filter_fn=_test_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="test",
    onestep=True,
)

trajectory_test_datapipe_kg = functools.partial(
    build_datapipes,
    dataset_opener=KolmogorovDatasetOpener,
    filter_fn=_test_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="test",
    onestep=False,
)
