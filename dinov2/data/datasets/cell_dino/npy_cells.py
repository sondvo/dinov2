# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC licence,
# found in the LICENSE_CELL_DINO_CODE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Callable, Optional, Union

import numpy as np
import torch

from ..extended import ExtendedVisionDataset


class _Split(Enum):
    ALL = "all"


class NPYCells(ExtendedVisionDataset):
    """Dataset for preprocessed microscopy arrays stored as [N, C, H, W] in a .npy file.

    `root` should point to the `.npy` file.
    """

    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "NPYCells.Split" = _Split.ALL,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform, **kwargs)
        self.split = split
        self._array = np.load(root, mmap_mode="r")

        if self._array.ndim != 4:
            raise ValueError(f"Expected .npy shape [N, C, H, W], got {self._array.shape}")
        if self._array.shape[1] != 4:
            raise ValueError(f"Expected 4 channels, got {self._array.shape[1]}")

    def __len__(self) -> int:
        return int(self._array.shape[0])

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        return 0

    def __getitem__(self, index: int):
        image = np.asarray(self._array[index]).astype(np.float32, copy=False)

        # Cell-DINO augmentations apply Div255, so remap uint16-like ranges to 0..255 first.
        img_max = float(np.max(image)) if image.size > 0 else 0.0
        if img_max > 255.0:
            image = image * (255.0 / 65535.0)
        image = torch.from_numpy(image)

        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target