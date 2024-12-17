# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import glob
import os
from pathlib import Path

import numpy as np


class MulranDataset:
    def __init__(self, data_dir: Path, *_, **__):
        self.data_dir = os.path.realpath(data_dir)
        self.sequence_id = os.path.basename(data_dir)
        self.velodyne_dir = os.path.join(self.data_dir, "Ouster/")

        self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))

        try:
            self.gt_closure_indices = np.loadtxt(
                os.path.join(self.data_dir, "loop_closure", "gt_closures.txt")
            )
        except FileNotFoundError:
            self.gt_closure_indices = None

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, idx):
        return self.read_point_cloud(self.scan_files[idx])

    def read_point_cloud(self, file_path: str):
        points = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))[:, :3]
        return points.astype(np.float64)
