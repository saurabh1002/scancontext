# MIT License
#
# Copyright (c) 2024 Saurabh Gupta, Ignacio Vizzo, Tiziano Guadagnino,
# Benedikt Mersch, Cyrill Stachniss.
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
import open3d as o3d


class IPBCarDataset:
    def __init__(self, data_dir: Path, *_, **__):
        self.sequence_id = os.path.basename(data_dir)
        self.sequence_dir = os.path.join(data_dir, "points")
        self.scan_files = sorted(glob.glob(self.sequence_dir + "/*.ply"))

        self.gt_file = os.path.join(data_dir, "poses.npy")
        self.gt_poses = self.load_poses(self.gt_file)

        try:
            self.gt_closure_indices = np.loadtxt(
                os.path.join(self.sequence_dir, "loop_closure", "local_map_gt_closures.txt")
            )
            self.local_maps_scan_range = np.load(
                os.path.join(self.sequence_dir, "MapClosures", "local_maps_scan_index_range.npy")
            )
        except FileNotFoundError:
            self.gt_closure_indices = None

        if len(self.scan_files) == 0:
            raise ValueError(f"Tried to read point cloud files in {data_dir} but none found")

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, idx):
        return self.read_point_cloud(idx)

    def get_data(self, idx: int):
        file_path = self.scan_files[idx]
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points)

    def read_point_cloud(self, idx: int):
        data = self.get_data(idx)
        points = data[:, :3]
        return points.astype(np.float64)

    def load_poses(self, poses_file):
        poses = np.load(poses_file)
        poses = np.linalg.inv(poses[0]) @ poses

        return poses
