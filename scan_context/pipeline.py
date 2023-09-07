# MIT License
#
# Copyright (c) 2023 Saurabh Gupta, Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch,
# Cyrill Stachniss.
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
import datetime
import os
from pathlib import Path
from typing import Optional

import numpy as np

from pybind.scan_context import ScanContext
from scan_context.tools.pipeline_results import PipelineResults
from scan_context.tools.progress_bar import get_progress_bar
from scan_context.tools.visualization import draw_scan_context


class ScanContextPipeline:
    def __init__(
        self,
        dataset,
        results_dir: Path,
        visualize: Optional[bool] = False,
    ):
        self._dataset = dataset
        self._first = 0
        self._last = len(self._dataset)

        self._visualize = visualize
        self.results_dir = results_dir

        self.scan_context = ScanContext()
        self.dataset_name = self._dataset.sequence_id

        self.closures = []
        self.gt_closure_indices = self._dataset.gt_closure_indices

        scan_context_thresholds = np.arange(0.1, 1.0, 0.05)
        self.results = PipelineResults(
            self.gt_closure_indices, self.dataset_name, scan_context_thresholds
        )

    def run(self):
        self._run_pipeline()
        if self.gt_closure_indices is not None:
            self._run_evaluation()
        self._log_to_file()

        return self.results

    def _run_pipeline(self):
        scan = self._dataset[self._first]
        self.scan_context.process_new_scan(scan)
        for query_idx in get_progress_bar(self._first + 1, self._last):
            scan = self._dataset[query_idx]
            self.scan_context.process_new_scan(scan)
            query_idx, candidate_ids, candidate_dists, candidate_yaws = self.scan_context.check_for_closure()
            if self._visualize:
                for candidate_id in candidate_ids:
                    draw_scan_context(
                        [
                            self.scan_context.get_scan_context(query_idx),
                            self.scan_context.get_scan_context(candidate_id),
                        ]
                    )
            if query_idx != -1:
                for candidate_id, dist, yaw in zip(candidate_ids, candidate_dists, candidate_yaws):
                    if dist < 0.4:
                        relative_tf = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0], [np.sin(yaw), np.cos(yaw), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        self.closures.append(np.r_[candidate_id, query_idx, relative_tf.flatten()])
                    self.results.append(query_idx, candidate_id, dist)

    def _run_evaluation(self) -> None:
        self.results.compute_metrics()

    def _log_to_file(self) -> None:
        self.results_dir = self._create_results_dir()
        if self.gt_closure_indices is not None:
            self.results.log_to_file_pr(os.path.join(self.results_dir, "metrics.txt"))
        self.results.log_to_file_closures(self.results_dir)
        np.savetxt(os.path.join(self.results_dir, "closures.txt"), np.asarray(self.closures))

    def _create_results_dir(self) -> Path:
        def get_timestamp() -> str:
            return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        results_dir = os.path.join(
            self.results_dir, "scan_context_results", self.dataset_name, get_timestamp()
        )
        latest_dir = os.path.join(
            self.results_dir, "scan_context_results", self.dataset_name, "latest"
        )
        os.makedirs(results_dir, exist_ok=True)
        os.unlink(latest_dir) if os.path.exists(latest_dir) or os.path.islink(latest_dir) else None
        os.symlink(results_dir, latest_dir)

        return results_dir
