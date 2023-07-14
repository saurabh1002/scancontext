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
from python.tools.pipeline_results import PipelineResults
from python.tools.progress_bar import get_progress_bar
from python.tools.visualization import draw_scan_context


class ScanContextPipeline:
    def __init__(
        self,
        dataset,
        visualize: Optional[bool] = False,
    ):
        self._dataset = dataset
        self._first = 0
        self._last = len(self._dataset)

        self._visualize = visualize
        self.results_dir = None

        self.scan_context = ScanContext()
        self.dataset_name = self._dataset.__class__.__name__

        self.gt_closure_indices = self._dataset.gt_closure_indices

        scan_context_thresholds = np.arange(0.1, 0.4, 0.05)
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
            query_idx, nearest_node_idx, dist, init_yaw = self.scan_context.check_for_closure()
            if self._visualize:
                draw_scan_context(
                    [
                        self.scan_context.get_scan_context(query_idx),
                        self.scan_context.get_scan_context(nearest_node_idx),
                    ]
                )
            if query_idx != -1 and nearest_node_idx != -1:
                self.results.append(query_idx, nearest_node_idx, dist)

    def _run_evaluation(self) -> None:
        self.results.compute_metrics()

    def _log_to_file(self) -> None:
        self.results_dir = self._create_results_dir()
        if self.gt_closure_indices is not None:
            self.results.log_to_file_pr(os.path.join(self.results_dir, "metrics.txt"))
        self.results.log_to_file_closures(self.results_dir)

    def _create_results_dir(self) -> Path:
        def get_timestamp() -> str:
            return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        results_dir = os.path.join(self._dataset.data_dir, "scan_context_results", get_timestamp())
        latest_dir = os.path.join(self._dataset.data_dir, "scan_context_results", "latest")
        os.makedirs(results_dir, exist_ok=True)
        os.unlink(latest_dir) if os.path.exists(latest_dir) or os.path.islink(latest_dir) else None
        os.symlink(results_dir, latest_dir)

        return results_dir
