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
import os
from pathlib import Path

import numpy as np

from scan_context.scan_context import ScanContext
from scan_context.tools.pipeline_results import PipelineResults
from scan_context.tools.progress_bar import get_progress_bar


def scan_to_map(scan_query, scan_ref, query_local_maps_scan_range, ref_local_maps_scan_range):
    map_query = np.where(
        (scan_query >= query_local_maps_scan_range[:, 0])
        & (scan_query < query_local_maps_scan_range[:, 1])
    )[0][0]
    map_ref = np.where(
        (scan_ref >= ref_local_maps_scan_range[:, 0]) & (scan_ref < ref_local_maps_scan_range[:, 1])
    )[0][0]
    return map_query, map_ref


class ScanContextPipeline:
    def __init__(
        self,
        dataset_query,
        dataset_ref,
        results_dir: Path,
    ):
        self._query_dataset = dataset_query
        self._query_dataset_name = (
            self._query_dataset.sequence_id
            if hasattr(self._query_dataset, "sequence_id")
            else os.path.basename(self._query_dataset.data_dir)
        )
        self._ref_dataset = dataset_ref
        self._ref_dataset_name = (
            self._ref_dataset.sequence_id
            if hasattr(self._ref_dataset, "sequence_id")
            else os.path.basename(self._ref_dataset.data_dir)
        )

        self.results_dir = results_dir

        self.scan_context = ScanContext()

        self.closures = []
        base_dir_query = self._query_dataset.sequence_dir
        file_path_closures_query = os.path.join(
            base_dir_query,
            "loop_closure",
            f"{self._ref_dataset.sequence_id}_local_map_gt_closures.txt",
        )
        if os.path.exists(file_path_closures_query) and os.path.exists(file_path_closures_query):
            self.gt_closures = np.loadtxt(file_path_closures_query, dtype=int)
            print(f"[INFO] Found closure ground truth at {file_path_closures_query}")
        else:
            self.gt_closures = None
            print(f"[INFO] No closure ground truth found at {file_path_closures_query}")

        self.ref_local_maps_scan_range = self._ref_dataset.local_maps_scan_range
        self.query_local_maps_scan_range = self._query_dataset.local_maps_scan_range

        scan_context_thresholds = np.arange(0.1, 1.1, 0.1)
        self.results = PipelineResults(self.gt_closures, scan_context_thresholds)

    def run(self):
        self._run_pipeline()
        self._run_evaluation()
        self._log_to_file()

        return self.results

    def _run_pipeline(self):
        for ref_idx in get_progress_bar(0, len(self._ref_dataset)):
            scan = self._ref_dataset[ref_idx]
            self.scan_context.process_new_scan(scan)

        for query_idx in get_progress_bar(0, len(self._query_dataset)):
            scan = self._query_dataset[query_idx]
            candidate_ids, candidate_dists, candidate_yaws = (
                self.scan_context.check_for_multisession_closure(scan)
            )
            for candidate_id, dist, yaw in zip(candidate_ids, candidate_dists, candidate_yaws):
                map_query, map_ref = scan_to_map(
                    query_idx,
                    candidate_id,
                    self.query_local_maps_scan_range,
                    self.ref_local_maps_scan_range,
                )
                self.results.append(map_ref, map_query, dist)
                if dist < 0.4:
                    relative_tf = np.array(
                        [
                            [np.cos(yaw), -np.sin(yaw), 0, 0],
                            [np.sin(yaw), np.cos(yaw), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ]
                    )
                    self.closures.append(np.r_[candidate_id, ref_idx, relative_tf.flatten()])

    def _run_evaluation(self) -> None:
        self.results.compute_metrics()

    def _log_to_file(self) -> None:
        self.results_dir = self._create_results_dir()
        self.results.log_to_file_pr(os.path.join(self.results_dir, "metrics.txt"))
        np.savetxt(
            os.path.join(self.results_dir, "multi-session_closures.txt"), np.asarray(self.closures)
        )

    def _create_results_dir(self) -> Path:
        results_dir = os.path.join(
            self.results_dir, f"{self._query_dataset_name}", f"{self._ref_dataset_name}"
        )
        os.makedirs(results_dir, exist_ok=True)

        return results_dir
