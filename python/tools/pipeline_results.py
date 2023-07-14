# MIT License
#
# Copyright (c) 2023 Saurabh Gupta, Tiziano Guadagnino, Cyrill Stachniss.
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
from typing import Dict, Set, Tuple

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table


class Metrics:
    def __init__(self, true_positives, false_positives, false_negatives):
        self.tp = true_positives
        self.fp = false_positives
        self.fn = false_negatives

        self.precision = self.tp / (self.tp + self.fp)
        self.recall = self.tp / (self.tp + self.fn)
        self.F1 = 2 * self.precision * self.recall / (self.precision + self.recall)


class PipelineResults:
    def __init__(self, gt_closures: np.ndarray, dataset_name: str, scan_context_thresholds) -> None:
        self._dataset_name = dataset_name
        self._scan_context_thresholds = scan_context_thresholds

        self.predicted_closures: Dict[float, Set[Tuple[int]]] = {}
        for threshold in self._scan_context_thresholds:
            self.predicted_closures[threshold] = set()

        self.metrics: Dict[float, Metrics] = {}

        gt_closures = gt_closures if gt_closures.shape[1] == 2 else gt_closures.T
        self.gt_closures: Set[Tuple[int]] = set(map(lambda x: tuple(sorted(x)), gt_closures))

    def print(self) -> None:
        if self.metrics:
            self.log_to_console()

    def append(self, query_idx: int, nn_idx: int, dist: float) -> None:
        indices = np.where(dist < self._scan_context_thresholds)[0]
        for index in indices:
            self.predicted_closures[self._scan_context_thresholds[index]].add((nn_idx, query_idx))

    def compute_metrics(
        self,
    ) -> None:
        for key in self._scan_context_thresholds:
            closures = self.predicted_closures[key]
            closures = set(map(lambda x: tuple(sorted(x)), closures))
            tp = len(self.gt_closures.intersection(closures))
            fp = len(closures) - tp
            fn = len(self.gt_closures) - tp
            self.metrics[key] = Metrics(tp, fp, fn)

    def _rich_table_pr(self, table_format: box.Box = box.HORIZONTALS) -> Table:
        table = Table(box=table_format, title=self._dataset_name)
        table.caption = f"Loop Closure Distance Threshold:"
        table.add_column("Scan Context Threshold", justify="center", style="cyan")
        table.add_column("True Positives", justify="center", style="magenta")
        table.add_column("False Positives", justify="center", style="magenta")
        table.add_column("False Negatives", justify="center", style="magenta")
        table.add_column("Precision", justify="left", style="green")
        table.add_column("Recall", justify="left", style="green")
        table.add_column("F1 score", justify="left", style="green")
        for [threshold, metric] in self.metrics.items():
            table.add_row(
                f"{threshold}",
                f"{metric.tp}",
                f"{metric.fp}",
                f"{metric.fn}",
                f"{metric.precision}",
                f"{metric.recall}",
                f"{metric.F1}",
            )
        return table

    def log_to_console(self) -> None:
        console = Console()
        console.print(self._rich_table_pr())

    def log_to_file_pr(self, filename) -> None:
        with open(filename, "wt") as logfile:
            console = Console(file=logfile, width=100, force_jupyter=False)
            console.print(self._rich_table_pr(table_format=box.ASCII_DOUBLE_HEAD))

    def log_to_file_closures(self, result_dir) -> None:
        np.save(
            os.path.join(result_dir, f"predicted_closures.npy"),
            self.predicted_closures,
        )
