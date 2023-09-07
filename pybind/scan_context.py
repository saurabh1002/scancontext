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

from typing import Tuple

import numpy as np

from . import scan_context_pybind


class ScanContext:
    def __init__(self) -> None:
        self._pipeline = scan_context_pybind._SCManager()

    def process_new_scan(self, scan: np.ndarray) -> None:
        scan = scan_context_pybind._VectorEigen3d(scan)
        self._pipeline._makeAndSaveScancontextAndKeys(scan)

    def check_for_closure(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        query_node_idx, candidate_ids, candidate_dists, candidate_yaws = self._pipeline._detectLoopClosureID()
        return query_node_idx, np.asarray(candidate_ids, int), np.asarray(candidate_dists), np.asarray(candidate_yaws)

    def get_scan_context(self, idx: int) -> np.ndarray:
        scan_context = self._pipeline._getScanContext(idx)
        return np.asarray(scan_context)
