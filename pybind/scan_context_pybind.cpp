// MIT License
//
// Copyright (c) 2023 Saurabh Gupta, Tiziano Guadagnino, Cyrill Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "ScanContext.hpp"
#include "stl_vector_eigen.h"

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(scan_context_pybind, m) {
    auto vector3dvector = pybind_eigen_vector_of_vector<Eigen::Vector3d>(
        m, "_VectorEigen3d", "std::vector<Eigen::Vector3d>",
        py::py_array_to_vectors_double<Eigen::Vector3d>);

    py::class_<SCManager, std::shared_ptr<SCManager>> scan_context(
        m, "_SCManager",
        "This is the low level C++ bindings, all the methods and "
        "constructor defined within this module (starting with a "
        "``_`` "
        "should not be used. Please reffer to the python Procesor "
        "class to "
        "check how to use the API");
    scan_context.def(py::init<>())
        .def("_makeAndSaveScancontextAndKeys", &SCManager::makeAndSaveScancontextAndKeys,
             "_scan_down"_a)
        .def("_detectLoopClosureID", &SCManager::detectLoopClosureID)
        .def(
            "_getScanContext",
            [](const SCManager &self, int idx) { return self.polarcontexts_[idx]; }, "idx"_a);
}
