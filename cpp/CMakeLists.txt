project(ScanContext)

cmake_minimum_required(VERSION 3.19)

set(${CMAKE_EXPORT_COMPILE_COMMANDS} TRUE)

find_package(Eigen3 REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/module/Scancontext)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pybind)
