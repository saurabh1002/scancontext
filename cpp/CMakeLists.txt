include(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/find_dependencies.cmake)

add_library(scan_context STATIC ScanContext.cpp)
target_include_directories(scan_context PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(scan_context PUBLIC Eigen3::Eigen)
target_compile_options(scan_context PUBLIC -fPIC)
