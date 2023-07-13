add_library(scan_context STATIC Scancontext.cpp)
target_include_directories(scan_context PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(scan_context PUBLIC Eigen3::Eigen)
target_compile_options(scan_context PUBLIC -fPIC)