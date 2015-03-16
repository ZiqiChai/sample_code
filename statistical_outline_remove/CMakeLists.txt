cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(remove_plane)

find_package(PCL 1.6 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (statistical_outlier_remove statistical_outlier_remove.cpp)
target_link_libraries (statistical_outlier_remove ${PCL_LIBRARIES})
