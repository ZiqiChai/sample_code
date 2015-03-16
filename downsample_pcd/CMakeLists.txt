cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(downsample_pcd)

find_package(PCL 1.6 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (downsample_pcd downsample_pcd.cpp)
target_link_libraries (downsample_pcd ${PCL_LIBRARIES})
