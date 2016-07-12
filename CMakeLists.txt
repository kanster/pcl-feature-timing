cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)

project(pcl_feature_comparison)

# find point cloud library
find_package(PCL 1.8.0 REQUIRED)
if(PCL_FOUND)
    message(STATUS "PCL INCLUDE DIRS ${PCL_INCLUDE_DIRS}")
    message(STATUS "PCL LIBS REQUIRED ${PCL_LIBRARIES}")
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
endif()

find_package(OpenCV 2.4 REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV INCLUDE DIRS ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV LIBS REQUIRED ${OpenCV_LIBS}")
endif()

add_executable(features_comparison features.cpp)
target_link_libraries(features_comparison ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${CMAKE_SOURCE_DIR}/libs/libbrand.a )

add_executable(keypoints_comparison keypoints.cpp)
target_link_libraries(keypoints_comparison ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
