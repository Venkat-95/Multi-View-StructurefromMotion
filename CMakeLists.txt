cmake_minimum_required(VERSION 3.10)

project(TestProgram)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


get_filename_component(OPENCV_LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../opencv/build" ABSOLUTE)

include_directories(${OPENCV_LIBS_DIR}/install/include/opencv4)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

link_directories(${OPENCV_LIBS_DIR}/install/lib)

add_executable(SfM src/StructurefromMotion.cpp)
target_link_libraries(SfM opencv_core opencv_highgui opencv_imgproc opencv_calib3d)


set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../opencv/build/")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(SfM ${OpenCV_LIBS})


