cmake_minimum_required(VERSION 2.8)

project(PLuTARC_centerfind2D)

# The c++11 cmake thing never works
set(CMAKE_CXX_FLAGS "-std=c++11")

set(OCV_DIR $ENV{SDK}/opencv)
set(FREEIMG_DIR $ENV{SDK}/FreeImage)
set(HDF5_DIR C:/Program\ Files\ (x86)/HDF_Group/HDF5/1.8.15/include)

file(GLOB SOURCES source/*.cpp)
file(GLOB HEADERS include/*.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
					${OCV_DIR}/include
					${FREEIMG_DIR}/include
					${HDF5_DIR}/include)
					
add_executable(PLuTARC_centerfind2D ${SOURCES} ${HEADERS})
target_link_libraries(PLuTARC_centerfind2D LINK_PUBLIC 
${OCV_DIR}/x86/staticlib/opencv_core300d
${OCV_DIR}/x86/staticlib/opencv_imgproc300d
${OCV_DIR}/x86/staticlib/opencv_highgui300d
${FREEIMG_DIR}/lib/Win32/FreeImage
${HDF5_DIR}/lib/hdf5)