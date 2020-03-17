cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(octdata4python)

FIND_PACKAGE(PythonInterp 3)

find_package(PythonInterp 3)
find_package(PythonLibs 3 REQUIRED)
find_package(Boost COMPONENTS system python3 numpy3 REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")

set(CMAKE_CXX_STANDARD 11)

find_package(LibOctData REQUIRED)
find_package(OpenCV REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wunreachable-code -Wconversion")
# 	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
endif()

PYTHON_ADD_MODULE(octdata4python octdata4python.cpp)

target_link_libraries(octdata4python LibOctData::octdata Boost::system Boost::python3 Boost::numpy3 ${OpenCV_LIBS} ${PYTHON_LIBRARIES})
include_directories(${LibOctData_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})


