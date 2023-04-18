cmake_minimum_required(VERSION 3.10)

project(processing)
set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")


# If not the -DCMAKE_BUILD_TYPE option is given when running CMake, use Release as default.
if( NOT DEFINED CMAKE_BUILD_TYPE )
    set(CMAKE_BUILD_TYPE Debug)
endif()
message("-- Build type: " ${CMAKE_BUILD_TYPE})


# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(/usr/local/include/opencv4)
include_directories(${OpenCV_INCLUDE_DIRS})


file(GLOB SOURCES
    inc/*.h
    inc/*.hpp
    src/*.cpp
)


add_executable(main ${SOURCES})
target_link_libraries(main PUBLIC
                        ${OpenCV_LIBS} 
)
