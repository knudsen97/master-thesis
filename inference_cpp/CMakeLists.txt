cmake_minimum_required(VERSION 3.0.0)
project(inference_cpp VERSION 0.1.0)

include(CTest)
enable_testing()

# include OpenCV 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

# include python 
# guide to install python lib
# https://www.linuxcapable.com/install-python-3-8-on-ubuntu-linux/?utm_content=cmp-true
# the important part is `sudo apt install python3.8-dev`
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

FILE(GLOB_RECURSE SRC_FILES 
${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)

add_executable(inference_cpp main.cpp ${SRC_FILES})
target_link_libraries(inference_cpp ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
