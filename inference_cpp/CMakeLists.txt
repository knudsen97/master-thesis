cmake_minimum_required(VERSION 3.3)
project(inference_cpp VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 14)
# include pytorch
find_package(Torch REQUIRED)


include(CTest)
enable_testing()


# include OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

# include python
# sudo apt install python3-dev
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Embed NumPy )
include_directories(${PYTHON_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# --------------- Find RobWork ---------------
set(RW_ROOT $ENV{RW_ROOT})
set(RWS_ROOT $ENV{RWS_ROOT})

# If not the -DCMAKE_BUILD_TYPE option is given when running CMake, use Release as default.
if( NOT DEFINED CMAKE_BUILD_TYPE )
    set(CMAKE_BUILD_TYPE Release)
endif()
message("-- Build type: " ${CMAKE_BUILD_TYPE})

# Include default settings for constructing a RobWork dependent project
set(RobWork_DIR ${RW_ROOT}/cmake)
find_package(RobWork REQUIRED)
message("-- RobWork found: " ${RW_ROOT})

# Include default settings for constructing a RobWorkStudio dependent project
set(RobWorkStudio_DIR ${RWS_ROOT}/cmake)
find_package(RobWorkStudio REQUIRED)
message("-- RobWorkStudio found: " ${RWS_ROOT})
# --------------------------------------------



FILE(GLOB_RECURSE SRC_FILES
${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)

add_executable(inference_cpp main.cpp ${SRC_FILES})
target_link_libraries(inference_cpp ${TORCH_LIBRARIES} ${OpenCV_LIBS} Python3::NumPy Python3::Python z expat)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

#cmake with this command

#  /usr/bin/cmake --no-warn-unused-cli -DCMAKE_PREFIX_PATH=/home/claus/pytorch_cpp/libtorch-cxx11-abi-shared-with-deps-latest/libtorch -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++ -S/home/claus/Documents/sdu/9sem/master/code/master-thesis/inference_cpp -B/home/claus/Documents/sdu/9sem/master/code/master-thesis/inference_cpp/build -G Ninja

# or simple with with (unstable)
# cmake -DCMAKE_PREFIX_PATH=/home/claus/pytorch_cpp/libtorch-cxx11-abi-shared-with-deps-latest/libtorch ..
