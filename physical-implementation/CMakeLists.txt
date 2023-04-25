cmake_minimum_required(VERSION 3.18)

project(RobWork)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Find Open3D and OpenCV and librealsense2
find_package( Open3D REQUIRED )
find_package( OpenCV REQUIRED )
# find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR} REQUIRED)

# If not the -DCMAKE_BUILD_TYPE option is given when running CMake, use Release as default.
if( NOT DEFINED CMAKE_BUILD_TYPE )
    set(CMAKE_BUILD_TYPE Release)
endif()
message("-- Build type: " ${CMAKE_BUILD_TYPE})

# Find RobWork
set(RW_ROOT $ENV{RW_ROOT})
set(RobWork_DIR ${RW_ROOT}/cmake)
find_package(RobWork REQUIRED)
message("-- RobWork found: " ${RW_ROOT})

# Find UR library
find_package(ur_client_library REQUIRED)


# Include all local header and source files
file(GLOB SOURCES
    src/CameraCalibration.cpp
    inc/CameraCalibration.hpp

    src/Sensor.cpp
    inc/Sensor.hpp

    src/Inference.cpp
    inc/Inference.hpp

    src/PredictionProcessor.cpp
    inc/PredictionProcessor.hpp

    src/main.cpp
)

file(GLOB CALIB_SOURCES
    src/CameraCalibration.cpp
    inc/CameraCalibration.hpp

    src/Sensor.cpp
    inc/Sensor.hpp

    src/calibration.cpp
) 

# And now we add any targets that we want
add_executable(main ${SOURCES})
add_executable(calibration ${CALIB_SOURCES})


# if you have additional libraries or include dirs then add them here
set(MAIN_LIBRARIES
  Open3D::Open3D
  ${OpenCV_LIBS} 
  ${ROBWORK_LIBRARIES}
  # ${REALSENSE2_FOUND} 
  ur_client_library::urcl
  )

set(MAIN_INCLUDE_DIRS
  ${OpenCV_INCLUDE_DIRS}
  ${ROBWORK_INCLUDE_DIRS} 
)

set(CALIB_LIBRARIES
  Open3D::Open3D
  ${OpenCV_LIBS} 
  # ${ROBWORK_LIBRARIES}
  )

set(CALIB_INCLUDE_DIRS
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(main PRIVATE ${MAIN_LIBRARIES})
target_include_directories(main PRIVATE ${MAIN_INCLUDE_DIRS})

target_link_libraries(calibration PRIVATE ${CALIB_LIBRARIES})
target_include_directories(calibration PRIVATE ${CALIB_INCLUDE_DIRS})
link_directories()

