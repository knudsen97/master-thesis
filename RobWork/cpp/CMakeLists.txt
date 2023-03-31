cmake_minimum_required(VERSION 3.18)

project(RobWork)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Find Open3D and OpenCV
find_package( Open3D REQUIRED )
find_package( OpenCV REQUIRED )

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

# Include all local header and source files
file(GLOB SOURCES
    src/*.cpp
    inc/*.hpp
    inc/*.h
)

# And now we add any targets that we want
add_executable(main ${SOURCES})

# if you have additional libraries or include dirs then add them here
set(USER_LIBRARIES
  Open3D::Open3D
  ${OpenCV_LIBS} 
  ${ROBWORK_LIBRARIES}
  RWS::sdurws_robworkstudioapp
  # z 
  # expat
  )

set(USER_INCLUDE_DIRS
  ${OpenCV_INCLUDE_DIRS}
  ${ROBWORK_INCLUDE_DIRS} 
)

target_link_libraries(main PRIVATE ${USER_LIBRARIES})
target_include_directories(main PRIVATE ${USER_INCLUDE_DIRS})
link_directories()

