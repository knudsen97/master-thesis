cmake_minimum_required(VERSION 3.18)

project(RobWork)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Find Open3D and OpenCV and librealsense2
find_package( Open3D REQUIRED )
find_package( OpenCV REQUIRED )
find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR} REQUIRED)

# If not the -DCMAKE_BUILD_TYPE option is given when running CMake, use Release as default.
if( NOT DEFINED CMAKE_BUILD_TYPE )
    set(CMAKE_BUILD_TYPE Release)
endif()
message("-- Build type: " ${CMAKE_BUILD_TYPE})



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
  ${REALSENSE2_FOUND}
  )

set(USER_INCLUDE_DIRS
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(main PRIVATE ${USER_LIBRARIES})
target_include_directories(main PRIVATE ${USER_INCLUDE_DIRS})
link_directories()

