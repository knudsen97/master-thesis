cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(image_based)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

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

# And now we add any targets that we want
add_executable(SampleTest main.cpp)
target_link_libraries(SampleTest PRIVATE ${ROBWORK_LIBRARIES})
target_include_directories(SampleTest PRIVATE ${ROBWORK_INCLUDE_DIRS} )

# if you have additional libraries or include dirs then add them here
set(USER_LIBRARIES RWS::sdurws_robworkstudioapp)
target_link_libraries(SampleTest PRIVATE ${USER_LIBRARIES})
target_include_directories(SampleTest PRIVATE ${USER_LIBRARIES})
link_directories( )