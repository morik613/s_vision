cmake_minimum_required(VERSION 2.8)

add_definitions(-Wall)
add_definitions(-std=c++11)

find_package(OpenCV REQUIRED)

add_executable(calib 
	src/calib.cpp
	src/FlyCap2CV.cpp
	)

add_executable(stereo 
	src/stereo.cpp
	src/FlyCap2CV.cpp
	)

include_directories(
  include
	${OpenCV_INCLUDE_DIRS}
  /usr/include
  )

target_link_libraries(calib
	${OpenCV_LIBRARIES}
  flycapture
  )

target_link_libraries(stereo
	${OpenCV_LIBRARIES}
  flycapture
)
