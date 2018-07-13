cmake_minimum_required(VERSION 2.8)
project( FastDetect )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( FastDetect FastDetect.cc )
target_link_libraries( FastDetect ${OpenCV_LIBS} )

