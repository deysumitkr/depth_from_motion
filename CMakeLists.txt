cmake_minimum_required(VERSION 2.8)
project( Depth_From_Motion )

find_package( OpenCV REQUIRED )
find_package( Boost 1.62 REQUIRED )
find_package( OpenGL REQUIRED )
find_package( GLUT REQUIRED )
find_package( glfw3 3.2 REQUIRED )

#set(OPENCV_ROOT /home/sumit/Downloads/softwares/OpenCV/opencv-2.4.13)
#set(OPENCV_ROOT /home/sumit/Downloads/softwares/OpenCV/opencv-3.1.0)
set(MY_HEADERS pointCloud.h)

#include_directories( ${OPENCV_ROOT}/include )
include_directories( ${OpenCV_INCLUDE_DIR} )
include_directories( ${Boost_INCLUDE_DIR} )
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )


add_executable( depth_from_motion main.cpp pointCloud.cpp container.cpp ${MY_HEADERS})

set_target_properties(
	depth_from_motion 
	PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries( 
	depth_from_motion 
	${OpenCV_LIBS}
	${Boost_LIBRARIES} 
	${OPENGL_LIBRARIES} ${GLUT_LIBRARY}
	glfw
	GLEW
	)
