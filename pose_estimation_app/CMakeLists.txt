cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(pose_estimation_app)

find_package(PCL 1.8 REQUIRED HINTS /usr/local/lib )
find_package(COVIS REQUIRED HINTS $ENV{CoViS_NEW_DIR} )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${COVIS_INCLUDE_DIRS})
add_definitions(${COVIS_DEFINITIONS})

#find_package(Boost 1.48 REQUIRED)

find_package(OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable (${PROJECT_NAME} pose_estimation_app.cpp)
target_link_libraries (${PROJECT_NAME} ${PCL_LIBRARIES} ${COVIS_LIBRARIES} ${OpenCV_LIBS})


# add the install targets
install(TARGETS ${PROJECT_NAME}
 # ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
 # LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION /usr/local/bin
)
