cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(pairwise_incremental_registration)

find_package(PCL 1.8 REQUIRED)
find_package(COVIS REQUIRED HINTS $ENV{CoViS_NEW_DIR} )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${COVIS_INCLUDE_DIRS})
add_definitions(${COVIS_DEFINITIONS})

find_package(OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable (${PROJECT_NAME} pairwise_incremental_registration.cpp)
target_link_libraries (${PROJECT_NAME} ${PCL_LIBRARIES} ${COVIS_LIBRARIES} ${OpenCV_LIBS})

# add the install targets
install(TARGETS ${PROJECT_NAME}
 # ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
 # LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION /usr/local/bin
)

