cmake_minimum_required(VERSION 2.8.3)
project(hand_gesture)

message("CMAKE " ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})

IF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.1)
    set(CMAKE_CXX_STANDARD 11)
ELSE (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.1)
    add_definitions("-std=c++11")
ENDIF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.1)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  naoqi_bridge_msgs
  roslaunch
)

roslaunch_add_file_check(launch)

#Generate messages in the 'msg' folder
 add_message_files(
   FILES
   Active.msg
   Shutdown.msg
   Target.msg
 )

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
 LIBRARIES
    ${PROJECT_NAME}

 CATKIN_DEPENDS
    rospy
    std_msgs
    naoqi_bridge_msgs
)

###########
## Build ##
###########
