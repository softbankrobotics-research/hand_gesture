# hand gesture
Ros Module for a Target reaching behavior with the Pepper robot

## Installation

This module require the following libraries:
- **numpy**
- **scipy**
- **stable_baselines**
  
Also it requires the following ros packages:
- rospy
- std_msgs
- geometry_msgs
- naoqi_bridge_msgs
- sensor_msgs
- std_msgs
- tf2_ros
- naoqi_driver*
  
*Independent package to connect with the robot
  
Download this repository and use catkin_make on your ros working directory

## Usage

Start the module using roslaunch:
```
roslaunch hand_gesture hand_gesture 
```

This module expects a target specified as a `PoseStamped` msg comming from the topic:
```
hand_detection/target_camera
```

## License
Licensed under the [Apache-2.0 License](https://github.com/arturocuma/hand_gesture/blob/develop/LICENSE)

