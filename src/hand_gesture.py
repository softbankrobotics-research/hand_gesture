#!/usr/bin/env python3.5
# coding: utf-8

# Copyright 2020 SoftBank Robotics Europe

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Project     : CROWDBOT
# Author      : Arturo Cruz Maya
# Departement : Software

import os
import sys
import argparse
import numpy as np
import gym
import pybullet
import pybullet_data
import rospy
import std_msgs.msg

from geometry_msgs.msg import PoseStamped
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

CV2_ROS = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if CV2_ROS in sys.path:
    sys.path.remove(CV2_ROS)
    sys.path.append(CV2_ROS)
from stable_baselines import PPO2


class HandGesture:
    """
    Class for performing the hand gesture
    """
    DETERMINISTIC = True
    GESTURE_TIME = 20  # in seconds
    HIP_PITCH_MAX = 0.05  # in rads
    HIP_PITCH_MIN = -0.4  # in rads
    MODEL_TARGET_MAX = 1.5  # in meters
    MODEL_TARGET_MIN = -1.5  # in meters
    JOINTS_SPEED = 0.1  # % of maximum speed
    HEAD_SPEED = 0.5  # % of joints speed
    HIP_SPEED = 0.5  # % of joints speed
    JOINTS_ANGLE = 0.04  # rads
    HEAD_ANGLE = 0.5  # % of the joints angles
    HIP_ANGLE = 0.5  # % of the joints angles
    TARGET_MIN_X = 0.30  # in meters
    TARGET_MAX_X = 0.70  # in meters
    TARGET_MIN_Y = -0.70  # in meters
    TARGET_MAX_Y = -0.10  # in meters
    TARGET_MIN_Z = 0.6  # in meters
    TARGET_MAX_Z = 1.4  # in meters
    RESET_POSITION_HEAD = -0.1  # in rads
    RESET_POSITION_SHOULDER = 0.7  # in rads
    NORMALIZATION_MIN = -1  # Lower limit
    NORMALIZATION_MAX = 1  # Upper limit
    TARGET_THRESHOLD = 0.06

    def __init__(self):
        """
        Constructor
        """

        self.path_file = rospy.get_param('hand_gesture/path_file')

        rospy.init_node(
            "hand_gesture",
            anonymous=True,
            disable_signals=False,
            log_level=rospy.INFO)

        self.model = None
        self.joint_states = JointState()
        self.joint_angles = JointAnglesWithSpeed()

        # The position to be reached by the right hand of the robot
        self.target_local_position = PoseStamped()
        # The position of the hand of the robot
        self.hand_local_position = PoseStamped()
        # The position of the head of the robot
        self.head_local_position = PoseStamped()
        self.head_local_position.pose.orientation.w = 1
        # Gesture status
        self.active = False
        self.gesture_started = False
        self.active_time = rospy.Time.now()
        self.shoulder_angle = 0
        self.head_angle = 0
        self.hands_norm = 0
        self.hands_close = False

        # Limits for the joint angles
        self.kinematic_chain = {
            # real hip limits [-1.0385,  1.0385],
            "HipPitch":       [self.HIP_PITCH_MIN,  self.HIP_PITCH_MAX],
            "RShoulderPitch": [-2.0857,  2.0857],
            "RShoulderRoll":  [-1.5620, -0.0087],
            "RElbowRoll":     [0.0087,  1.56207],
            "HeadYaw":        [-2.0857,  2.0857],
            "HeadPitch":      [-0.7068, 0.6370]
            }

        self.joint_names = [
            "HipPitch",
            "RShoulderPitch",
            "RShoulderRoll",
            "RElbowRoll",
            "HeadYaw",
            "HeadPitch"]

        # Load the pre-trained model
        self.load_model()

        self.pub_init_targets = rospy.Publisher(
             "hand_gesture/init_targets",
             Bool,
             queue_size=1)

        self.pub_joints = rospy.Publisher(
             "joint_angles",
             JointAnglesWithSpeed,
             queue_size=1)

        # Subscribers
        rospy.Subscriber(
            "hand_gesture/target_local_position",
            PoseStamped,
            self.callback_target)

        rospy.Subscriber(
            "hand_gesture/hand_local_position",
            PoseStamped,
            self.callback_hand)

        rospy.Subscriber(
            "hand_gesture/head_local_position",
            PoseStamped,
            self.callback_head)

        rospy.Subscriber(
            "joint_states",
            JointState,
            self.callback_joint_states)

        rospy.Subscriber(
            "hand_gesture/active",
            Bool,
            self.callback_active)

    def callback_joint_states(self, joint_states):
        """
        Update the angles of the joints of the robot
        Parameters:
            joint_states type sensor_msgs/JointState
        Subscriber Topic:
            joint_states
        """
        self.joint_states = joint_states

    def callback_target(self, target_local_position):
        """
        Update the position of the target in the world reference

        Parameters:
            target_local_position type geometry_msgs/PoseStamped
        Subscriber Topic:
            hand_gesture/target_local_position
        """
        self.target_local_position = target_local_position

    def callback_hand(self, hand_local_position):
        """
        Update the position of the hand of the robot in the world reference

        Parameters:
            hand_local_position type geometry_msgs/PoseStamped
        Subscriber Topic:
           hand_gesture/hand_local_position
        """
        self.hand_local_position = hand_local_position

    def callback_head(self, head_local_position):
        """
        Update the position of the head of the robot in the world references

        Parameters:
            head_local_position type geometry_msgs/PoseStamped
        Subscriber Topic:
           hand_gesture/head_local_position
        """
        self.head_local_position = head_local_position

    def callback_active(self, active):
        """
        Activates or deactivates the gesture motion

        Parameters:
            active type std_msgs/Bool
        Subscriber Topic:
           hand_gesture/active
        """
        if active.data is True:
            self.resetGesture()
            self.set_head_pose()
            rospy.sleep(1.)
            if self.active is False:
                rospy.loginfo("Gesture Module ON")
                self.active = True
                self.active_time = rospy.Time.now()
        else:
            self.resetGesture()

    def resetGesture(self):
        """
        Reset the gesture to its initial state
        """
        self.active = False
        self.hands_norm = 0
        self.hands_close = False
        self.gesture_started = False
        self.target_local_position.pose.position.x = 1
        self.target_local_position.pose.position.y = 0
        self.target_local_position.pose.position.z = 0
        self.pub_init_targets.publish(True)
        self.set_initial_pose()
        rospy.loginfo("Gesture Module OFF")

    def load_model(self):
        """
        Load a pre-trained PPO2 model on stable_baselines
        """
        self.model = PPO2.load(self.path_file)

    def set_initial_pose(self):
        """
        Set the position of the robot to the standart initial position

        Publisher Topic:
            joint_angles
        """
        body_joint_names = [
            "HeadPitch",
            "HeadYaw",
            "HipPitch",
            "HipRoll",
            "KneePitch",
            "LElbowRoll",
            "LElbowYaw",
            "LHand",
            "LShoulderPitch",
            "LShoulderRoll",
            "LWristYaw",
            "RElbowRoll",
            "RElbowYaw",
            "RHand",
            "RShoulderPitch",
            "RShoulderRoll",
            "RWristYaw"]

        body_joint_values = [
            -0.21168947219848633,
            -0.007669925689697266,
            -0.026077747344970703,
            -0.004601955413818359,
            0,
            -0.5184855461120605,
            -1.2179806232452393,
            0.5896309614181519,
            1.5800002813339233,
            0.11658263206481934,
            -0.03072190284729004,
            0.5184855461120605,
            1.225650668144226,
            0.5887521505355835,
            1.5800001621246338,
            -0.11504864692687988,
            0.027570009231567383]

        joint_angles = JointAnglesWithSpeed()
        joint_angles.joint_names = body_joint_names
        joint_angles.joint_angles = body_joint_values
        joint_angles.speed = self.JOINTS_SPEED
        joint_angles.relative = 0
        self.pub_joints.publish(joint_angles)

    def set_head_pose(self):
        """
        Set the head pose of the robot to gaze down right to start
        detecting the hand of the person

        Publisher Topic:
            joint_angles
        """
        joint_angles = JointAnglesWithSpeed()
        joint_angles.joint_names = ["HeadYaw", "HeadPitch"]
        joint_angles.joint_angles = [-0.4, 0.5]
        joint_angles.speed = self.JOINTS_SPEED
        joint_angles.relative = 0
        self.pub_joints.publish(joint_angles)

    def normalize_with_bounds(self, values, range_min, range_max,
                              bound_low, bound_high):
        """
        Normalizes values (list) according to a specific range

        Parameters:
            values  the values to be normalized
            range_min  the minimum value of the normalization
            range_max  the maximum value of the normalization
            bound_low  the low bound of the values to be normalized
            bound_high  the high bound of the values to be normalized
        return values_scaled - a list contanining values between
            range_min and range_max normalized from
            bound_min and bound_max
        """
        if isinstance(values, float):
            values = [values]
        values_std = [(x - bound_low) / (bound_high-bound_low)
                      for x in values]
        values_scaled = [x * (range_max - range_min) + range_min
                         for x in values_std]
        return values_scaled

    def get_observation(self):
        """
        Get the observation of the state

        Returns:
            obs - a list containing lists of normalized observations
            [[joint angles]  joint angles of the kinematic chain
            [hand 3D pos]  3D position of the hand of the robot (r_gripper)
            [target 3D pos]  3D position of the target (hand of the person)
            [unit vec head to hand]  Orientation vector of the robot head
                towards the target
            [unit vec head]]  Orientation vector of the head of the robot
        """
        # Get position of the target position in the local reference
        target = np.array([
            self.target_local_position.pose.position.x,
            self.target_local_position.pose.position.y,
            self.target_local_position.pose.position.z])

        # Get position of the hand  of Pepper  in the local reference
        hand_pos = np.array([
            self.hand_local_position.pose.position.x,
            self.hand_local_position.pose.position.y,
            self.hand_local_position.pose.position.z])

        # Get position and orientation of the hand  of Pepper
        # in the local reference
        head_pos = np.array([
            self.head_local_position.pose.position.x,
            self.head_local_position.pose.position.y,
            self.head_local_position.pose.position.z])

        head_rot = np.array([
            self.head_local_position.pose.orientation.x,
            self.head_local_position.pose.orientation.y,
            self.head_local_position.pose.orientation.z,
            self.head_local_position.pose.orientation.w])

        hands_norm = np.linalg.norm(hand_pos - target)
        self.hands_norm = hands_norm

        # Get information about the head direction and hand
        # Head to Hand reward based on the direction of the head to the hand
        head_target_norm = np.linalg.norm(
            np.array(head_pos) - np.array(target))

        head_target_vec = np.array(target) - np.array(head_pos)
        head_target_unit_vec = head_target_vec / head_target_norm

        rot_obj = R.from_quat(head_rot)
        head_rot_obj = rot_obj.apply([1, 0, 0])
        head_norm = np.linalg.norm(head_rot_obj)
        head_unit_vec = np.array((head_rot_obj / head_norm))

        # Compute de normal distance between both orientations
        orientations_norm = np.linalg.norm(
            np.array(head_target_unit_vec) - head_unit_vec)

        # Normalize the position of the robot hand
        norm_hand_pos = self.normalize_with_bounds(
            hand_pos,
            self.NORMALIZATION_MIN,
            self.NORMALIZATION_MAX,
            self.MODEL_TARGET_MIN,
            self.MODEL_TARGET_MAX)

        # Normalize the position of the target
        norm_target = self.normalize_with_bounds(
            target,
            self.NORMALIZATION_MIN,
            self.NORMALIZATION_MAX,
            self.MODEL_TARGET_MIN,
            self.MODEL_TARGET_MAX)

        # Normalize the angles of the joints in the kinematic chain
        norm_angles = list()
        index_joint = 0
        self.angles = []
        for joint in self.joint_names:
            index_joints = 0
            for joints in self.joint_states.name:
                if joint == joints:
                    angle = [self.joint_states.position[index_joints]]
                    self.angles.append(
                        self.joint_states.position[index_joints])
                    if joint == "RShoulderPitch":
                        self.shoulder_angle = angle[0]
                    if joint == "HeadYaw":
                        self.head_angle = angle[0]
                    if joint == "HipPitch":
                        bound_min = self.HIP_PITCH_MIN
                        bound_max = self.HIP_PITCH_MAX
                    else:
                        bound_min = self.kinematic_chain[joint][0]
                        bound_max = self.kinematic_chain[joint][1]
                    norm_angle = self.normalize_with_bounds(
                        angle,
                        self.NORMALIZATION_MIN,
                        self.NORMALIZATION_MAX,
                        bound_min,
                        bound_max)
                    norm_angles.extend(norm_angle)

                index_joints = index_joints + 1
            index_joint = index_joint + 1

        obs = [e for e in norm_angles] +\
              [e for e in norm_hand_pos] +\
              [e for e in norm_target] +\
              [e for e in head_target_unit_vec] +\
              [e for e in head_unit_vec]

        return obs

    def set_angles(self, actions):
        """
        Set the speed and angle of the joints in relative position
        and send them to the naoqi driver

        Parameters:
            actions  the values returned by the pre-trained model
                given as a list of float values
        Publisher topic:
            /joint_angles
        """

        angles = 0
        speed = 0
        i = 0
        # Adjust the speed of the joints
        for action in actions:
            angles = action * self.JOINTS_ANGLE
            speed = abs(action)
            # Limit the angle of the HipPitch joint
            if self.joint_names[i] == "HipPitch":
                speed = speed * self.HIP_SPEED
                angles = angles * self.HIP_ANGLE
                '# Back tilt not allowed'
                if angles > 0:
                    angles = 0
                # Angles out of limits
                # force them to be in
                if (self.angles[i] < self.HIP_PITCH_MIN
                   or self.angles[i] > self.HIP_PITCH_MAX):
                    angles = angles * -1
            # Limit the angles of the Head joints
            elif (self.joint_names[i] == "HeadYaw" or
                  self.joint_names[i] == "HeadPitch"):
                speed = speed * self.HEAD_SPEED
                angles = angles * self.HEAD_ANGLE

            self.joint_angles.joint_names = [self.joint_names[i]]
            self.joint_angles.joint_angles = [angles]
            self.joint_angles.speed = speed
            self.joint_angles.relative = 1
            self.pub_joints.publish(self.joint_angles)
            i = i + 1

    def start(self):
        """
        Starts the gesture of the robot if it received True on the Active topic
        """

        # Get the observations
        obs = self.get_observation()

        # Get the position of the target if the gesture is active
        if self.active:
            time_now = rospy.Time.now()
            time_difference = float(time_now.secs) - self.active_time.secs

            # Check if the hand is close enough to the target
            # or if the time surpass the limit
            if ((self.hands_norm < self.TARGET_THRESHOLD)
               or time_difference > self.GESTURE_TIME):
                # if time_difference > self.GESTURE_TIME:
                self.hands_close = True

            if self.hands_close is False:
                target = np.zeros(3)
                target[0] = self.target_local_position.pose.position.x
                target[1] = self.target_local_position.pose.position.y
                target[2] = self.target_local_position.pose.position.z
                # Get the actions predicted with the model and set the angles
                # if the target is inside the bounds
                if (target[0] > self.TARGET_MIN_X
                   and target[0] < self.TARGET_MAX_X
                   and target[1] > self.TARGET_MIN_Y
                   and target[1] < self.TARGET_MAX_Y
                   and target[2] > self.TARGET_MIN_Z
                   and target[2] < self.TARGET_MAX_Z):
                    actions, _states = self.model.predict(
                        obs,
                        deterministic=self.DETERMINISTIC)
                    self.set_angles(actions)
                    # Re-initialize the time to perform the gesture
                    if self.gesture_started is False:
                        self.active_time = rospy.Time.now()
                        self.gesture_started = True
            else:
                # Set the initial pose if the shoulder or the head angles
                # surpass default limits
                if (self.shoulder_angle < self.RESET_POSITION_SHOULDER
                   or self.head_angle < self.RESET_POSITION_HEAD):
                    self.set_initial_pose()
                    rospy.loginfo("Trying to reach initial position")
                    rospy.sleep(1.0)
                # Reset the gesture
                else:
                    self.resetGesture()



if __name__ == '__main__':
    handGesture = HandGesture()
    try:
        rate = rospy.Rate(50.0)
        while not rospy.is_shutdown():
            time_before = rospy.Time.now()
            handGesture.start()
            time_after = rospy.Time.now()
            time_difference = (time_after.nsecs - time_before.nsecs)/100000
            # print(time_difference)
            rate.sleep()
    except KeyboardInterrupt:
        pass
