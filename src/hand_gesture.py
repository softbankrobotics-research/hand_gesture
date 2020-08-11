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

CV2_ROS = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if CV2_ROS in sys.path:
    sys.path.remove(CV2_ROS)
    sys.path.append(CV2_ROS)

import rospy
import argparse
import gym
import pybullet
import pybullet_data
import numpy as np
import time

from geometry_msgs.msg import PoseStamped
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import std_msgs.msg

from stable_baselines import PPO2


class HandGesture:
    """
    Class for performing the hand gesture
    """
    def __init__(self):
        """
        Constructor
        """

        publisher_gesture = rospy.init_node("hand_gesture",
                                            anonymous=True,
                                            disable_signals=False,
                                            log_level=rospy.INFO)

        self.robot_ip = "10.0.160.95"
        self.motion = None
        self.posture = None
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
        self.time_active = 0

        # Limits for the joint angles
        self.kinematic_chain = {
            "HipPitch":       [-0.4,  0.05],  # real limits [-1.0385,  1.0385],
            "RShoulderPitch": [-2.0857,  2.0857],
            "RShoulderRoll":  [-1.5620, -0.0087],
            "RElbowRoll":     [0.0087,  1.56207],
            "HeadYaw":        [-2.0857,  2.0857],
            "HeadPitch":      [-0.7068, 0.6370]
            }

        self.joint_body_names = [
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

        self.joint_body_values = [
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

        self.joint_names = ["HipPitch",
                            "RShoulderPitch",
                            "RShoulderRoll",
                            "RElbowRoll",
                            "HeadYaw",
                            "HeadPitch"]

        try:
            assert self.robot_ip is not None
        except AssertionError:
            self.logFatal("Cannot retreive the robot's IP, the state " +
                          " the module won't be launched")
            return

        # Load the pre-trained model
        self.load_model()

        # Subscribers
        subscriber_target = rospy.Subscriber(
            "hand_gesture/target_local_position",
            PoseStamped,
            self.callback_target)

        subscriber_hand = rospy.Subscriber(
            "hand_gesture/hand_local_position",
            PoseStamped,
            self.callback_hand)

        subscriber_head = rospy.Subscriber(
            "hand_gesture/head_local_position",
            PoseStamped,
            self.callback_head)

        subscriber_head = rospy.Subscriber(
            "joint_states",
            JointState,
            self.callback_joint_states)

        subscriber_active = rospy.Subscriber(
            "hand_gesture/active",
            Bool,
            self.callback_active)

        subscriber_shutdown = rospy.Subscriber(
            "hand_gesture/shutdown",
            Bool,
            self.callback_shutdown)

        self.pub_joints = rospy.Publisher(
             "joint_angles",
             JointAnglesWithSpeed,
             queue_size=1)

    def load_model(self):
        """
        Load a pre-trained PPO2 model on stable_baselines
        """
        #file = "ppo2_hand_determined_borg"
        file = "ppo2_hand_reverent_bose"
        path = "../models/"+file
        self.model = PPO2.load(path)

    def callback_joint_states(self, joint_states):
        """
        Get the angles of the joints of the robot
        """
        self.joint_states = joint_states

    def callback_target(self, target_local_position):
        """
        Get the position of the target in the world reference
        """
        self.target_local_position = target_local_position

    def callback_hand(self, hand_local_position):
        """
        Get the position of the hand of the robot in the world reference
        """
        self.hand_local_position = hand_local_position

    def callback_head(self, head_local_position):
        """
        Get the position of the head of the robot in the world references
        """
        self.head_local_position = head_local_position

    def callback_active(self, active):
        """
        Activates or deactivates the gesture motion
        """
        if active.data is True:
            if self.active is False:
                print("Gesture Module ON")
                self.active = True
                self.time_active = rospy.get_rostime()
        else:
            self.active = False
            joint_angles = JointAnglesWithSpeed()
            joint_angles.joint_names = self.joint_body_names
            joint_angles.joint_angles = self.joint_body_values
            joint_angles.speed = 0.1
            joint_angles.relative = 0
            self.pub_joints.publish(joint_angles)
            print("Gesture module OFF")

    def callback_shutdown(self, data):
        """
        Shutdown the node
        """
        print("Shutting down the gesture module")
        if data.data is True:
            print("Shutdown")
            # do something

    def normalize_with_bounds(self, values, range_min, range_max,
                              bound_min, bound_max):
        """
        Normalizes values (list) according to a specific range
        """
        if isinstance(values, float):
            values = [values]
        values_std = [(x - bound_min) / (bound_max-bound_min)
                      for x in values]
        values_scaled = [x * (range_max - range_min) + range_min
                         for x in values_std]
        return values_scaled

    def get_observation(self):
        """
        Returns the observation

        Returns:
            obs - a list containing lists of normalized observations
            [joint angles] +\
            [norm hand - target] +\
            [unit vec head to hand] +\
            [unit vec head]
        """
        # Get position of the target position in the local reference
        target_pos = np.array([self.target_local_position.pose.position.x,
                               self.target_local_position.pose.position.y,
                               self.target_local_position.pose.position.z])
        # print(target_pos)

        # Get position of the hand  of Pepper  in the local frame
        hand_pose = np.array([self.hand_local_position.pose.position.x,
                              self.hand_local_position.pose.position.y,
                              self.hand_local_position.pose.position.z])

        # Get position and orientation of the hand  of Pepper
        # in the local frame
        head_pose = np.array([self.head_local_position.pose.position.x,
                              self.head_local_position.pose.position.y,
                              self.head_local_position.pose.position.z])

        head_rot = np.array([self.head_local_position.pose.orientation.x,
                             self.head_local_position.pose.orientation.y,
                             self.head_local_position.pose.orientation.z,
                             self.head_local_position.pose.orientation.w])

        # target_pos_bis = target_pos
        # target_pos_bis[1] = target_pos_bis[1] - 0.05
        hands_norm = np.linalg.norm(hand_pose - target_pos)

        # Get information about the head direction and hand
        # Head to Hand reward based on the direction of the head to the hand
        head_hand_norm = np.linalg.norm(np.array(head_pose)
                                        - np.array(target_pos))

        vec_head_hand = np.array(target_pos) - np.array(head_pose)
        unit_vec_head_hand = vec_head_hand / head_hand_norm
        # print(unit_vec_head_hand)

        rot_obj = R.from_quat([head_rot[0],
                               head_rot[1],
                               head_rot[2],
                               head_rot[3]])

        # rot_vec = rot_obj.as_rotvec()
        rot_vec = rot_obj.apply([1, 0, 0])
        #rot_vec_ = np.array([rot_vec[2],rot_vec[0],-rot_vec[1]])
        head_norm = np.linalg.norm(rot_vec)
        unit_vec_head = (rot_vec / head_norm)
        unit_vec_head = np.array([unit_vec_head[0],
                                  unit_vec_head[1],
                                  unit_vec_head[2]])
        # print(unit_vec_head)

        # Compute de normal distance between both orientations
        orientations_norm = np.linalg.norm(
                np.array(unit_vec_head_hand) - unit_vec_head)

        # Fill and return the observation
        hand_poses = [pose for pose in hand_pose]
        norm_hand_poses = self.normalize_with_bounds(hand_poses,
                                                     -1, 1, -1.5, 1.5)

        hand_poses_bis = [pose_bis for pose_bis in target_pos]
        norm_hand_poses_bis = self.normalize_with_bounds(hand_poses_bis,
                                                         -1, 1, -1.5, 1.5)
        #print(hand_poses_bis)
        #print(norm_hand_poses_bis)

        norm_angles = list()
        name_angles = list()
        index_joint = 0
        self.angles = []
        for joint in self.joint_names:
            index_joints = 0
            for joints in self.joint_states.name:
                if joint == joints:
                    if joint == "HipPitch":
                        bound_min = -0.4
                        bound_max = 0.05
                    else:
                        bound_min = self.kinematic_chain[joint][0]
                        bound_max = self.kinematic_chain[joint][1]
                    angle = [self.joint_states.position[index_joints]]
                    self.angles.append(
                        self.joint_states.position[index_joints])
                    norm_angle = self.normalize_with_bounds(angle, -1, 1,
                                                            bound_min,
                                                            bound_max)
                    norm_angles.extend(norm_angle)
                    name_angles.extend([joint])
                    if joint == "X":
                        print(joint)
                        print(bound_min)
                        print(bound_max)
                        print(angle)
                        print(norm_angle)
                index_joints = index_joints + 1
            index_joint = index_joint + 1
        obs = [n for n in norm_hand_poses] +\
              [a for a in norm_angles] +\
              [n for n in norm_hand_poses_bis]+\
              [e for e in unit_vec_head_hand] +\
              [e for e in unit_vec_head]
        return obs

    def set_angles(self):
        """
        Set the speed and angle of the joints in relative position
        and send them to the naoqi driver

        Publisher topic:
            joint_angles
        """

        angles = []
        speeds = []
        speed = []
        i = 0
        # Adjust the speed of the joints
        for action in self.action:
            if action > 0:
                speed = action * 0.1
                angles = action * 0.2
            elif action < 0:
                speed = action * - 0.1
                angles = action * 0.2
            else:
                angles = 0
                speed = 0
            if (self.angles[i] < self.kinematic_chain[self.joint_names[i]][0]
                or self.angles[i] > self.kinematic_chain[self.joint_names[i]][1]):
                print("Limits! of:")
                print(self.joint_names[i])

            # Limit the angle of the HipPitch joint
            if self.joint_names[i] == "HipPitch":
                angles = angles * 0.1
                speed = 0.1
                if(self.angles[i] < -0.35 or self.angles[i] > 0.03):
                    #print("Angles out of limits")
                    angles = angles * -1
            elif self.joint_names[i] == "RElbowRoll":
                angles = angles * 1
            elif (self.joint_names[i] == "HeadYaw" or
                  self.joint_names[i] == "HeadPitch"):
                angles = angles * 0.5
                speed = speed * 0.5

            self.joint_angles.joint_names = [self.joint_names[i]]
            self.joint_angles.joint_angles = [angles]
            self.joint_angles.speed = speed
            self.joint_angles.relative = 1
            self.pub_joints.publish(self.joint_angles)
            i = i + 1

    def start(self):
        """
        Starts the gesture of the robot if it received True on the Active topic
        Subscriber topic
            "hand_gesture/active"
        """

        if self.active:
            obs = self.get_observation()
            self.action, _states = self.model.predict(obs, deterministic=True)
            self.set_angles()


if __name__ == '__main__':
    handGesture = HandGesture()
    try:
        rate = rospy.Rate(50.0)
        while not rospy.is_shutdown():
            handGesture.start()
            rate.sleep()
    except KeyboardInterrupt:
        pass
