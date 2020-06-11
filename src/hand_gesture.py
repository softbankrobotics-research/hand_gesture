#!/usr/bin/env python
# coding: utf-8

# Copyright 2020 SoftBank Robotics

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

import qi
import os
import sys
import rospy
import argparse
import gym
import pybullet
import pybullet_data
import numpy as np
import tf2

from geometry_msgs.msg import PoseStamped
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed

from gym import spaces
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

from scipy.spatial.transform import Rotation as R

class HandGesture:
	"""
    Class for performing the hand gesture
    """
    def __init__(self):
        """
        Constructor
        """
        self.robot_ip = None
        self.motion = None
        self.posture = None
        self.vec_env = None

        #The position to be reached by the right hand of the robot
        self.target_position = None

        #Gesture status
        self.active_gesture = False

        self.r_kinematic_chain = [
            "HipPitch",
            "RShoulderPitch",
            "RShoulderRoll",
            "RElbowRoll",
            "HeadYaw",
            "HeadPitch"
            ]

        if not self.virtual_env:
            self.logInfo("Real world hand gesture mode selected")
            try:
                assert self.robot_ip is not None
                self.session = qi.Session()
                self.session.connect("tcp://" + self.robot_ip + ":9559")
                self.motion = self.session.service("ALMotion")
                self.posture = self.session.service("ALRobotPosture")

            except AssertionError:
                self.logFatal("Cannot retreive the robot's IP, the state " +\
                    " the module won't be launched")
                return
        else:
            self.logInfo("Virtual hand gesture mode selected")

        self.listener = tf2_ros.TransformListener()

        publisher_gesture = rospy.init_node("hand_gesture",
         	anonymous=True,
            disable_signals=False,
            log_level=rospy.INFO)

		subscriber_target = rospy.Subscriber("hand_detection/target_position",
         	PoseStamped,
            self.callback)

    def loadModel(self):
        path = "models/ppo2_hand_determined_borg"
        model = PPO2.load(path)

    def callback(self, target_position)
        print("callback") 
        obs = self.getObservation(target_position)     
        while self.active_gesture:
        	useSensors  = True
            sensorAngles = motion.getAngles(self.r_kinematic_chain, useSensors)
            action, _states = model.predict(obs)
            self._setVelocities(self.r_kinematic_chain, action) 

    def setVelocities(self, angles, normalized_velocities):
        """
        Sets velocities on the robot joints
        """
        for angle, velocity in zip(angles, normalized_velocities):
            # Unnormalize the velocity

            if angle == "HeadPitch" or angle == "HeadYaw":
                velocity *= 0.7
            else:
                velocity *= 1.0
                #velocity *= self.pepper.joint_dict[angle].getMaxVelocity() *0.25


        self.motion.changeAngles(angles, normalized_velocities, velocity)

    def getObservation(self, target_position):
        """
        Returns the observation

        Returns:
            obs - the list containing the observations
        """
        # Get position of the target position in the base reference
        try: 
            (target,rot_target) = listener.lookupTransform(
        	"base_link",
        	target_position,
        	rospy.Time(0))
        except (
        	tf2_ros.LookupException,
        	tf2_ros.ConnectivityException,
        	tf2_ros.ExtrapolationException):
            pass
        target_pos = np.array(target)

        # Get position of the hand and head of Pepper  in the base frame
        try: 
            (r_hand,rot_r_hand) = listener.lookupTransform(
        	"base_link",
        	"r_gripper", #TODO Modify
        	rospy.Time(0))
        except (
        	tf2_ros.LookupException,
        	tf2_ros.ConnectivityException,
        	tf2_ros.ExtrapolationException):
            pass
        hand_pose = np.array(r_hand)

        try: 
            (head,rot_head) = listener.lookupTransform(
        	"base_link",
        	"Head", #TODO Modify
        	rospy.Time(0))
        except (
        	tf2_ros.LookupException,
        	tf2_ros.ConnectivityException,
        	tf2_ros.ExtrapolationException):
            pass
        head_pose = np.array(r_hand)
        
        #target_pos_bis = target_pos
        #target_pos_bis[1] = target_pos_bis[1] - 0.05

        hands_norm = np.linalg.norm(hand_pose - target_pos)

        # Get information about the head direction and hand
        # Head to Hand reward based on the direction of the head to the hand
        head_hand_norm =  np.linalg.norm(np.array(head_pose) - np.array(target_pos))
        vec_head_hand = np.array(target_pos) - np.array(head_pose)
        unit_vec_head_hand = vec_head_hand / head_hand_norm


        rot_obj =  R.from_quat([head_rot[2],head_rot[1],head_rot[0],head_rot[3]])
        #rot_vec = rot_obj.as_rotvec()
        rot_vec_ = rot_obj.apply([0,0,1])
        #rot_vec_ = np.array([rot_vec[0],rot_vec[1],rot_vec[2]])
        head_norm = np.linalg.norm(rot_vec_)
        unit_vec_head = (rot_vec_ / head_norm)
        unit_vec_head = np.array([unit_vec_head[2],
                                  -unit_vec_head[1],
                                  -unit_vec_head[0]])

        # Compute de normal distance between both orientations
        orientations_norm = np.linalg.norm(
                np.array(unit_vec_head_hand) - unit_vec_head
                )

        # Fill and return the observation
        hand_poses = [pose for pose in hand_pose]
        norm_hand_poses = self.normalize_with_bounds(
                    hand_poses,-1,1,-0.5,1.3)

        hand_poses_bis = [pose_bis for pose_bis in hand_pose_bis]
        norm_hand_poses_bis = self.normalize_with_bounds(
                    hand_poses_bis,-1,1,-0.5,1.3)

        angles = self.pepper.getAnglesPosition(self.r_kinematic_chain)
        norm_angles = list()
        for joint in self.r_kinematic_chain:
            #TODO..................
            bound_max = #get joint upper limit
            bound_min = #get joint lower limit
            angle = self.pepper.getAnglesPosition(joint)
            norm_angle = self.normalize_with_bounds(
                            angle, -1, 1, bound_min, bound_max
                         )
            norm_angles.extend(norm_angle)

        obs = [n for n in norm_hand_poses] +\
            [a for a in norm_angles] +\
            [n for n in norm_hand_poses_bis] +\
            [e for e in unit_vec_head_hand] +\
            [e for e in unit_vec_head]

        return obs


if __name__ == '__main__':
	handGesture = HandGesture()
	try:
        main()
        rospy.spin()