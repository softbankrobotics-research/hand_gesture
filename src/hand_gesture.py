#!/usr/bin/env python3.5
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

#import qi
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

        #The position to be reached by the right hand of the robot
		self.target_local_position = PoseStamped()
		#The position of the hand of the robot
		self.hand_local_position = PoseStamped()
		#The position of the head of the robot
		self.head_local_position = PoseStamped()
		self.head_local_position.pose.orientation.w = 1
		#Gesture status
		self.active = False
		self.time_active = 0

		self.kinematic_chain = {
            #"HipPitch":       [-1.0385,  1.0385],
			"HipPitch":       [-0.4,  0.2],
            "RShoulderPitch": [-2.0857,  2.0857],
            "RShoulderRoll":  [-1.5620, -0.0087],
            "RElbowRoll":     [ 0.0087,  1.5620],
            "HeadYaw":        [-2.0857,  2.0857],
            "HeadPitch":      [-0.7068 , 0.4451]
            }
		self.joint_names = ["HipPitch",
		"RShoulderPitch",
		"RShoulderRoll",
		"RElbowRoll",
		"HeadYaw",
		"HeadPitch"]


		try:
			assert self.robot_ip is not None
			#self.session = qi.Session()
			#self.session.connect("tcp://" + self.robot_ip + ":9559")
			#self.motion = self.session.service("ALMotion")
			#self.posture = self.session.service("ALRobotPosture")
		except AssertionError:
			self.logFatal("Cannot retreive the robot's IP, the state " +\
                " the module won't be launched")
			return

		self.loadModel()

		subscriber_target = rospy.Subscriber(
		    "hand_gesture/target_local_position",
         	PoseStamped,
            self.callbackTarget)
		subscriber_hand = rospy.Subscriber(
		    "hand_gesture/hand_local_position",
         	PoseStamped,
            self.callbackHand)
		subscriber_head = rospy.Subscriber(
		    "hand_gesture/head_local_position",
         	PoseStamped,
            self.callbackHead)

		subscriber_head = rospy.Subscriber(
		    "joint_states",
         	JointState,
            self.callbackJointStates)

		subscriber_active = rospy.Subscriber(
		    "hand_gesture/active",
         	Bool,
            self.callbackActive)

		subscriber_shutdown = rospy.Subscriber(
		    "hand_gesture/shutdown",
         	Bool,
            self.callbackShutdown)

		self.pub_joints = rospy.Publisher(
		     "joint_angles",
			 JointAnglesWithSpeed,
			 queue_size=1)

	def loadModel(self):
		"""
		Load a pre-trained PPO2 model on stable_baselines
        """
		#file = "ppo2_hand_berserk_wozniak" # hand only
		#file = "ppo2_hand_condescending_khorana" #hand only
		file = "ppo2_hand_determined_borg"
		path = "../models/"+file
		self.model = PPO2.load(path)

	def callbackJointStates(self, joint_states):
		"""
		Get the target position JointState message
		"""
		self.joint_states = joint_states

	def callbackTarget(self, target_local_position):
		"""
		Get the target position PointStamped message
		"""
		self.target_local_position = target_local_position

	def callbackHand(self, hand_local_position):
		"""
		Get the target position PointStamped message
		"""
		self.hand_local_position = hand_local_position

	def callbackHead(self, head_local_position):
		"""
		Get the target position PointStamped message
		"""
		self.head_local_position = head_local_position

	def callbackActive(self, active):
		"""
		Activates or deactivates the gesture motion
		"""
		if active.data == True:
			if self.active == False:
				print("activating gesture")
				self.active = True
				self.time_active = rospy.get_rostime()
		else:
			self.active = False

	def callbackShutdown(self, data):
		"""
		Shutdown the node
		"""
		print("Shutting down the gesture module")
		if data.data == True:
			print("Shutdown")
			#do something

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
			#self.motion.changeAngles(angle, 0.1, velocity)

	def normalize_with_bounds(
        self, values, range_min, range_max,
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

	def getObservation(self):
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

        #target_pos_bis = target_pos
        #target_pos_bis[1] = target_pos_bis[1] - 0.05
		hands_norm = np.linalg.norm(hand_pose - target_pos)

        # Get information about the head direction and hand
        # Head to Hand reward based on the direction of the head to the hand
		head_hand_norm =  np.linalg.norm(np.array(head_pose) - np.array(target_pos))

		vec_head_hand = np.array(target_pos) - np.array(head_pose)
		unit_vec_head_hand = vec_head_hand / head_hand_norm

		"""
		rot_obj =  R.from_quat([head_rot[2],
		                        head_rot[1],
								head_rot[0],
								head_rot[3]])
		"""
		rot_obj =  R.from_quat([head_rot[2],
		                        head_rot[1],
								head_rot[0],
								head_rot[3]])
        #rot_vec = rot_obj.as_rotvec()
		rot_vec_ = rot_obj.apply([0,0,1])
        #rot_vec_ = np.array([rot_vec[0],rot_vec[1],rot_vec[2]])
		head_norm = np.linalg.norm(rot_vec_)
		unit_vec_head = (rot_vec_ / head_norm)
		unit_vec_head = np.array([unit_vec_head[0],
                                  -unit_vec_head[1],
                                  -unit_vec_head[2]])

        # Compute de normal distance between both orientations
		orientations_norm = np.linalg.norm(
                np.array(unit_vec_head_hand) - unit_vec_head
                )

        # Fill and return the observation
		hand_poses = [pose for pose in hand_pose]
		norm_hand_poses = self.normalize_with_bounds(
                    hand_poses,-1,1,-0.5,1.3)

		hand_poses_bis = [pose_bis for pose_bis in target_pos]
		norm_hand_poses_bis = self.normalize_with_bounds(
                    hand_poses_bis,-1,1,-0.5,1.3)
		norm_angles = list()
		name_angles = list()
		index_joint = 0
		self.angles = []
		for joint in self.joint_names:
			index_joints = 0
			for joints in self.joint_states.name:
				if joint == joints:
					bound_min = self.kinematic_chain[joint][0]
					bound_max = self.kinematic_chain[joint][1]
					angle=[self.joint_states.position[index_joints]]
					self.angles.append(self.joint_states.position[index_joints])

					norm_angle = self.normalize_with_bounds(
		                            angle, -1, 1, bound_min, bound_max
		                         )

					norm_angles.extend(norm_angle)
					name_angles.extend([joint])

				index_joints = index_joints + 1
			index_joint = index_joint + 1

		#print(name_angles)
		#print(norm_angles)
		#print(self.joint_states.name)
		#print(self.joint_states.position)
		#print(hand_pose)
		#print(self.angles)
		#print(target_pos)
		obs = [n for n in norm_hand_poses] +\
		                [a for a in norm_angles] +\
		                [n for n in norm_hand_poses_bis] +\
		                [e for e in unit_vec_head_hand] +\
		                [e for e in unit_vec_head]
		return obs

	def setAngles(self):
		#self.joint_angles.joint_names = self.joint_names

		angles = []
		speeds = []
		speed = []
		i = 0
		"""
		for action in self.action:
			if action>0:
				speed = action * 0.07
				angles = 0.05
			elif action<0:
				speed = action * -0.07
				angles = -0.05
			else:
				angles = 0
				speed = 0
			#print(angles)
			#print(speeds)
			if self.joint_names[i]=="HipPitch":
				if(self.angles[i]<-0.40 or self.angles[i]>0.2):
					angles = 0
					speed = 0
				else:
					angles = angles*0.1
			elif self.joint_names[i]=="RElbowRoll":
				speed = 1
				angles = angles*1
			elif (self.joint_names[i]=="HeadYaw" or self.joint_names[i]=="HeadPitch"):
				angles = angles*0.5
				speed = 0.7
			else:
				speed = 1
		"""
		for action in self.action:
			if action>0:
				speed = action * 0.05
				angles = 0.08
			elif action<0:
				speed = action * -0.05
				angles = -0.08
			else:
				angles = 0
				speed = 0
			if self.joint_names[i]=="HipPitch":
				if(self.angles[i]<-0.40 or self.angles[i]>0.2):
					angles = 0
					speed = 0
			self.joint_angles.joint_names = [self.joint_names[i]]
			self.joint_angles.joint_angles = [angles]
			self.joint_angles.speed = speed
			#self.joint_angles.speeds = speeds
			self.joint_angles.relative = 1
			#print(self.joint_angles)
			self.pub_joints.publish(self.joint_angles)
			i = i + 1
			#print(self.joint_names)

	def start(self):
		if self.active:
			obs = self.getObservation()
			#print (obs)
			self.action, _states = self.model.predict(obs, deterministic=True)
			print(self.action)
			self.setAngles()
			#self.setVelocities(self.kinematic_chain, action)


if __name__ == '__main__':
	handGesture = HandGesture()
    #handGesture.start()
	try:
		rate = rospy.Rate(30.0)
		while not rospy.is_shutdown():
			#rospy.sleep(0.5)
			handGesture.start()
			rate.sleep()
	except KeyboardInterrupt:
		pass
