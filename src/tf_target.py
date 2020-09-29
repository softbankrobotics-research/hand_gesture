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


import os
import sys
import threading
import rospy
import argparse
import tf
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool


class tfTarget:
    """
    Class defining the target frame
    """
    def __init__(self):
        """
        Constructor
        """
        self.init_targets()

        # Initialize node
        rospy.init_node(
            "tf_target",
            anonymous=True,
            disable_signals=False,
            log_level=rospy.INFO)

        # Subscribers
        rospy.Subscriber(
            "hand_gesture/init_targets",
            Bool,
            self.callback_init_targets)

        rospy.Subscriber(
            "hand_detection/target_camera",
            PoseStamped,
            self.callback_target_camera)

        rospy.Subscriber(
            "hand_gesture/target_local_position",
            PoseStamped,
            self.callback_target_position)

        # Publisher
        self.pub_tf = rospy.Publisher(
            "/tf",
            tf2_msgs.msg.TFMessage,
            queue_size=1)

        self.tf_listener = tf.TransformListener()

    def init_targets(self):
        """
        Initialize the position of the target camera
        and the target position
        """
        self.target_camera = PoseStamped()
        self.target_camera.pose.position.x = 1
        self.target_camera.pose.position.y = 0
        self.target_camera.pose.position.z = 0
        self.target_camera.pose.orientation.x = 0
        self.target_camera.pose.orientation.y = 0
        self.target_camera.pose.orientation.z = 0
        self.target_camera.pose.orientation.w = 1

        self.target_position = PoseStamped()
        self.target_position.pose.position.x = 1
        self.target_position.pose.position.y = 0
        self.target_position.pose.position.z = 0
        self.target_position.pose.orientation.x = 0
        self.target_position.pose.orientation.y = 0
        self.target_position.pose.orientation.z = 0
        self.target_position.pose.orientation.w = 1

    def callback_init_targets(self, active):
        """
        Initialize the targets and publish the frame of
        the target position

        Parameters:
            active  a Boolean value to initialize the targets
        """
        mutex = threading.Lock()
        mutex.acquire()
        self.init_targets()
        self.publish_frame(
            "base_link",
            "target_position",
            self.target_position)
        mutex.release()

    def callback_target_position(self, target_position):
        """
        Update the target position

        Parameters:
            target_position the position of the target
            of type geometry_msgs/PointStamped
        """
        self.target_position = target_position

    def callback_target_camera(self, target_camera):
        """
        Update the target from the camera PoV

        Parameters:
            target_camera the target from the point of view
            of the camera of type geometry_msgs/PointStamped
        """
        self.target_camera = target_camera
        # Transform the target frame to local reference
        try:
            (target, _) = self.tf_listener.lookupTransform(
                "base_link",
                "target_camera",
                rospy.Time())
            self.target_position = PoseStamped()
            self.target_position.header.frame_id = 'target_position'
            self.target_position.header.stamp = rospy.Time.now()
            self.target_position.pose.position.x = target[0]
            self.target_position.pose.position.y = target[1]
            self.target_position.pose.position.z = target[2]
        except (tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException):
            pass

    def publish_frame(self, frame_head, frame_child, frame):
        """
        Get the position and orientation of a frame specified as frame_child
        with respect to other frame specified as frame_head

        Parameters:
            frame_head  the ros frame reference
            frame_child  the ros frame to update
            frame  the values to assign to the frame_child

        Publisher Topic:
            /tf
        """
        t = geometry_msgs.msg.TransformStamped()
        t.header.frame_id = frame_head
        t.header.stamp = rospy.Time.now()
        t.child_frame_id = frame_child
        t.transform.translation.x = \
            frame.pose.position.x
        t.transform.translation.y = \
            frame.pose.position.y
        t.transform.translation.z = \
            frame.pose.position.z
        t.transform.rotation.x = 0
        t.transform.rotation.y = 0
        t.transform.rotation.z = 0
        t.transform.rotation.w = 1

        tfm = tf2_msgs.msg.TFMessage([t])
        self.pub_tf.publish(tfm)

    def start(self):
        """
        Transform frames to local reference (odom)
        publish PoseStamped msgs of the head, hand and target
        """
        self.publish_frame(
            "RealSense_optical_frame",
            "target_camera",
            self.target_camera)
        self.publish_frame(
            "base_link",
            "target_position",
            self.target_position)


if __name__ == '__main__':
    tfTarget = tfTarget()
    try:
        rate = rospy.Rate(50.0)
        while not rospy.is_shutdown():
            tfTarget.start()
            rate.sleep()
    except KeyboardInterrupt:
        pass
