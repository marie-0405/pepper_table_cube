#!/usr/bin/env python  
import roslib
import rospy

from geometry_msgs.msg import Pose

import math
import tf

def handle_hand_pose(msg, name):
    br = tf.TransformBroadcaster()
    position = msg.position
    orientation = msg.orientation
    print(position)
    print(orientation)
    br.sendTransform((position.x, position.y, position.z),
                     (orientation.x, orientation.y, orientation.z, orientation.w),
                     rospy.Time.now(),
                     name,
                     "odom")

if __name__ == '__main__':
    rospy.init_node('pepper_tf_broadcaster')
    # name = 'Position'
    # rospy.Subscriber('/%s/pose' % name, Pose, handle_hand_pose, name)
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
      br.sendTransform((0.3, 0.3, 0.8),
                        (0.0, 0.0, 0.0, 1.0),
                        rospy.Time.now(),
                        "Neck",
                        "world")
    rospy.spin()
