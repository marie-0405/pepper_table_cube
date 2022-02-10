#!/usr/bin/env python  
import roslib
import rospy

from geometry_msgs.msg import Pose

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
    name = 'Position'
    rospy.Subscriber('/%s/pose' % name, Pose, handle_hand_pose, name)
    rospy.spin()
