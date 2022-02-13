#!/usr/bin/env python
from test.link import Link
from test.model import Model

import os
import pandas as pd
import rospy

from control_msgs.msg import FollowJointTrajectoryActionFeedback

FILE_NAME = 'feedback'
RATE = 25

dirname = os.path.dirname(__file__)
cwd = os.getcwd()


if __name__ == '__main__':
  rospy.init_node('get_feedback', anonymous=True)

  topic = '/pepper_dcm/RightArm_controller/follow_joint_trajectory/feedback'
  joint = rospy.wait_for_message(topic, FollowJointTrajectoryActionFeedback)
  joint_names = joint.feedback.joint_names[:-1]
  
  # print(joint_names)

  rate = rospy.Rate(RATE)
  joint_dict = {}

  for i, joint_name in enumerate(joint_names):
    joint_dict['desired_' + joint_name] = joint.feedback.desired.positions[i]
    joint_dict['actual_' + joint_name] = joint.feedback.actual.positions[i]

  df = pd.DataFrame(joint_dict, index=[0, ])
  count = 0

  try:
    while not rospy.is_shutdown():
      count += 1 
      joint = rospy.wait_for_message(topic, FollowJointTrajectoryActionFeedback)
      desired_pos = joint.feedback.desired.positions
      actual_pos = joint.feedback.actual.positions
      df.loc[count] = [desired_pos[3], desired_pos[2], desired_pos[0], desired_pos[1],
                        actual_pos[3], actual_pos[2], actual_pos[0], actual_pos[1]]                     
      rate.sleep()

  except rospy.ROSInterruptException:
    df.to_csv(dirname + '/test/data/{}.csv'.format(FILE_NAME), index=False)
  
