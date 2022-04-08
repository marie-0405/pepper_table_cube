#!/usr/bin/env python

import rospy
import time
from rosgraph_msgs.msg import Clock


def callback(msg):
  print (msg.clock.secs)    

if __name__ == '__main__':   
  try:
    rospy.init_node('time_sub', anonymous=True)
    rospy.Subscriber('/clock', Clock, callback)                                       
    rospy.spin() 
  except rospy.ROSInterruptException: pass

    
  