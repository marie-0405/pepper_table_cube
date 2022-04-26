# coding: UTF-8
import pandas as pd
import rospkg
import rospy

class Information(object):

  def __init__(self, **kwargs):
    rospy.loginfo(kwargs)
    self.df = pd.DataFrame(kwargs)
  
  def write(self, file_name):
    # Get the file path
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('pepper_training')
    file_path = pkg_path + '/training_results/' + file_name

    # output dataframe to csv
    self.df.to_csv(file_path)

