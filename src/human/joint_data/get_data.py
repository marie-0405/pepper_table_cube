from ntpath import join
import pandas as pd

FILE_NAME = 'pushing_task_3d'
joint_df = pd.read_csv('./joint_data/{}.csv'.format(FILE_NAME))
print(joint_df)
right_arm = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"]

# Get right arm joint data
joint_df = joint_df.loc[:, right_arm]
joint_df.loc[:, 'RWristYaw'] = 0.0  # Set 0 to all of RWristYaw
print(joint_df.values)