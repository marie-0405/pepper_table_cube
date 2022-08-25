import os
import qi
import argparse
import sys
import pandas as pd
import nep_aldebaran  # added

cwd = os.getcwd()
FILE_NAME = "pushing_task4_3d"

# Define NAO/Pepper parameters, port is often 9559
robot_port = "9559"
# Change this value for the IP of your robot, for the simulator is 127.0.0.1
robot_ip = '127.0.0.1'

# Set path of animations and type of robot
path_animations = ""   # Not used in this example
type_robot = "pepper"  # or nao


# df = pd.read_csv(cwd + "\\..\\resources\\obama.csv")
df = pd.read_csv(cwd + '/joint_data/{}.csv'.format(FILE_NAME))
columns = list(df.columns.values)
JointAngles = [h for h in columns if 'TimeStamp' not in h]

HipRoll = df['HipRoll'].values.tolist()
HipPitch = df['HipPitch'].values.tolist()
HeadYaw = df['HeadYaw'].values.tolist()
HeadPitch = df['HeadPitch'].values.tolist()
LShoulderPitch = df['LShoulderPitch'].values.tolist()
RShoulderPitch = df['RShoulderPitch'].values.tolist()
LElbowYaw = df['LElbowYaw'].values.tolist()
RElbowYaw = df['RElbowYaw'].values.tolist()
LShoulderRoll = df['LShoulderRoll'].values.tolist()
RShoulderRoll = df['RShoulderRoll'].values.tolist()
LElbowRoll = df['LElbowRoll'].values.tolist()
RElbowRoll = df['RElbowRoll'].values.tolist()
TimeStamp = df['TimeStamp'].values.tolist()

movement = {'RShoulderRoll': RShoulderRoll, 'LShoulderRoll': LShoulderRoll,
            'RElbowRoll': RElbowRoll, 'LElbowRoll': LElbowRoll, 'HeadYaw': HeadYaw,
            'HeadPitch': HeadPitch, 'LShoulderPitch': LShoulderPitch,
            'RShoulderPitch': RShoulderPitch, 'LElbowYaw': LElbowYaw,
            'RElbowYaw': RElbowYaw, 'HipRoll': HipRoll, 'HipPitch': HipPitch}

def main(session):

    move = nep_aldebaran.BodyMove(robot_ip, robot_port, type_robot, path_animations)
    move.onRunMode("wake_up")  # added
    motion_service  = session.service("ALMotion")
    motion_service.setStiffnesses("Head", 1.0)
    motion_service.setStiffnesses("LArm", 1.0)
    motion_service.setStiffnesses("RArm", 1.0)

    names  = JointAngles
    angleLists = [movement[JointAngles[i]] for i in range(len(JointAngles))]
    timeLists = [TimeStamp for j in range(len(JointAngles))]
    isAbsolute  = True
    motion_service.angleInterpolation(names, angleLists, timeLists, isAbsolute)

    motion_service.setStiffnesses("Head", 0.0)
    motion_service.setStiffnesses("LArm", 0.0)
    motion_service.setStiffnesses("RArm", 0.0)
    move.onRunMode("rest")  # added


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="127.0.0.1")
    parser.add_argument("--port", type=int, default=9559,
                        help="9559")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) + ".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session)