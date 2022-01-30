import nep_aldebaran
import time
# Define NAO/Pepper parameters, port is often 9559
robot_port = "9559"
# Change this value for the IP of your robot, for the simulator is 127.0.0.1
robot_ip = '192.168.11.30'

# Set path of animations and type of robot
path_animations = ""   # Not used in this example
type_robot = "pepper"  # or nao

# Class used to move Pepper/NAO
move = nep_aldebaran.BodyMove(robot_ip, robot_port, type_robot, path_animations)
# Examples
move.onRunMode("wake_up") #Wake up or rest NAO/Pepper - "rest" or "wake_up"
time.sleep(2)
move.onBreathe("off")