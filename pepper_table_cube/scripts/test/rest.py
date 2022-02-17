# DESCRIPTION:
# Use this lines to stop default behavior of Pepper and Autonomus life
# These lines will solve many issues when using Pepper
import nep_aldebaran
# Define NAO/Pepper parameters, port is often 9559
robot_port = "9559"
# Change this value for the IP of your robot, for the simulator is 127.0.0.1
robot_ip = '192.168.0.102'

behavior = nep_aldebaran.BehaviorManager(robot_ip, robot_port)
behavior.onStop()
autonomus = nep_aldebaran.AutonomusLife(robot_ip, robot_port)
autonomus.onStop()
