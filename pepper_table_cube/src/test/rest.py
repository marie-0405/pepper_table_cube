# DESCRIPTION:
# Use these lines to stop the default behavior of Pepper and Autonomous life
# These lines will solve many issues when using Pepper

import nep_aldebaran
# Define NAO/Pepper parameters, port is often 9559
robot_port = "9559"
# Change this value for the IP of your robot, ex
robot_ip = '192.168.11.30'

behavior = nep_aldebaran.BehaviorManager(robot_ip, robot_port)
behavior.onStop()
autonomus = nep_aldebaran.AutonomusLife(robot_ip, robot_port)
autonomus.onStop()
