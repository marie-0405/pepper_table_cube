cmake_minimum_required(VERSION 2.8.3)
project(my_pepper_gazebo_plugin)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY launch worlds
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
