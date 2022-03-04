# How to execute
## 1. Launch pepper_table_cube.launch
To launch the gazebo environment, run the following and start simulation (click the start button of gazebo).
```
roslaunch pepper_table_cube pepper_table_cube.launch
```

## 2. Observe feedback of right arm
To observe the right arm joint states, run the following.
get_feedback.py gets the feedbacks of right arm, and output them as csv file(scripts/test/data/feedback_*.csv).
```
rosrun pepper_table_cube get_feedback.py
```

## 3. Move Pepper robot using topic
This code moves Pepper's right arm with topic.
```
rosrun pepper_table_cube action_pepper_move.py
```

## 4. Shutdown the get_feedback.py
By Ctrl + c, shutdown the get_feedback.py. Then csv file is created.

## 5. Display figure of joint states
```
cd ~/catkin_ws/src/research_pepper/pepper_table_cube/scripts
python display_feedback.py
```
