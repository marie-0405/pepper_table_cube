;; Auto-generated. Do not edit!


(when (boundp 'pepper_training::QLearnPoint)
  (if (not (find-package "PEPPER_TRAINING"))
    (make-package "PEPPER_TRAINING"))
  (shadow 'QLearnPoint (find-package "PEPPER_TRAINING")))
(unless (find-package "PEPPER_TRAINING::QLEARNPOINT")
  (make-package "PEPPER_TRAINING::QLEARNPOINT"))

(in-package "ROS")
;;//! \htmlinclude QLearnPoint.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass pepper_training::QLearnPoint
  :super ros::object
  :slots (_state_tag _action_number ))

(defmethod pepper_training::QLearnPoint
  (:init
   (&key
    ((:state_tag __state_tag) (instance std_msgs::String :init))
    ((:action_number __action_number) (instance std_msgs::Int32 :init))
    )
   (send-super :init)
   (setq _state_tag __state_tag)
   (setq _action_number __action_number)
   self)
  (:state_tag
   (&rest __state_tag)
   (if (keywordp (car __state_tag))
       (send* _state_tag __state_tag)
     (progn
       (if __state_tag (setq _state_tag (car __state_tag)))
       _state_tag)))
  (:action_number
   (&rest __action_number)
   (if (keywordp (car __action_number))
       (send* _action_number __action_number)
     (progn
       (if __action_number (setq _action_number (car __action_number)))
       _action_number)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/String _state_tag
    (send _state_tag :serialization-length)
    ;; std_msgs/Int32 _action_number
    (send _action_number :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/String _state_tag
       (send _state_tag :serialize s)
     ;; std_msgs/Int32 _action_number
       (send _action_number :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/String _state_tag
     (send _state_tag :deserialize buf ptr-) (incf ptr- (send _state_tag :serialization-length))
   ;; std_msgs/Int32 _action_number
     (send _action_number :deserialize buf ptr-) (incf ptr- (send _action_number :serialization-length))
   ;;
   self)
  )

(setf (get pepper_training::QLearnPoint :md5sum-) "9fdfa6d65584c76899b1ef7f3f790a2c")
(setf (get pepper_training::QLearnPoint :datatype-) "pepper_training/QLearnPoint")
(setf (get pepper_training::QLearnPoint :definition-)
      "std_msgs/String state_tag
std_msgs/Int32 action_number
================================================================================
MSG: std_msgs/String
string data

================================================================================
MSG: std_msgs/Int32
int32 data
")



(provide :pepper_training/QLearnPoint "9fdfa6d65584c76899b1ef7f3f790a2c")


