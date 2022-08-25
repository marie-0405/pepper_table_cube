;; Auto-generated. Do not edit!


(when (boundp 'pepper_training::QLearnElement)
  (if (not (find-package "PEPPER_TRAINING"))
    (make-package "PEPPER_TRAINING"))
  (shadow 'QLearnElement (find-package "PEPPER_TRAINING")))
(unless (find-package "PEPPER_TRAINING::QLEARNELEMENT")
  (make-package "PEPPER_TRAINING::QLEARNELEMENT"))

(in-package "ROS")
;;//! \htmlinclude QLearnElement.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass pepper_training::QLearnElement
  :super ros::object
  :slots (_qlearn_point _reward ))

(defmethod pepper_training::QLearnElement
  (:init
   (&key
    ((:qlearn_point __qlearn_point) (instance pepper_training::QLearnPoint :init))
    ((:reward __reward) (instance std_msgs::Float32 :init))
    )
   (send-super :init)
   (setq _qlearn_point __qlearn_point)
   (setq _reward __reward)
   self)
  (:qlearn_point
   (&rest __qlearn_point)
   (if (keywordp (car __qlearn_point))
       (send* _qlearn_point __qlearn_point)
     (progn
       (if __qlearn_point (setq _qlearn_point (car __qlearn_point)))
       _qlearn_point)))
  (:reward
   (&rest __reward)
   (if (keywordp (car __reward))
       (send* _reward __reward)
     (progn
       (if __reward (setq _reward (car __reward)))
       _reward)))
  (:serialization-length
   ()
   (+
    ;; pepper_training/QLearnPoint _qlearn_point
    (send _qlearn_point :serialization-length)
    ;; std_msgs/Float32 _reward
    (send _reward :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; pepper_training/QLearnPoint _qlearn_point
       (send _qlearn_point :serialize s)
     ;; std_msgs/Float32 _reward
       (send _reward :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; pepper_training/QLearnPoint _qlearn_point
     (send _qlearn_point :deserialize buf ptr-) (incf ptr- (send _qlearn_point :serialization-length))
   ;; std_msgs/Float32 _reward
     (send _reward :deserialize buf ptr-) (incf ptr- (send _reward :serialization-length))
   ;;
   self)
  )

(setf (get pepper_training::QLearnElement :md5sum-) "4b8989080d174aa31a6bec6bc8011ef4")
(setf (get pepper_training::QLearnElement :datatype-) "pepper_training/QLearnElement")
(setf (get pepper_training::QLearnElement :definition-)
      "pepper_training/QLearnPoint qlearn_point
std_msgs/Float32 reward

================================================================================
MSG: pepper_training/QLearnPoint
std_msgs/String state_tag
std_msgs/Int32 action_number
================================================================================
MSG: std_msgs/String
string data

================================================================================
MSG: std_msgs/Int32
int32 data
================================================================================
MSG: std_msgs/Float32
float32 data
")



(provide :pepper_training/QLearnElement "4b8989080d174aa31a6bec6bc8011ef4")


