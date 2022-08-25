;; Auto-generated. Do not edit!


(when (boundp 'pepper_training::QLearnMatrix)
  (if (not (find-package "PEPPER_TRAINING"))
    (make-package "PEPPER_TRAINING"))
  (shadow 'QLearnMatrix (find-package "PEPPER_TRAINING")))
(unless (find-package "PEPPER_TRAINING::QLEARNMATRIX")
  (make-package "PEPPER_TRAINING::QLEARNMATRIX"))

(in-package "ROS")
;;//! \htmlinclude QLearnMatrix.msg.html


(defclass pepper_training::QLearnMatrix
  :super ros::object
  :slots (_q_learn_matrix ))

(defmethod pepper_training::QLearnMatrix
  (:init
   (&key
    ((:q_learn_matrix __q_learn_matrix) (let (r) (dotimes (i 0) (push (instance pepper_training::QLearnElement :init) r)) r))
    )
   (send-super :init)
   (setq _q_learn_matrix __q_learn_matrix)
   self)
  (:q_learn_matrix
   (&rest __q_learn_matrix)
   (if (keywordp (car __q_learn_matrix))
       (send* _q_learn_matrix __q_learn_matrix)
     (progn
       (if __q_learn_matrix (setq _q_learn_matrix (car __q_learn_matrix)))
       _q_learn_matrix)))
  (:serialization-length
   ()
   (+
    ;; pepper_training/QLearnElement[] _q_learn_matrix
    (apply #'+ (send-all _q_learn_matrix :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; pepper_training/QLearnElement[] _q_learn_matrix
     (write-long (length _q_learn_matrix) s)
     (dolist (elem _q_learn_matrix)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; pepper_training/QLearnElement[] _q_learn_matrix
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _q_learn_matrix (let (r) (dotimes (i n) (push (instance pepper_training::QLearnElement :init) r)) r))
     (dolist (elem- _q_learn_matrix)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get pepper_training::QLearnMatrix :md5sum-) "d1d271db6fedbe3a1e9e76a0df239d5a")
(setf (get pepper_training::QLearnMatrix :datatype-) "pepper_training/QLearnMatrix")
(setf (get pepper_training::QLearnMatrix :definition-)
      "pepper_training/QLearnElement[] q_learn_matrix
================================================================================
MSG: pepper_training/QLearnElement
pepper_training/QLearnPoint qlearn_point
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



(provide :pepper_training/QLearnMatrix "d1d271db6fedbe3a1e9e76a0df239d5a")


