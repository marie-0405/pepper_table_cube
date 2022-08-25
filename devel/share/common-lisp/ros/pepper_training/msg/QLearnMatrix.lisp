; Auto-generated. Do not edit!


(cl:in-package pepper_training-msg)


;//! \htmlinclude QLearnMatrix.msg.html

(cl:defclass <QLearnMatrix> (roslisp-msg-protocol:ros-message)
  ((q_learn_matrix
    :reader q_learn_matrix
    :initarg :q_learn_matrix
    :type (cl:vector pepper_training-msg:QLearnElement)
   :initform (cl:make-array 0 :element-type 'pepper_training-msg:QLearnElement :initial-element (cl:make-instance 'pepper_training-msg:QLearnElement))))
)

(cl:defclass QLearnMatrix (<QLearnMatrix>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <QLearnMatrix>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'QLearnMatrix)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name pepper_training-msg:<QLearnMatrix> is deprecated: use pepper_training-msg:QLearnMatrix instead.")))

(cl:ensure-generic-function 'q_learn_matrix-val :lambda-list '(m))
(cl:defmethod q_learn_matrix-val ((m <QLearnMatrix>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pepper_training-msg:q_learn_matrix-val is deprecated.  Use pepper_training-msg:q_learn_matrix instead.")
  (q_learn_matrix m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <QLearnMatrix>) ostream)
  "Serializes a message object of type '<QLearnMatrix>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'q_learn_matrix))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'q_learn_matrix))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <QLearnMatrix>) istream)
  "Deserializes a message object of type '<QLearnMatrix>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'q_learn_matrix) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'q_learn_matrix)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'pepper_training-msg:QLearnElement))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<QLearnMatrix>)))
  "Returns string type for a message object of type '<QLearnMatrix>"
  "pepper_training/QLearnMatrix")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'QLearnMatrix)))
  "Returns string type for a message object of type 'QLearnMatrix"
  "pepper_training/QLearnMatrix")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<QLearnMatrix>)))
  "Returns md5sum for a message object of type '<QLearnMatrix>"
  "d1d271db6fedbe3a1e9e76a0df239d5a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'QLearnMatrix)))
  "Returns md5sum for a message object of type 'QLearnMatrix"
  "d1d271db6fedbe3a1e9e76a0df239d5a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<QLearnMatrix>)))
  "Returns full string definition for message of type '<QLearnMatrix>"
  (cl:format cl:nil "pepper_training/QLearnElement[] q_learn_matrix~%================================================================================~%MSG: pepper_training/QLearnElement~%pepper_training/QLearnPoint qlearn_point~%std_msgs/Float32 reward~%~%================================================================================~%MSG: pepper_training/QLearnPoint~%std_msgs/String state_tag~%std_msgs/Int32 action_number~%================================================================================~%MSG: std_msgs/String~%string data~%~%================================================================================~%MSG: std_msgs/Int32~%int32 data~%================================================================================~%MSG: std_msgs/Float32~%float32 data~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'QLearnMatrix)))
  "Returns full string definition for message of type 'QLearnMatrix"
  (cl:format cl:nil "pepper_training/QLearnElement[] q_learn_matrix~%================================================================================~%MSG: pepper_training/QLearnElement~%pepper_training/QLearnPoint qlearn_point~%std_msgs/Float32 reward~%~%================================================================================~%MSG: pepper_training/QLearnPoint~%std_msgs/String state_tag~%std_msgs/Int32 action_number~%================================================================================~%MSG: std_msgs/String~%string data~%~%================================================================================~%MSG: std_msgs/Int32~%int32 data~%================================================================================~%MSG: std_msgs/Float32~%float32 data~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <QLearnMatrix>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'q_learn_matrix) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <QLearnMatrix>))
  "Converts a ROS message object to a list"
  (cl:list 'QLearnMatrix
    (cl:cons ':q_learn_matrix (q_learn_matrix msg))
))
