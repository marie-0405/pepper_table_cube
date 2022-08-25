; Auto-generated. Do not edit!


(cl:in-package pepper_training-msg)


;//! \htmlinclude QLearnElement.msg.html

(cl:defclass <QLearnElement> (roslisp-msg-protocol:ros-message)
  ((qlearn_point
    :reader qlearn_point
    :initarg :qlearn_point
    :type pepper_training-msg:QLearnPoint
    :initform (cl:make-instance 'pepper_training-msg:QLearnPoint))
   (reward
    :reader reward
    :initarg :reward
    :type std_msgs-msg:Float32
    :initform (cl:make-instance 'std_msgs-msg:Float32)))
)

(cl:defclass QLearnElement (<QLearnElement>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <QLearnElement>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'QLearnElement)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name pepper_training-msg:<QLearnElement> is deprecated: use pepper_training-msg:QLearnElement instead.")))

(cl:ensure-generic-function 'qlearn_point-val :lambda-list '(m))
(cl:defmethod qlearn_point-val ((m <QLearnElement>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pepper_training-msg:qlearn_point-val is deprecated.  Use pepper_training-msg:qlearn_point instead.")
  (qlearn_point m))

(cl:ensure-generic-function 'reward-val :lambda-list '(m))
(cl:defmethod reward-val ((m <QLearnElement>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pepper_training-msg:reward-val is deprecated.  Use pepper_training-msg:reward instead.")
  (reward m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <QLearnElement>) ostream)
  "Serializes a message object of type '<QLearnElement>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'qlearn_point) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'reward) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <QLearnElement>) istream)
  "Deserializes a message object of type '<QLearnElement>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'qlearn_point) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'reward) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<QLearnElement>)))
  "Returns string type for a message object of type '<QLearnElement>"
  "pepper_training/QLearnElement")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'QLearnElement)))
  "Returns string type for a message object of type 'QLearnElement"
  "pepper_training/QLearnElement")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<QLearnElement>)))
  "Returns md5sum for a message object of type '<QLearnElement>"
  "4b8989080d174aa31a6bec6bc8011ef4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'QLearnElement)))
  "Returns md5sum for a message object of type 'QLearnElement"
  "4b8989080d174aa31a6bec6bc8011ef4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<QLearnElement>)))
  "Returns full string definition for message of type '<QLearnElement>"
  (cl:format cl:nil "pepper_training/QLearnPoint qlearn_point~%std_msgs/Float32 reward~%~%================================================================================~%MSG: pepper_training/QLearnPoint~%std_msgs/String state_tag~%std_msgs/Int32 action_number~%================================================================================~%MSG: std_msgs/String~%string data~%~%================================================================================~%MSG: std_msgs/Int32~%int32 data~%================================================================================~%MSG: std_msgs/Float32~%float32 data~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'QLearnElement)))
  "Returns full string definition for message of type 'QLearnElement"
  (cl:format cl:nil "pepper_training/QLearnPoint qlearn_point~%std_msgs/Float32 reward~%~%================================================================================~%MSG: pepper_training/QLearnPoint~%std_msgs/String state_tag~%std_msgs/Int32 action_number~%================================================================================~%MSG: std_msgs/String~%string data~%~%================================================================================~%MSG: std_msgs/Int32~%int32 data~%================================================================================~%MSG: std_msgs/Float32~%float32 data~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <QLearnElement>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'qlearn_point))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'reward))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <QLearnElement>))
  "Converts a ROS message object to a list"
  (cl:list 'QLearnElement
    (cl:cons ':qlearn_point (qlearn_point msg))
    (cl:cons ':reward (reward msg))
))
