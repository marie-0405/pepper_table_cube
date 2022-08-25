; Auto-generated. Do not edit!


(cl:in-package pepper_training-msg)


;//! \htmlinclude QLearnPoint.msg.html

(cl:defclass <QLearnPoint> (roslisp-msg-protocol:ros-message)
  ((state_tag
    :reader state_tag
    :initarg :state_tag
    :type std_msgs-msg:String
    :initform (cl:make-instance 'std_msgs-msg:String))
   (action_number
    :reader action_number
    :initarg :action_number
    :type std_msgs-msg:Int32
    :initform (cl:make-instance 'std_msgs-msg:Int32)))
)

(cl:defclass QLearnPoint (<QLearnPoint>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <QLearnPoint>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'QLearnPoint)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name pepper_training-msg:<QLearnPoint> is deprecated: use pepper_training-msg:QLearnPoint instead.")))

(cl:ensure-generic-function 'state_tag-val :lambda-list '(m))
(cl:defmethod state_tag-val ((m <QLearnPoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pepper_training-msg:state_tag-val is deprecated.  Use pepper_training-msg:state_tag instead.")
  (state_tag m))

(cl:ensure-generic-function 'action_number-val :lambda-list '(m))
(cl:defmethod action_number-val ((m <QLearnPoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pepper_training-msg:action_number-val is deprecated.  Use pepper_training-msg:action_number instead.")
  (action_number m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <QLearnPoint>) ostream)
  "Serializes a message object of type '<QLearnPoint>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'state_tag) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'action_number) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <QLearnPoint>) istream)
  "Deserializes a message object of type '<QLearnPoint>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'state_tag) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'action_number) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<QLearnPoint>)))
  "Returns string type for a message object of type '<QLearnPoint>"
  "pepper_training/QLearnPoint")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'QLearnPoint)))
  "Returns string type for a message object of type 'QLearnPoint"
  "pepper_training/QLearnPoint")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<QLearnPoint>)))
  "Returns md5sum for a message object of type '<QLearnPoint>"
  "9fdfa6d65584c76899b1ef7f3f790a2c")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'QLearnPoint)))
  "Returns md5sum for a message object of type 'QLearnPoint"
  "9fdfa6d65584c76899b1ef7f3f790a2c")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<QLearnPoint>)))
  "Returns full string definition for message of type '<QLearnPoint>"
  (cl:format cl:nil "std_msgs/String state_tag~%std_msgs/Int32 action_number~%================================================================================~%MSG: std_msgs/String~%string data~%~%================================================================================~%MSG: std_msgs/Int32~%int32 data~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'QLearnPoint)))
  "Returns full string definition for message of type 'QLearnPoint"
  (cl:format cl:nil "std_msgs/String state_tag~%std_msgs/Int32 action_number~%================================================================================~%MSG: std_msgs/String~%string data~%~%================================================================================~%MSG: std_msgs/Int32~%int32 data~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <QLearnPoint>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'state_tag))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'action_number))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <QLearnPoint>))
  "Converts a ROS message object to a list"
  (cl:list 'QLearnPoint
    (cl:cons ':state_tag (state_tag msg))
    (cl:cons ':action_number (action_number msg))
))
