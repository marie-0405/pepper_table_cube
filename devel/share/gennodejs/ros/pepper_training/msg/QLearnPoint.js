// Auto-generated. Do not edit!

// (in-package pepper_training.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class QLearnPoint {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.state_tag = null;
      this.action_number = null;
    }
    else {
      if (initObj.hasOwnProperty('state_tag')) {
        this.state_tag = initObj.state_tag
      }
      else {
        this.state_tag = new std_msgs.msg.String();
      }
      if (initObj.hasOwnProperty('action_number')) {
        this.action_number = initObj.action_number
      }
      else {
        this.action_number = new std_msgs.msg.Int32();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type QLearnPoint
    // Serialize message field [state_tag]
    bufferOffset = std_msgs.msg.String.serialize(obj.state_tag, buffer, bufferOffset);
    // Serialize message field [action_number]
    bufferOffset = std_msgs.msg.Int32.serialize(obj.action_number, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type QLearnPoint
    let len;
    let data = new QLearnPoint(null);
    // Deserialize message field [state_tag]
    data.state_tag = std_msgs.msg.String.deserialize(buffer, bufferOffset);
    // Deserialize message field [action_number]
    data.action_number = std_msgs.msg.Int32.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.String.getMessageSize(object.state_tag);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'pepper_training/QLearnPoint';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '9fdfa6d65584c76899b1ef7f3f790a2c';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/String state_tag
    std_msgs/Int32 action_number
    ================================================================================
    MSG: std_msgs/String
    string data
    
    ================================================================================
    MSG: std_msgs/Int32
    int32 data
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new QLearnPoint(null);
    if (msg.state_tag !== undefined) {
      resolved.state_tag = std_msgs.msg.String.Resolve(msg.state_tag)
    }
    else {
      resolved.state_tag = new std_msgs.msg.String()
    }

    if (msg.action_number !== undefined) {
      resolved.action_number = std_msgs.msg.Int32.Resolve(msg.action_number)
    }
    else {
      resolved.action_number = new std_msgs.msg.Int32()
    }

    return resolved;
    }
};

module.exports = QLearnPoint;
