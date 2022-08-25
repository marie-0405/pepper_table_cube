// Auto-generated. Do not edit!

// (in-package pepper_training.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let QLearnPoint = require('./QLearnPoint.js');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class QLearnElement {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.qlearn_point = null;
      this.reward = null;
    }
    else {
      if (initObj.hasOwnProperty('qlearn_point')) {
        this.qlearn_point = initObj.qlearn_point
      }
      else {
        this.qlearn_point = new QLearnPoint();
      }
      if (initObj.hasOwnProperty('reward')) {
        this.reward = initObj.reward
      }
      else {
        this.reward = new std_msgs.msg.Float32();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type QLearnElement
    // Serialize message field [qlearn_point]
    bufferOffset = QLearnPoint.serialize(obj.qlearn_point, buffer, bufferOffset);
    // Serialize message field [reward]
    bufferOffset = std_msgs.msg.Float32.serialize(obj.reward, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type QLearnElement
    let len;
    let data = new QLearnElement(null);
    // Deserialize message field [qlearn_point]
    data.qlearn_point = QLearnPoint.deserialize(buffer, bufferOffset);
    // Deserialize message field [reward]
    data.reward = std_msgs.msg.Float32.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += QLearnPoint.getMessageSize(object.qlearn_point);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'pepper_training/QLearnElement';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '4b8989080d174aa31a6bec6bc8011ef4';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
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
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new QLearnElement(null);
    if (msg.qlearn_point !== undefined) {
      resolved.qlearn_point = QLearnPoint.Resolve(msg.qlearn_point)
    }
    else {
      resolved.qlearn_point = new QLearnPoint()
    }

    if (msg.reward !== undefined) {
      resolved.reward = std_msgs.msg.Float32.Resolve(msg.reward)
    }
    else {
      resolved.reward = new std_msgs.msg.Float32()
    }

    return resolved;
    }
};

module.exports = QLearnElement;
