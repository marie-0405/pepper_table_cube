// Auto-generated. Do not edit!

// (in-package pepper_training.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let QLearnElement = require('./QLearnElement.js');

//-----------------------------------------------------------

class QLearnMatrix {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.q_learn_matrix = null;
    }
    else {
      if (initObj.hasOwnProperty('q_learn_matrix')) {
        this.q_learn_matrix = initObj.q_learn_matrix
      }
      else {
        this.q_learn_matrix = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type QLearnMatrix
    // Serialize message field [q_learn_matrix]
    // Serialize the length for message field [q_learn_matrix]
    bufferOffset = _serializer.uint32(obj.q_learn_matrix.length, buffer, bufferOffset);
    obj.q_learn_matrix.forEach((val) => {
      bufferOffset = QLearnElement.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type QLearnMatrix
    let len;
    let data = new QLearnMatrix(null);
    // Deserialize message field [q_learn_matrix]
    // Deserialize array length for message field [q_learn_matrix]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.q_learn_matrix = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.q_learn_matrix[i] = QLearnElement.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.q_learn_matrix.forEach((val) => {
      length += QLearnElement.getMessageSize(val);
    });
    return length + 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'pepper_training/QLearnMatrix';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd1d271db6fedbe3a1e9e76a0df239d5a';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    pepper_training/QLearnElement[] q_learn_matrix
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
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new QLearnMatrix(null);
    if (msg.q_learn_matrix !== undefined) {
      resolved.q_learn_matrix = new Array(msg.q_learn_matrix.length);
      for (let i = 0; i < resolved.q_learn_matrix.length; ++i) {
        resolved.q_learn_matrix[i] = QLearnElement.Resolve(msg.q_learn_matrix[i]);
      }
    }
    else {
      resolved.q_learn_matrix = []
    }

    return resolved;
    }
};

module.exports = QLearnMatrix;
