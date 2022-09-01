import nep
import time


node = nep.node("publisher_sample")    # Create a new nep node
conf = node.hybrid('192.168.0.101')
pub = node.new_pub("test","json", conf)      # Set the topic and message type 

i = 0
while True:         #  Publish a message each second
    i = i + 1
    msg =  data = {"message":i}
    pub.publish(msg)
    print ("sending: " + str(msg)) 
    time.sleep(1)