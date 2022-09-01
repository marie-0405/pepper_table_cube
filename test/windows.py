import nep
import time

# Create a new nep node
node = nep.node("receiver")     
# Set a direct connection using port <3000> and in <'one2many'> mode
# Important: 	You need to change the IP address <'127.0.0.1'> by 
# 		the IP address of you PC running the publisher node                                                   
conf = node.hybrid("192.168.0.101")                         
# Create a new nep subscriber with the topic <'test'>
sub = node.new_sub("test", "json", conf) 

while True:
    # Read data in a non-blocking mode
    s, msg = sub.listen() 
    # if s == True, then there is data in the socket      
    if s:                       
        print(msg)
    else:
        # An small sleep will reduce computational load
        time.sleep(.0001)