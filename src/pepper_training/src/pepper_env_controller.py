import nep

from interface.ienvironment_controller import IEnvController


class PepperEnvController(IEnvController):
  def __init__(self):
    super().__init__()
    # Create a new nep node
    node = nep.node("Calculator")                                                       
    conf = node.hybrid("192.168.11.62")                         
    # conf = node.hybrid("192.168.3.14")                         
    self.sub = node.new_sub("env", "json", conf)
    self.pub = node.new_pub("calc", "json", conf)
    
  def _get_msg(self):
    while True:
      s, msg = self.sub.listen()
      if s:
        print(msg)
        return msg
  
  def get_action_and_state_size(self):
    action_size, state_size = self._get_msg().values()
    return action_size, state_size
  
  def get_state(self):
    state = self._get_msg()['state']
    return state
  
  def publish_action(self, action):
    print("ACTION", action.cpu().tolist())
    self.pub.publish({'action': action.cpu().tolist()})  # need tolist for sending message as json
  
  def step(self):
    """
    return next_state, reward, and done.
    """
    msg = self._get_msg()
    next_state = msg['next_state']
    reward = msg['reward']
    done = msg['done']
    return next_state, reward, done
  
  def _get_reward(self):
    pass

  
  def _get_done(self):
    pass