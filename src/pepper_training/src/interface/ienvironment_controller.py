from abc import ABC, abstractmethod

class IEnvController(ABC):
  @property
  @abstractmethod
  def get_action_and_state_size(self):
    pass

  @property  
  @abstractmethod
  def _get_reward(self):
    pass

  @property  
  @abstractmethod
  def _get_done(self):
    pass
  
  @abstractmethod
  def step(self):
    """
    return next_state, reward, and done.
    """
    pass