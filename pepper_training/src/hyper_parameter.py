import numpy as np


class HyperParameter():
  def __init__(self, min_value, max_value, delta):
    self.values = np.arange(min_value, max_value + delta, delta)


if __name__ == '__main__':
  hyper_parameter = HyperParameter(0.2,0.8, 0.1)
  print(hyper_parameter.values)
