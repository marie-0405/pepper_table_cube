import unittest
from result_controller import ResultController

class TestResultController(unittest.TestCase):
  def test_write(self):
    q_matrix = "q_matrix"
    rewards = [1, 2, 3]
    succeeds = [False, False, False]
    result_controller = ResultController("test")
    result_controller.write(rewards, succeeds, q_matrix)

def main():
  TestResultController.test_write()

if __name__ == '__main__':
  unittest.main()
