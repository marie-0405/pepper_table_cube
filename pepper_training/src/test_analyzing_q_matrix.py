from result_controller import ResultController

FILE_END_NAME = 'a=0.99-g=0.9'
result_controller = ResultController(FILE_END_NAME)
print(result_controller.count_q_matrix())