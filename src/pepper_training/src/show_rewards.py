#!/usr/bin/env python
import sys

from result_controller import ResultController

file_name_end = sys.argv[1] if len(sys.argv)==2 else ''

result_controller = ResultController(file_name_end)
result_controller.plot('reward')


