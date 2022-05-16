#!/usr/bin/env python
import sys

from information import Information

file_name = sys.argv[1] if len(sys.argv)==2 else 'reward.csv'

information = Information(file_name)
information.plot('reward')


