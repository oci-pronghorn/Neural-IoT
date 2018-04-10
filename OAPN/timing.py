"""
Team Neural Net of IoT Devices
Test Script for Collecting Neural Net runtime Data on Raspberry Pi
2018 SIUE Senior Project
"""

import time
import subprocess
import sys

if(len(sys.argv) < 2):
	print "Specify jar to time"
	sys.exit()

start = time.time()

subprocess.call(['java', '-jar', 'target/' + sys.argv[1]])

end = time.time()
print "Execution time of net was " + str(end - start)
