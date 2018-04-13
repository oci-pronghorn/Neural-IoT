"""
Team Neural Net of IoT Devices
Script to create trained nets to match timing script
2018 SIUE Senior Project
"""

import time
import subprocess
import sys

#num layers
for i in range(3,11):
    #layer width
    for j in range(3,11):
        start = time.time()
        #subprocess.call(['java', '-jar', 'target/' + sys.argv[1]], "-n", j, "-l", i, "-win", "OUTPUT-weights" + i + j, -"bin", "OUTPUT-biases" + i + j) 
        end = time.time()

        print("Execution time of net was " + str(end - start) + " with " + str(i) +
        " layers, " + str(j) + " nodes per layer, and " + str(i * j) + " total nodes.") 
