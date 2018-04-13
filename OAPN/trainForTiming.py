"""
Team Neural Net of IoT Devices
Script to create trained nets to match timing script
2018 SIUE Senior Project
Usage Note: if Neural Net being tested fails to generate output files, especially for larger net sizes, increase sleepTime 
"""

import time
import subprocess
import sys
import os
import threading

sleepTime = 15

def my_threaded_func():
    time.sleep(sleepTime)
    os.system("killall java")

#num layers
for i in range(3,10):
    #layer width
    for j in range(3,10):
        #print ('java', '-jar', 'target/' + sys.argv[1], "-n", str(j), "-l", str(i), "-wout", "OUTPUT-weights" + str(i) + str(j), "-bout", "OUTPUT-biases" + str(i) + str(j))

        #spin off thread to end pronghorn because it stalls when done
        thread = threading.Thread(target=my_threaded_func, args=())
        thread.start()

        subprocess.call(['java', '-jar', 'target/' + sys.argv[1], "-n", str(j), "-l", str(i), "-wout", "OUTPUT-weights" + str(i) + str(j), "-bout", "OUTPUT-biases" + str(i) + str(j)]) 
        
        #os.system("cat OUTPUT | grep -c x")
        #print "were classified incorrectly out of"
        #os.system("wineTraining.data | wc -l")
        #print ""
