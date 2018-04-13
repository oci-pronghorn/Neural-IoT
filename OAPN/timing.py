"""
Team Neural Net of IoT Devices
Test Script for Collecting Neural Net runtime Data on Raspberry Pi
2018 SIUE Senior Project
"""

import subprocess
import sys
import threading
import time
import os

minTime = 1000
bestNumLayers = 0
bestLayerWidth = 0
sleepTime = 3


def javakill():
    time.sleep(sleepTime)
    os.system("killall java")

if(len(sys.argv) < 2):
    print "Specify jar to time"
    sys.exit()


#Vary network size

#num layers
for i in range(3,11):
    #layer width
    for j in range(3,11):
        #spin off thread to end pronghorn because it stalls when done
        thread = threading.Thread(target=javakill, args=())
        thread.start()        


        subprocess.call(['java', '-jar', 'target/' + sys.argv[1], "-n", str(j), "-l", str(i), "-testing", "-win", "OUTPUT-weights" + str(i) + str(j), "-bin", "OUTPUT-biases" + str(i) + str(j)])        

"""   
        print("Execution time of net was " + str(end - start) + " with " + str(i) +
        " layers, " + str(j) + " nodes per layer, and " + str(i * j) + " total nodes.") 

    if ((end - start) < minTime):
        minTime = (end - start)
        bestNumLayers = i
        bestLayerWidth = j

#Which size performed best?
print "\nThe best number of layers was " + str(bestNumLayers)
print "The best layer width was " + str(bestLayerWidth)
print "The time for this combo was " + str(minTime) + " seconds."

#Use most performant network size from above to classify many test examples, then find avg

avg = 0;
numToTest = 1000
for i in (0,numToTest):
    start = time.time()
    #subprocess.call(['java', '-jar', 'target/' + sys.argv[1]], "-n", bestLayerWidth, "-l", bestNumLayers, "-testing", "-win", "fn", -"bin", "fn") 
    end  = time.time()
    avg += (end - start)

avg = avg / numToTest
print "\nThe average classification time for " + str(numToTest) + " test examples was " + str(avg) + " seconds."
"""

