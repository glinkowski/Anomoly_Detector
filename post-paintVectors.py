# ---------------------------------------------------------
# author: Greg Linkowski
# project: Anomoly Detection
#		for Computer Vision final project
# 
# After collecting vectors from a video, paint those
#	vectors onto a still of the location
# 
# Input ----
#	video of original location
#	text file of vectors
#
# Output ----
#	... image of location w/ vectors on it
# ---------------------------------------------------------

import cv2
import sys
import numpy as np



####### ####### ####### ####### 
# PARAMETERS

inVidName = 'mntl01.MTS'
inVidDir = 'MNTL/'

mvSuffix = '-vectors.txt'

inVidFile = inVidDir + inVidName
mvFile = inVidDir + inVidName.split('.')[0] + mvSuffix

####### ####### ####### ####### 



####### ####### ####### ####### 
# BEGIN MAIN FUNCTION
print("")


# Skip first N frames of video, then keep a still
print(inVidFile)
inVid = cv2.VideoCapture(inVidFile)
numSkip = 4
for i in xrange(numSkip) :
	ret, stillFrame = inVid.read()
ret, stillFrame = inVid.read()


# Draw the vectors onto the still
# Draw a circle at the origin, and a line to the destination
with open(mvFile, 'r') as fin :
	for line in fin :
		line = line.rstrip()
		lv = line.split('\t')
#		print lv
		if (float(lv[2]) > 200) :	continue
		cv2.circle(stillFrame, (int(lv[0]), int(lv[1])),
			6, (64,192,0), 6)
		cv2.line(stillFrame, (int(lv[0]), int(lv[1])),
			(int(lv[4]), int(lv[5])), (64,192,0), 4)
#end with


# Output the image
cv2.imwrite( inVidDir + inVidName.split('.')[0] + '-still.png', stillFrame )


print("\nDone.\n")