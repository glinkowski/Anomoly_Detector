# ---------------------------------------------------------
# authors: Mohammad Saad & Greg Linkowski
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
import math



####### ####### ####### ####### 
# PARAMETERS

# # Source file name (assuming all pre-proc is done)
# inVidName = 'mntl02.MTS'
# inVidDir = 'MNTL/'

# Vectors longer than this will be ignored
mvThreshold = 2000

# Color to draw the vectors on the frame
#drawColor = (128, 255, 64)
#drawColor = (180, 20, 200)
#drawColor = (255, 0, 255)
mvColor = (0, 192, 255)


# mvSuffix = '-vectors.txt'
# inVidFile = inVidDir + inVidName
# mvFile = inVidDir + inVidName.split('.')[0] + mvSuffix

####### ####### ####### ####### 



####### ####### ####### ####### 
# BEGIN MAIN FUNCTION
print("")


if len(sys.argv) < 2 :
	print("ERROR: $ python pre-rmBack.py <video path & name>")
#end if

inVidFile = sys.argv[1]
mvFile = inVidFile[0:-4] + '-vectors.txt'


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
		if (float(lv[2]) > mvThreshold) :	continue
		cv2.circle(stillFrame, (int(lv[0]), int(lv[1])),
			6, mvColor, 6)
		x2 = int(lv[0]) + float(lv[2])*math.cos(float(lv[3]))
		x2 = int(round(x2))
		y2 = int(lv[1]) + float(lv[2])*math.sin(float(lv[3]))
		y2 = int(round(y2))
#		print x2, y2
		cv2.line(stillFrame, (int(lv[0]), int(lv[1])),
			(x2, y2), mvColor, 4)
		# cv2.line(stillFrame, (int(lv[0]), int(lv[1])),
		# 	(int(lv[4]), int(lv[5])), (64,192,0), 4)
#end with


# Display image (buggy)
cv2.imshow("still", stillFrame);
cv2.waitKey(0)
cv2.destroyAllWindows()


# Output the image
#cv2.imwrite( inVidDir + inVidName.split('.')[0] + '-still.png', stillFrame )
cv2.imwrite( inVidFile[0:-4] + '-still.png', stillFrame )


print("\nDone.\n")