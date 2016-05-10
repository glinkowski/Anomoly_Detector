# ---------------------------------------------------------
# authors: Mohammad Saad & Greg Linkowski
# project: Anomoly Detection
#		for Computer Vision final project
# 
# Apply the svm to a -vectors.txt file, and draw the
#	classified vectors in red & green on an image from
#	the video.
#
# 
# Use ----
#	...$ python detect_people.py <input video>
#
#
# Input ----
#	video on which to overlay vectors
#	NOTE: classifier should be contained in this folder
#
# Output ----
#	image with vectors overlaid
# ---------------------------------------------------------

import cv2
import numpy as np
import sys
from os import listdir
#import random
from sklearn import svm
from sklearn.externals import joblib
import math


####### ####### ####### ####### 
# PARAMETERS

# subdirectory where classifier is stored
svmFolder = 'SVM/classifier.pkl'

# Vectors longer than this will be ignored
mvThreshold = 500

####### ####### ####### ####### 



####### ####### ####### ####### 
# BEGIN MAIN FUNCTION
print("")


# Define the path/names for the required elements
if len(sys.argv) < 2 :
	print("ERROR: $ python pre-rmBack.py <video path & name>")
#end if

inVidFile = sys.argv[1]
mvFile = inVidFile[0:-4] + '-vectors.txt'
inDir = inVidFile
tchar = inVidFile[-1]
while tchar is not '/' :
	inDir = inDir[0:-1]
	tchar = inDir[-1]
#end loop
svmFile = inDir + svmFolder



# Skip first N frames of video, then keep a still
print(inVidFile)
inVid = cv2.VideoCapture(inVidFile)
numSkip = 4
for i in xrange(numSkip) :
	ret, stillFrame = inVid.read()
ret, stillFrame = inVid.read()


# Read in the vectors
mVectors = np.zeros((0,5), dtype=np.float64)
with open(mvFile, 'r') as fin :
	for line in fin :
		line = line.rstrip()
		lv = line.split('\t')
		if (float(lv[2]) < mvThreshold) : 
			row = ([ float(lv[0]), float(lv[1]),
				float(lv[2]), float(lv[3]), 0 ])
			mVectors = np.vstack((mVectors, row))
#end with
#print(mVectors)


# Call forth the classifier
cfier = joblib.load(svmFile)

for i in xrange(mVectors.shape[0]) :
	mVectors[i,4] = cfier.predict(mVectors[i,0:4])
#	print cfier.predict(mVectors[i,0:4])
#	print mVectors[i,0:4], mVectors[i,4]
#end loop
#print(mVectors)


# Draw the vectors onto the still
# Draw a circle at the origin, and a line to the destination
for i in xrange(mVectors.shape[0]) :

	# if True, use green; false, use red
	if mVectors[i,4] :
		mvColor = (0, 255, 0)
	else :
		mvColor = (0, 0, 255)
	#end if

	x1 = int(mVectors[i,0])
	y1 = int(mVectors[i,1])

#	print int(mVectors[i,0]), int(mVectors[i,1])
	cv2.circle(stillFrame, (x1, y1), 6, mvColor, 6)

	x2 = x1 + (mVectors[i,2] * math.cos(mVectors[i,3]))
	x2 = int(round(x2))
	y2 = y1 + (mVectors[i,2] * math.sin(mVectors[i,3]))
	y2 = int(round(y2))
	cv2.line(stillFrame, (int(mVectors[i,0]), int(mVectors[i,1])),
		(x2, y2), mvColor, 4)
#end with


# Display image (buggy)
cv2.imshow("still", stillFrame);
cv2.waitKey(0)
cv2.destroyAllWindows()


# Output the image
#cv2.imwrite( inVidDir + inVidName.split('.')[0] + '-still.png', stillFrame )
cv2.imwrite( inVidFile[0:-4] + '-labeled.png', stillFrame )


# Save the vectors to file
outFName = inVidFile[0:-4] + '-vLabelled.txt'
with open(outFName, 'w') as fout :
	firstLine = True
	for row in mVectors :
		if firstLine == True :
			firstLine = False
		else :
			fout.write("\n")
		#end if
		fout.write("{}\t{}\t{}\t{}\t{}".format(row[0],
			row[1], row[2], row[3], row[4]))
#end with


print("\nDone.\n")