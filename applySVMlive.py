# ---------------------------------------------------------
# authors: Mohammad Saad & Greg Linkowski
# project: Anomoly Detection
#		for Computer Vision final project
# 
# Apply the svm to a video file, and draw the
#	classified vectors in red & green onto the output.
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
from imutils.object_detection import non_max_suppression
#from math import sqrt
import math
import time




####### ####### ####### ####### 
# Helper Function Declarations

def distance(cen1, cen2):
	return math.sqrt((cen1[0] - cen2[0])**2 + (cen1[1] - cen2[1])**2)

def magnitude(vec):
	return math.sqrt(vec[0]**2 + vec[1]**2)


def centroid_calc(rect):
	return ((rect[0] + rect[2])/2, (rect[1]+rect[3])/2)

####### ####### ####### ####### 



####### ####### ####### ####### 
# PARAMETERS

# subdirectory where classifier is stored
svmFolder = 'SVM/'
# whether to use the background subtracted video
useVMod = True
vModExt = '-rmBack02-k4.avi'
vMOutExt = '-rmBack.avi'

# Vectors longer than this will be ignored
mvThreshold = 300
# Limit the maximum size of the detections
rectThresh = 500	# quad = 200

# Set slower FPS -- skip X frames per second
outFPS = 3		# the final fps (should divide 30)
# Output video compression code
compression = cv2.cv.CV_FOURCC('M','P','4','V')
# Output color of vectors/rectangles drawn on video
cPos = (0,255,0)
cNeg = (0, 0, 255)
cRect = (0, 192, 255)
# Number of frames to display vector
tPos = 3
tNeg = 7

####### ####### ####### ####### 



####### ####### ####### ####### 
# BEGIN MAIN FUNCTION
print("")

tstart = time.time()


# Define the path/names for the required elements
if len(sys.argv) < 2 :
	print("ERROR: $ python pre-rmBack.py <video path & name>")
#end if

vOrigFile = sys.argv[1]
vModFile = vOrigFile[0:-4] + vModExt
vMOutFile = vOrigFile[0:-4] + vMOutExt
#print(vModFile)
inDir = vOrigFile
tchar = vOrigFile[-1]
while tchar is not '/' :
	inDir = inDir[0:-1]
	tchar = inDir[-1]
#end loop
svmFile = inDir + svmFolder + 'classifier.pkl'



# Call forth the classifier
cfier = joblib.load(svmFile)


# Get the input video(s)
if useVMod :
	vMod = cv2.VideoCapture(vModFile)
	vOrig = cv2.VideoCapture(vOrigFile)
	print("original video fps = {}".format(
		vOrig.get(cv2.cv.CV_CAP_PROP_FPS)))
	print("bg subtr video fps = {}".format(
		vMod.get(cv2.cv.CV_CAP_PROP_FPS)))
else :
	vOrig = cv2.VideoCapture(vOrigFile)
#end if
inWidth = int(vOrig.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
inHeight = int(vOrig.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))


# Prep the output video
vOutFile = sys.argv[1][0:-4] + '-final.avi'
print("output will be saved to: {}".format(vOutFile))
vOut = cv2.VideoWriter(vOutFile, compression, outFPS, (inWidth, inHeight), True)
if useVMod :
	vmOut = cv2.VideoWriter(vMOutFile, compression, outFPS, (inWidth, inHeight), True)
#end if


# Prep the people detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# Declare some stuff
people_dict = {}
prev_pick = []
people_history  = {}
count = 0
skipCount = 0
skipStop = int(round(30 / outFPS))
#motion_vectors = []
mvList = list()
mvRow = list()


# Step through the input video
while vOrig.isOpened():

	# Skipping so many frames per second
	if (skipCount < skipStop) :
		ret, frame = vOrig.read()
		if useVMod :
#			ret, frame = vOrig.read()
			ret, frame = vMod.read()
		skipCount += 1
		continue
	skipCount = 0


	if not (count % 10) :
		print("processed: {} --------".format(count))
	#end if


	if useVMod :
		rOrig, fOrig = vOrig.read()
		rMod, fMod = vMod.read()
		if not rOrig :
			print("ran out of original video")
			break
		if not rMod :
			print("ran out of modified video")
			break
	else :
		rOrig, fOrig = vOrig.read()
		if not rOrig :
			print("ran out of video")
			break
	#end if

	
	# frame(s) exist ... track people
	if useVMod :
		(rects, weights) = hog.detectMultiScale(fMod,
			winStride=(4,4), padding=(8,8), scale=1.05)
	else :
		(rects, weights) = hog.detectMultiScale(fOrig,
			winStride=(4,4), padding=(8,8), scale=1.05)
	rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects if (h < rectThresh)])
	pick = non_max_suppression(rects, probs = None, overlapThresh = 0.65)
	

	# Extract motion vectors where there are previous picks
	if(len(prev_pick) != 0):
		for i in range(0, len(pick)):
			
			# initialize distance array
			dists = []
			# calculate centroid of current person
			center = centroid_calc(pick[i])
#			print("  center: {0},{1}".format(center[0], center[1]))
			# if no other picks left in prev_pick, skip (i.e. we have a new person)
			if(len(prev_pick) == 0):
				continue
			
			# loop through prev pick, calculate centroids and distances
			for j in range(0, len(prev_pick)):
				prev_center = centroid_calc(prev_pick[j])
				dists.append(distance(center, prev_center))
			# find prev square with minimum distance
			dists = np.array(dists)
			min_rect = np.argmin(dists)

			# find motion vector
			displacement = (center[0] - centroid_calc(prev_pick[min_rect])[0],center[1] - centroid_calc(prev_pick[min_rect])[1])
			# calculate magnitude
			mag = magnitude(displacement)
			# calculate angle
			angle = np.arctan2(displacement[1], displacement[0])

			# delete previous value, as we have found its match
			np.delete(prev_pick, min_rect)


			# Determine if this is Pos or Neg vector
			#	and save to list
			mvRow = ([prev_center[0], prev_center[1], mag,
				angle, 0, 0])
			if (mvRow[2] < mvThreshold) :
				mvClass = cfier.predict(mvRow[0:4])
				mvRow[4] = mvClass
				mvList.append(mvRow)
			#end if

# #			# Save this motion vector to list
# #			mvRow = [prev_center[0], prev_center[1], mag, angle, center[0], center[1]]
# #			if mvRow[2] < mvThreshold :
# #				mvList.append(mvRow)

# 			mVectors = np.zeros((1,5))
# 			mVectors[i,4] = cfier.predict(mVectors[i,0:4])
# #TODO: THIS.
# 			# Determine if this is Pos or Neg vector
# 			#	and save to list
# 			# col 4 = class, col 5 = timer
# 			mvRow = [prev_center[0], prev_center[1], mag, angle, 0, 0]
# #			class = cfier.predict(mvRow[0:4])
# 			if (mvRow[2] < mvThreshold) :
# 				class = 1
# #				class = cfier.predict(mvRow[0:4])
# #				class = cfier.decision_function(mvRow[0:4])
# 				mvRow[4] = class
# 				mvList.append(mvRow)


	# set current to previous
	prev_pick = pick

	# Draw boxes around detected people
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(fOrig, (xA,yA), (xB,yB), cRect, 2)


	# Draw the appropriate vectors in frame
	for row in mvList :
		if (row[4] == 1) and (row[5] < tPos) :
			x1 = int(row[0])
			x2 = x1 + int(row[2] * math.cos(row[3]))
			y1 = int(row[1])
			y2 = y1 + int(row[2] * math.sin(row[3]))
			cv2.circle(fOrig, (x1, y1), 6, cPos, 6)
			cv2.line(fOrig, (x1, y1), (x2, y2), cPos, 4)
			if useVMod :
				cv2.circle(fMod, (x1, y1), 6, cPos, 6)
				cv2.line(fMod, (x1, y1), (x2, y2), cPos, 4)
		elif (row[4] == 0) and (row[5] < tNeg) :
			x1 = int(row[0])
			x2 = x1 + int(row[2] * math.cos(row[3]))
			y1 = int(row[1])
			y2 = y1 + int(row[2] * math.sin(row[3]))
			cv2.circle(fOrig, (x1, y1), 6, cNeg, 6)
			cv2.line(fOrig, (x1, y1), (x2, y2), cNeg, 4)
			if useVMod:
				cv2.circle(fOrig, (x1, y1), 6, cNeg, 6)
				cv2.line(fOrig, (x1, y1), (x2, y2), cNeg, 4)
		#end if
		row[5] += 1
	#end loop

#	# increment timer
#	mvList[:,5] = np.add(mvList[:,5], 1)


	# write to output
	vOut.write(fOrig)
	if useVMod :
		vmOut.write(fMod)
	#end if

	count += 1
#end loop

vOrig.release()
vOut.release()
if useVMod :
	vMod.release()
	vmOut.release()
#end if


# Save the vectors to file
outFName = vOrigFile[0:-4] + '-vLabelled.txt'
with open(outFName, 'w') as fout :
	firstLine = True
	for row in mvList :
		if firstLine == True :
			firstLine = False
		else :
			fout.write("\n")
		#end if
		fout.write("{}\t{}\t{}\t{}\t{}".format(row[0],
			row[1], row[2], row[3], row[4]))
#end with


ttotal = (time.time() - tstart)/60
print("\n    {:1.3f} minutes".format(ttotal))

print("\nDone.\n")