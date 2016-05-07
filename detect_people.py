# ---------------------------------------------------------
# authors: Mohammad Saad & Greg Linkowski
# project: Anomoly Detection
#		for Computer Vision final project
# 
# After background subtraction, run HOG to detect people.
#	Track those people to create motion vectors. Save the
#	video with rectangles overlaid. Save the vectors to
#	a text file (x, y, magnitude, angle).
#
# 
# Use ----
#	...$ python detect_people.py <input video>
#
#
# Input ----
#	processed video of location (background removed)
#
# Output ----
#	<inVid name>-tracked.avi
#			processed video with rectangles overlaid
#	<inVid name>-vectors.txt
#			text file of vectors
# ---------------------------------------------------------

import numpy as np
import cv2
import sys
from imutils.object_detection import non_max_suppression
from math import sqrt
import time



####### ####### ####### ####### 
# Helper Function Declarations

def distance(cen1, cen2):
	return sqrt((cen1[0] - cen2[0])**2 + (cen1[1] - cen2[1])**2)

def magnitude(vec):
	return sqrt(vec[0]**2 + vec[1]**2)


def centroid_calc(rect):
	return ((rect[0] + rect[2])/2, (rect[1]+rect[3])/2)

####### ####### ####### ####### 



####### ####### ####### ####### 
# PARAMETERS

# Set slower FPS -- skip X frames per second
outFPS = 2		# the final fps (should divide 30)

# Output video compression code
compression = cv2.cv.CV_FOURCC('M','P','4','V')
#compression = cv2.cv.CV_FOURCC('M','J','P','G')

# Limit the maximum size of the detections
#   ie: if rect height > rectThresh, ignore
rectThresh = 200

# Output color of rectangles drawn on video
rectColor = (0,255,0)


####### ####### ####### ####### 



####### ####### ####### ####### 
# BEGIN MAIN FUNCTION
print("")

tstart = time.time()



inVid = cv2.VideoCapture(sys.argv[1])
inWidth = int(inVid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
inHeight = int(inVid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))


hog = cv2.HOGDescriptor()
#hog = cv2.HOGDescriptor(nLevels=4)
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


#outFPS = 6		# how many fps to keep
outPrefix = sys.argv[1][0:-4]
#outPrefix = sys.argv[1].split('-')[0]
outVName = outPrefix + '-tracking.avi'
out = cv2.VideoWriter(outVName, compression, outFPS, (inWidth,inHeight), True)
#out = cv2.VideoWriter(sys.argv[2], cv2.cv.CV_FOURCC('M','J','P','G'), 30.0, (640,480), True)


people_dict = {}
prev_pick = []
people_history  = {}
count = 0
skipCount = 0
skipStop = int(round(30 / outFPS))
#motion_vectors = []
mvList = list()
mvRow = list()
while inVid.isOpened():

	# Skipping so many frames per second
	if (skipCount < skipStop) :
		ret, frame = inVid.read()
		skipCount += 1
		continue
	skipCount = 0

#TODO: Skip first 2-3 seconds (adjust for outFPS)
	# # Skip initial frames (as bgsubtract sets up)
	#  if(count < 30): 
	#  	ret, frame = video.read()
	#  	count += 1
	#  	continue

	if not (count % 10) :
		print("processed: {} --------".format(count))
	#end if

	ret, frame = inVid.read()
	
	if ret:

#		frame = cv2.resize(frame, (640,480))

		# (rects, weights) = hog.detectMultiScale(frame,
		# 	winStride=(4,4), padding=(8,8), scale=1.10)
		(rects, weights) = hog.detectMultiScale(frame,
			winStride=(4,4), padding=(8,8), scale=1.05)


#		rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
		rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects if (h < rectThresh)])
		pick = non_max_suppression(rects, probs = None, overlapThresh = 0.65)
		
		# extract motion vectors if there are previous picks

		# print len(prev_pick)
		if(len(prev_pick) != 0):
			for i in range(0, len(pick)):
				

				# initialize distance array
				dists = []

				# calculate centroid of current person
				center = centroid_calc(pick[i])

				print("  center: {0},{1}".format(center[0], center[1]))
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
#					motion_vectors.append(displacement)
				
				# append centroid
#					motion_vectors.append(center)

				# calculate magnitude
				mag = magnitude(displacement)
#					motion_vectors.append(mag)

				# calculate angle
				angle = np.arctan2(displacement[1], displacement[0])
#					motion_vectors.append(angle)

				# delete previous value, as we have found its match
				np.delete(prev_pick, min_rect)

				# Save this motion vector to list
				mvRow = [prev_center[0], prev_center[1], mag, angle, center[0], center[1]]
				mvList.append(mvRow)

		# set current to previous
		prev_pick = pick
		# to save time while running, comment out
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(frame, (xA,yA), (xB,yB), rectColor, 2)


		out.write(frame)

	else:
		break

	count += 1
#end loop

inVid.release()
out.release()

#print motion_vectors
#print mvList

outFName = outPrefix + '-vectors.txt'
with open(outFName, 'w') as fout :
	firstLine = True
	for row in mvList :
		if firstLine == True :
			firstLine = False
		else :
			fout.write("\n")
		#end if
		fout.write("{}\t{}\t{}\t{}\t{}\t{}".format(row[0],
			row[1], row[2], row[3], row[4], row[5]))
#end with

# outFName = outPrefix + '-vectors.txt'
# #with open(sys.argv[3], 'w') as f:
# with open(outFName, 'w') as f:
# 	for i in range(0, len(motion_vectors)):
# 		f.write("{0},{1},{2},{3},{4}\n".format(motion_vectors[i][0], motion_vectors[i][1],motion_vectors[i][2],motion_vectors[i][3],motion_vectors[i][4]))
# f.close()


ttotal = (time.time() - tstart)/60
print("\n    {0.3f} minutes".format(ttotal))

print("\nDone.\n")