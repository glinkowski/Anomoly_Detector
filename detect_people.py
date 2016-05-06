import numpy as np
import cv2
import sys
from imutils.object_detection import non_max_suppression
from math import sqrt

def distance(cen1, cen2):
	return sqrt((cen1[0] - cen2[0])**2 + (cen1[1] - cen2[1])**2)

def magnitude(vec):
	return sqrt(vec[0]**2 + vec[1]**2)


def centroid_calc(rect):
	return ((rect[0] + rect[2])/2, (rect[1]+rect[3])/2)


video = cv2.VideoCapture(sys.argv[1])

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

outPrefix = sys.argv[1].split('-')[0]
outVName = outPrefix + '-tracking.avi'
out = cv2.VideoWriter(outVName, cv2.cv.CV_FOURCC('M','J','P','G'), 30, (640,480), True)
#out = cv2.VideoWriter(sys.argv[2], cv2.cv.CV_FOURCC('M','J','P','G'), 30.0, (640,480), True)

people_dict = {}
prev_pick = []
people_history  = {}
count = 0
motion_vectors = []
while video.isOpened():
	if(count < 575): # used for testing, skipping initial videos
		ret, frame = video.read()
		count += 1
		continue
	if not (count % 25) :
		print("processed: {}".format(count))
	#end if

	ret, frame = video.read()
	
#	frame = cv2.resize(frame, (640,480))
	if ret:
		if count % 5 == 0: # process every 5 frames for speed up
			(rects, weights) = hog.detectMultiScale(frame, winStride=(4,4), padding=(8,8), scale=1.10)
	
			rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
			pick = non_max_suppression(rects, probs = None, overlapThresh = 0.65)
			
			# extract motion vectors if there are previous picks

			# print len(prev_pick)
			if(len(prev_pick) != 0):
				for i in range(0, len(pick)):
					

					# initialize distance array
					dists = []

					# calculate centroid of current person
					center = centroid_calc(pick[i])

					# print center

					print "center: {0},{1}".format(center[0], center[1])
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
					motion_vectors.append(displacement)
					
					# append centroid
					motion_vectors.append(center)

					# calculate magnitude
					mag = magnitude(displacement[0], displacement[1])
					motion_vectors.append(mag)

					# calculate angle
					angle = np.arctan2(displacement[1], displacemnet[0])
					motion_vectors.append(angle)

					# delete previous value, as we have found its match
					np.delete(prev_pick, min_rect)

		# set current to previous
		prev_pick = pick
		# to save time while running, comment out
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(frame, (xA,yA), (xB,yB), (0,255,0), 2)

		

		out.write(frame)

	else:
		break

	count += 1
#end loop

video.release()
out.release()

outFName = outPrefix + '-vectors.txt'
#with open(sys.argv[3], 'w') as f:
with open(outFName, 'w') as f:
	for i in range(0, len(motion_vectors)):
		f.write("{0},{1},{2},{3},{4}\n".format(motion_vectors[i][0], motion_vectors[i][1],motion_vectors[i][2],motion_vectors[i][3],motion_vectors[i][4]))
f.close()