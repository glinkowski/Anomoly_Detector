import numpy as np
import cv2
import sys
from imutils.object_detection import non_max_suppression
from math import sqrt

def distance(cen1, cen2):
	return math.sqrt((cen1[0] - cen2[0])**2 + (cen1[1] - cen2[1])**2)


video = cv2.VideoCapture(sys.argv[1])

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


out = cv2.VideoWriter(sys.argv[2], cv2.cv.CV_FOURCC('M','J','P','G'), 30.0, (640,480), True)

people_dict = {}

count = 0
while video.isOpened():
	if(count < 450): # used for testing, skipping initial videos
		count += 1
		continue
	if not (count % 25) :
		print("processed: {}".format(count))
	#end if

	ret, frame = video.read()
	frame = cv2.resize(frame, (640,480))
	if ret:
		if count % 5 == 0: # process every 5 frames for speed up
			(rects, weights) = hog.detectMultiScale(frame, winStride=(4,4), padding=(8,8), scale=1.10)
	
			rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
			pick = non_max_suppression(rects, probs = None, overlapThresh = 0.35)
			
			# extract motion vectors if not first frame
			if count != 0:
				for (xA, yA, xB, yB) in pick:
					dist_arr = {}
					centroid = ((xA+xB)/2, (yA + yB)/2)
					dist_thres = 1e10
					for cen2 in people_dict:
						if distance(centroid,cen2) < dist_thres:
							 curr_cen = cen2
			else:
				for i in range(0, len(pick)):
					people_dict[i] = ((pick[i][0] + pick[i][2])/2, (pick[i][1]+pick[i][3])/2)
					


			prev_pick = pick
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(frame, (xA,yA), (xB,yB), (0,255,0), 2)

		

		out.write(frame)

	else:
		break

	count += 1
#end loop

video.release()
out.release()

