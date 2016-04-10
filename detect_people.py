import numpy as np
import cv2
import sys
from imutils.object_detection import non_max_suppression

video = cv2.VideoCapture(sys.argv[1])

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


out = cv2.VideoWriter(sys.argv[2], cv2.cv.CV_FOURCC('M','J','P','G'), 30.0, (640,480), True)



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
		if count % 5 == 0:
			(rects, weights) = hog.detectMultiScale(frame, winStride=(4,4), padding=(8,8), scale=1.10)
	
			rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
			pick = non_max_suppression(rects, probs = None, overlapThresh = 0.35)

		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(frame, (xA,yA), (xB,yB), (0,255,0), 2)

		

		out.write(frame)

	else:
		break

	count += 1
#end loop

video.release()
out.release()

