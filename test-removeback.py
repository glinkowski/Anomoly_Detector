# ---------------------------------------------------------
# author: Greg Linkowski
# project: Anomoly Detection
#		for Computer Vision final project
# 
# Testing the idea of subtracting out the background
#	NOTE: the background should be static
# 
# Creates ----
#	rmBack01.avi: the binary back/fore-ground mask
#	rmBack02.avi: the masked video
# ---------------------------------------------------------

import cv2
import sys
import numpy as np



####### ####### ####### ####### 
# PARAMETERS

isColor = False

####### ####### ####### ####### 



####### ####### ####### ####### 
# BEGIN MAIN FUNCTION
print("")


inVid = cv2.VideoCapture(sys.argv[1])

outVid1 = cv2.VideoWriter('rmBack01.avi', cv2.cv.CV_FOURCC('M','J','P','G'), 30.0, (640,480), isColor)
outVid2 = cv2.VideoWriter('rmBack02.avi', cv2.cv.CV_FOURCC('M','J','P','G'), 30.0, (640,480), isColor)

bgnd = cv2.BackgroundSubtractorMOG()
count = 0
while inVid.isOpened():
	count += 1
	if not (count % 25) :
		print("processed: {}".format(count))

	# Get the frame
	ret, frame = inVid.read()


	if ret:
		# Resize to smaller scale
		frame = cv2.resize(frame, (640,480))
		# Convert to grayscale
		if not isColor :
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		#end if

		bgmask = bgnd.apply(frame)

		outVid1.write(bgmask)

		bgmask = np.divide(bgmask, 255)
#		print np.max(bgmask), np.min(bgmask)
		frame2 = np.multiply(frame, bgmask)
		outVid2.write(frame2)

	else :
		break
	#end if

#end loop

inVid.release()
outVid1.release()
outVid2.release()


print("\nDone.\n")