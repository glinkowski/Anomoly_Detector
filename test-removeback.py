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

# if isColor is False, convert to grayscale
isColor = True

# number of initial frames to discard
numThrow = 120

# size of the morphology kernel
kernSize = 3

# resize the video ?
resize = True
rWidth = 640
rHeight = 480

####### ####### ####### ####### 



####### ####### ####### ####### 
# BEGIN MAIN FUNCTION
print("")


# Open input video and get frame rate
inVid = cv2.VideoCapture(sys.argv[1])
fps = inVid.get(cv2.cv.CV_CAP_PROP_FPS)
print("original video fps = {}".format(fps))
# used with % for output in processing loop :
fps = int(round(fps))

inWidth = int(inVid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
inHeight = int(inVid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
print("original video resolution = {}x{}".format(inWidth, inHeight))


# Initialize output video (for linux/mac, h.264 code is MP4V)
#outVid1 = cv2.VideoWriter('rmBack01.avi', cv2.cv.CV_FOURCC('M','J','P','G'), 30, (640,480), False)
#outVid2 = cv2.VideoWriter('rmBack02.avi', cv2.cv.CV_FOURCC('M','J','P','G'), 30, (640,480), isColor)

if resize :
	outVid1 = cv2.VideoWriter('rmBack01.avi', cv2.cv.CV_FOURCC('M','P','4','V'), 30, (rWidth,rHeight), False)
	outVid2 = cv2.VideoWriter('rmBack02.avi', cv2.cv.CV_FOURCC('M','P','4','V'), 30, (rWidth,rHeight), isColor)
else :
	outVid1 = cv2.VideoWriter('rmBack01.avi', cv2.cv.CV_FOURCC('M','P','4','V'), 30, (inWidth,inHeight), False)
	outVid2 = cv2.VideoWriter('rmBack02.avi', cv2.cv.CV_FOURCC('M','P','4','V'), 30, (inWidth,inHeight), isColor)
#end if	

# Throw out first N frames
count = 0
while count <= numThrow:
	# Get a frame
	ret, frame = inVid.read()
	count += 1
#end loop



#bgnd = cv2.BackgroundSubtractorMOG()
bgnd = cv2.BackgroundSubtractorMOG2()
morphKern = np.ones((kernSize, kernSize),np.uint8)

count = 0
while inVid.isOpened():
	count += 1
	if not (count % (fps * 2)) :
		print("processed: {}".format(count))

	# Get the frame
	ret, frame = inVid.read()


	if ret:
		# Resize to smaller scale
		if resize :
			frame = cv2.resize(frame, (640,480))
		# Convert to grayscale
		if not isColor :
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		#end if

		bgmask = bgnd.apply(frame) 
		bgopen = cv2.morphologyEx(bgmask, cv2.MORPH_OPEN, morphKern)

		outVid1.write(bgmask)
#		outVid1.write(bgopen)

		# Scale bgmask to [0, 1] to black out frame
#		bgmask = np.divide(bgmask, 255)
		bgopen = np.divide(bgopen, 255)
		if isColor :
			frame2 = np.zeros(frame.shape, dtype=np.uint8)
			for i in xrange(3) :
#				frame2[:,:,i] = np.multiply(frame[:,:,i], bgmask)
				frame2[:,:,i] = np.multiply(frame[:,:,i], bgopen)
		else :
#			frame2 = np.multiply(frame, bgmask)
			frame2 = np.multiply(frame, bgopen)
		#end if
		outVid2.write(frame2)

	else :
		break
	#end if

#end loop

inVid.release()
outVid1.release()
outVid2.release()


print("\nDone.\n")