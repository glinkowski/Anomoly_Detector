# ---------------------------------------------------------
# authors: Mohammad Saad & Greg Linkowski
# project: Anomoly Detection
#		for Computer Vision final project
# 
# Testing the idea of subtracting out the background
#	NOTE: the background should be static
# 
# Use ----
#	...$ python pre-rmBack.py <input video>
#
# Output ----
#	<invideo>-rmBack01.avi: the binary back/fore-ground mask
#	<invideo>-rmBack02-k<kernel>.avi: the masked video
# ---------------------------------------------------------

import cv2
import sys
import numpy as np



####### ####### ####### ####### 
# PARAMETERS

# time to discard at the beginning (in min)
timeThrow = 0.0
# time to keep in final video (in sec)
timeKeep = 0.0

# size of the morphology kernel
kernSize = 5

# if isColor is False, convert to grayscale
isColor = True
# resize the video ?
resize = False
rWidth = 640
rHeight = 480

compression = cv2.cv.CV_FOURCC('M','P','4','V')
#compression = cv2.cv.CV_FOURCC('M','J','P','G')
####### ####### ####### ####### 



####### ####### ####### ####### 
# BEGIN MAIN FUNCTION
print("")


# Open input video and get frame rate
inVid = cv2.VideoCapture(sys.argv[1])
fps = inVid.get(cv2.cv.CV_CAP_PROP_FPS)
print("original video fps = {}".format(fps))
# used with % for output in processing loop :
if fps < 121 :
	fps = int(round(fps))
else :	# to handle fps = NaN
	fps = 30
#end if

inWidth = int(inVid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
inHeight = int(inVid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
print("original video resolution = {}x{}".format(inWidth, inHeight))


# Initialize output video (for linux/mac, h.264 code is MP4V)
#outVid1 = cv2.VideoWriter('rmBack01.avi', cv2.cv.CV_FOURCC('M','J','P','G'), 30, (640,480), False)
#outVid2 = cv2.VideoWriter('rmBack02.avi', cv2.cv.CV_FOURCC('M','J','P','G'), 30, (640,480), isColor)

outPrefix = sys.argv[1].split('.')[0]
#print sys.argv[1].split('.')
#print outPrefix
outName1 = outPrefix + '-rmBack01.avi'
outName2 = outPrefix + '-rmBack02-k{}.avi'.format(kernSize)
if resize :
	outVid1 = cv2.VideoWriter(outName1, compression, 30, (rWidth,rHeight), False)
	outVid2 = cv2.VideoWriter(outName2, compression, 30, (rWidth,rHeight), isColor)
else :
	outVid1 = cv2.VideoWriter(outName1, compression, 30, (inWidth,inHeight), False)
	outVid2 = cv2.VideoWriter(outName2, compression, 30, (inWidth,inHeight), isColor)
#end if	


# Throw out first N frames
numThrow = int(round(timeThrow * fps)) * 60
count = 0
while count <= numThrow:
	# Get a frame
	ret, frame = inVid.read()
	count += 1
#end loop



#bgnd = cv2.BackgroundSubtractorMOG()
bgnd = cv2.BackgroundSubtractorMOG2()
#bgnd = cv2.BackgroundSubtractorGMG()
morphKern = np.ones((kernSize, kernSize),np.uint8)

count = 0
numKeep = int(round(timeKeep * fps)) * 60
while inVid.isOpened():
	count += 1
	if (numKeep > 3) and (count > numKeep) :
		break

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

		bgmask = bgnd.apply(frame, learningRate=0.003) 
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