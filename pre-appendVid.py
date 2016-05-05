# ---------------------------------------------------------
# author: Greg Linkowski
# project: Anomoly Detection
#		for Computer Vision final project
# 
# Concatenate several videos in a row. Can be used to combine
#	several training videos, or to add a clean shot of the
#	location to the beginning of a busy video.
# 
# Creates ----
#	appended.avi
# ---------------------------------------------------------

import cv2
import sys
import numpy as np



####### ####### ####### ####### 
# PARAMETERS

# folder where videos reside
vPath = 'MNTL/'
#ordered list of videos
vList = ['mntl01.MTS',
		'mntl02.MTS']

# number of initial frames to discard
numThrow = [0, 0]


####### ####### ####### ####### 



####### ####### ####### ####### 
# BEGIN MAIN FUNCTION
print("")


# Open input video to get stats
inVid = cv2.VideoCapture(vPath + vList[0])
fps = inVid.get(cv2.cv.CV_CAP_PROP_FPS)
print("original video fps = {}".format(fps))
# used with % for output in processing loop :
fps2 = int(round(fps))

inWidth = int(inVid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
inHeight = int(inVid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
print("original video resolution = {}x{}".format(inWidth, inHeight))

inVid.release()

# Initialize output video (for linux/mac, h.264 code is MP4V)
outVid = cv2.VideoWriter(vPath + 'append.avi', cv2.cv.CV_FOURCC('M','P','4','V'), fps/2, (inWidth,inHeight), True)


# Concatenate the videos
for i in xrange(len(vList)) :
	inVid = cv2.VideoCapture(vPath + vList[i])

	# Throw out first N frames
	count = 0
	while count <= numThrow[i]:
		# Get a frame
		ret, frame = inVid.read()
		count += 1
	#end loop


	# Copy input video into output
	count = 0
	print("adding {}".format(vList[i]))
	while inVid.isOpened():
		count += 1
		if not (count % (fps2 * 2)) :
			print("    processed frames: {}".format(count))

		# Get the frame
		ret, frame = inVid.read()

		if ret:
			outVid.write(frame)
		else :
			print("    finished reading video")
			break
		#end if
	#end loop

	inVid.release()
#end loop


outVid.release()



print("\nDone.\n")