# ---------------------------------------------------------
# author: Greg Linkowski
# project: Anomoly Detection
#		for Computer Vision final project
# 
# Crop & Downsample the video
#	Downsample both the resoultion and the frames
# 
# ---------------------------------------------------------

import cv2
import sys



####### ####### ####### ####### 
# PARAMETERS

# time to discard at the beginning (in min)
timeThrow = 0.025
# time to keep in final video (in min)
timeKeep = 0.925

# FPS for output (should divide 30)
changeFPS = False
outFPS = 3


# display frame-by-frame progress
showProgress = False


# if FPS can't be extracted from original
guessFPS = 3
forceFPS = False

# if isColor is False, convert to grayscale
isColor = True
# resize the video ?
resize = False
rWidth = 640
rHeight = 480
# output format
compression = cv2.cv.CV_FOURCC('M','P','4','V')

####### ####### ####### ####### 



####### ####### ####### ####### 
# BEGIN MAIN FUNCTION
print("")


# Open input video and get frame rate
inVid = cv2.VideoCapture(sys.argv[1])
inFPS = inVid.get(cv2.cv.CV_CAP_PROP_FPS)
print("original video fps = {}".format(inFPS))
# used with % for output in processing loop :
if inFPS < 121 :
	inFPS = int(round(inFPS))
else :	# to handle fps = NaN
	inFPS = guessFPS
#end if
if forceFPS :
	inFPS = guessFPS
#end if
if not changeFPS :
	outFPS = inFPS
#end if
print("output video fps = {}".format(outFPS))

inWidth = int(inVid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
inHeight = int(inVid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
print("original video resolution = {}x{}".format(inWidth, inHeight))


# Initialize output video (for linux/mac, h.264 code is MP4V)
outPrefix = sys.argv[1][0:-4]
outName = outPrefix + '-clip.avi'
if resize :
	outVid = cv2.VideoWriter(outName, compression, outFPS, (rWidth,rHeight), isColor)
else :
	outVid = cv2.VideoWriter(outName, compression, outFPS, (inWidth,inHeight), isColor)
#end if

# Throw out first N frames
numThrow = int(round(timeThrow * inFPS * 60))
count = 0
while count <= numThrow:
	# Get a frame
	ret, frame = inVid.read()
	count += 1
#end loop


skipCount = 0
skipStop = int(round(inFPS / outFPS))
count = 0
numKeep = int(round(timeKeep * inFPS * 60))
while inVid.isOpened():
	count += 1

	# Run until stopping point
	if (numKeep > 3) and (count > numKeep) :
		break

	if not (count % (outFPS * 3)) :
		print("processed: {}".format(count))

	# Skip desired frames per second
	if (skipCount < skipStop) :
		ret, frame = inVid.read()
		skipCount += 1
		continue
	skipCount = 0

	# Get the frame
	ret, frame = inVid.read()
	if ret:

		# Display image (buggy)
		if showProgress :
			cv2.imshow("input", frame);
			cv2.waitKey(10)
		#end if

		# Resize to smaller scale
		if resize :
			frame = cv2.resize(frame, (rWidth,rHeight))
		# Convert to grayscale
		if not isColor :
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		#end if
		outVid.write(frame)
#end loop


cv2.destroyAllWindows()


inVid.release()
outVid.release()


print("\nDone.\n")