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

# number of frames to skip / downsample
nSkip = 0

# total number of frames for final video
nFinal = 250

convertToBW = True

####### ####### ####### ####### 



####### ####### ####### ####### 
# BEGIN MAIN FUNCTION


inVid = cv2.VideoCapture(sys.argv[1])

outVid = cv2.VideoWriter('clip.avi', cv2.cv.CV_FOURCC('M','J','P','G'), 30.0, (640,480), (not convertToBW))


count = 0
added = 0
while inVid.isOpened():

	# Get the frame
	ret, frame = inVid.read()

	# exit at desired length
	if added >= nFinal :
		break

	# skip frames as long as desired
	count += 1
	if (count % (nSkip+1)) :
#		print( "Skipping frame {}, mod={}".format(count, (count % (nSkip+1))) )
		continue
	#end if


	if ret:
		# Resize to smaller scale
		frame = cv2.resize(frame, (640,480))

		# Convert to grayscale
		if convertToBW :
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		#end if

		outVid.write(frame)
		added += 1

		if not (added % 25) :
			print("processed {}".format(added))
		#end if
	else :
		break
	#end if

#end loop

inVid.release()
outVid.release()


print("\nDone.\n")