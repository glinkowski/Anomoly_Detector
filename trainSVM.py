# ---------------------------------------------------------
# authors: Mohammad Saad & Greg Linkowski
# project: Anomoly Detection
#		for Computer Vision final project
# 
# After detect_people, the vectors will be in text files
#	presumably in the same folder. File names will start
#	with 'TrainPos' or 'TrainNeg'. There is the option
#	of augmenting the training vectors in the case the
#	data is somewhat slim.
# This script trains the SVM and saves the model
#
# 
# Use ----
#	...$ python detect_people.py <input folder>
#
#
# Input ----
#	text files containing training vectors (+/-)
#
# Output ----
#	the SVM model
# ---------------------------------------------------------

import numpy as np
import sys
from os import listdir
import random
from sklearn import svm
from sklearn.externals import joblib



####### ####### ####### ####### 
# PARAMETERS

# whether to augment the vectors
augPos = True
augNeg = True

# image dimensions
iwidth = 1920
iheight = 1080

# limits on the vector magnitude
magMax = 150
magMin = 15
magFlux = 15

####### ####### ####### ####### 



####### ####### ####### ####### 
# BEGIN MAIN FUNCTION
print("")



# Get list of vector files in the directory
vDir = sys.argv[1]
if not vDir.endswith('/') :
	vDir = vDir + '/'

vFiles = [(vDir + f) for f in listdir(vDir) if f.endswith('.txt')]
vFiles.sort()



# Populate the negative vectors
#trNeg = list()
trNeg = np.zeros((0,4), dtype=np.float32)
for vf in vFiles :
	if vf.startswith(vDir + 'TrainNeg') :
		print("Reading file {}".format(vf))
		with open(vf, 'r') as fin :
			for line in fin :
				line = line.rstrip()
				lv = line.split('\t')

				row = ([ float(lv[0]), float(lv[1]),
					float(lv[2]), float(lv[3]) ])
#				trNeg.append(row)
				trNeg = np.vstack((trNeg, row))
		#end with
#end loop

# Populate the positive vectors
#trPos = list()
trPos = np.zeros((0,4), dtype=np.float32)
for vf in vFiles :
	if vf.startswith(vDir + 'TrainPos') :
		print("Reading file {}".format(vf))
		with open(vf, 'r') as fin :
			for line in fin :
				line = line.rstrip()
				lv = line.split('\t')

				row = ([ float(lv[0]), float(lv[1]),
					float(lv[2]), float(lv[3]) ])
#				trPos.append(row)
				trPos = np.vstack((trPos, row))
		#end with
#end loop



# Augment the negative examples
if augNeg :

	# discard obvious nonsense
	capNeg = trNeg[(trNeg[:,2] <= (magMax * 4)), :]

	# horizontal flip
	augA = np.copy(capNeg)
	augA[:,0] = np.subtract(iwidth, augA[:,0])

	# zero out
	augB = np.copy(trPos)
	augB[:,2] = np.zeros((augB.shape[0]))

	# concatenate to the original vector list
	trNeg = np.vstack((capNeg, augA, augB))
#	trNeg = np.copy(augA)
	# save to file
	with open(vDir + 'negAugs.txt', 'w') as fout :
		firstLine = True
		for i in xrange(trNeg.shape[0]) :
			if firstLine :
				firstLine = False
			else :
				fout.write('\n')
			fout.write('{:d}\t{:d}\t{:f}\t{:f}'.format(
				int(trNeg[i,0]), int(trNeg[i,1]),
				trNeg[i,2], trNeg[i,3]))
	#end with
#end if

# Augment the positive examples
if augPos :

	# cap the maximum positive magnitude
#	trPos[:,2] = np.minimum(trPos[:,2], 200)
	capPos = trPos[(trPos[:,2] <= magMax), :]
#	print np.amax(capPos[:,2])
#	print trPos.shape, capPos.shape
#	print np.amax(capPos, axis=0)

	# tiny random adjustments
	augC = np.copy(capPos)
	for r in xrange(augA.shape[0]) :

		xrand = random.randint(-11, 11)
		yrand = random.randint(-11, 11)
		mrand = random.randint(-magFlux, magFlux)
		arand = random.randint(-9, 9) * 0.002

#		print augA[r,:]
		augC[r,0] += xrand
		augC[r,1] += yrand
#		print augA[r,2]
		augC[r,2] += mrand
#		print augA[r,2]
		augC[r,2] = max(augC[r,2], magMin)
#		augC[r,2] = min(augC[r,2], 150)
		augC[r,3] += arand

	#end loop

	# concatenate to the original vector list
	trPos = np.vstack((capPos, augC))

	print np.amax(trPos, axis=0)

	# save to file
	with open(vDir + 'posAugs.txt', 'w') as fout :
		firstLine = True
		for i in xrange(trPos.shape[0]) :
			if firstLine :
				firstLine = False
			else :
				fout.write('\n')
			fout.write('{:d}\t{:d}\t{:f}\t{:f}'.format(
				int(trPos[i,0]), int(trPos[i,1]),
				trPos[i,2], trPos[i,3]))
	#end with
#end if


# Train the SVM
trData = np.vstack((trNeg, trPos))
#print trData.shape, (trNeg.shape[0] + trPos.shape[0])
# labels = 0,1 
trLabels = np.vstack(( np.zeros((trNeg.shape[0],1)),
	np.ones((trPos.shape[0],1)) ))
trLabels = np.ravel(trLabels)
#print trLabels.shape
cfier = svm.SVC()
cfier.fit(trData, trLabels)

#SVC(kernel='rbf', )

joblib.dump(cfier, vDir + 'classifier.pkl')

#NOTE: to call later:
#	cfier = joblib.load(vDir + 'classifier.pkl')


print('\nDone. ... for now\n')