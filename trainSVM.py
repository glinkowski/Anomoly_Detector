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
import os
import random
from sklearn import svm, grid_search
from sklearn.externals import joblib



####### ####### ####### ####### 
# PARAMETERS

# whether to augment the vectors
augNeg = True
augPos = True

# image dimensions
iwidth = 1920
iheight = 1080

# limits on the vector magnitude
magMax = 100
magMin = 15
magFlux = 40

# SVM properties
svmKernel = 'rbf'
doGridSearch = True

# grid search values (where to start?)
#	meant to be used with rbf
svmParams = { #'kernel':('rbf', 'linear'), 
			'C':(0.01, 0.01, 1, 10),
			'gamma':(0.00025, 0.0025, 0.025, 0.25, 2.5)}

####### ####### ####### ####### 



####### ####### ####### ####### 
# BEGIN MAIN FUNCTION
print("")



# Get list of vector files in the directory
vDir = sys.argv[1]
if not vDir.endswith('/') :
	vDir = vDir + '/'

vFiles = [(vDir + f) for f in os.listdir(vDir) if f.endswith('.txt')]
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
	# trNeg = np.vstack((capNeg, augA, augB))
	trNeg = np.vstack((trNeg, augA, augB))
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

		xrand = random.randint(-19, 19)
		yrand = random.randint(-19, 19)
		mrand = random.randint(-magFlux, magFlux)
#		arand = random.randint(-9, 9) * 0.004

	#TODO: vectorize this!
		augC[r,0] += xrand
		augC[r,1] += yrand
		augC[r,2] += mrand
		augC[r,2] = max(augC[r,2], magMin)
#		augC[r,2] = min(augC[r,2], 150)
#		augC[r,3] += arand

		# flip the angle
		if augC[r,3] >= 0 :
			augC[r,3] = augC[r,3] - 3.14159
		else :
			augC[r,3] = augC[r,3] + 3.14159
		#end if
	#end loop

	# some minimum (non-zero) vectors
	augD = np.copy(capPos)
	augD[:,2] = np.minimum(magMin, augD[:,2])

	# concatenate to the original vector list
	trPos = np.vstack((capPos, augC, augD))

#	print np.amax(trPos, axis=0)

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


# Prep the Training Data
print("Training the classifier ...")
print("    positives: {}, negatives: {}".format( 
	trPos.shape[0], trNeg.shape[0]))
trData = np.vstack((trNeg, trPos))
#print trData.shape, (trNeg.shape[0] + trPos.shape[0])
# labels = 0,1 
trLabels = np.vstack(( np.zeros((trNeg.shape[0],1)),
	np.ones((trPos.shape[0],1)) ))
trLabels = np.ravel(trLabels)
#print trLabels.shape
#cfier = svm.SVC(kernel=svmKernel)#, degree=3, gamma=4)


# Train the SVM
if not os.path.exists(vDir + 'SVM/') :
	os.makedirs(vDir + 'SVM/')
#end if
if doGridSearch :
	cfGrid = grid_search.GridSearchCV(svm.SVC(), param_grid=svmParams)
	cfGrid.fit(trData, trLabels)
	print("Grid search results:\n  score={}\n  params={}".format(
		cfGrid.best_score_, cfGrid.best_params_))
	print("Saving the SVM ...")
	joblib.dump(cfGrid, vDir + 'SVM/classifier.pkl')

	print(cfGrid.get_params())
else :
	cfier=svm.SVC(kernel=svmKernel)#, degree=3, gamma=4)
	cfier.fit(trData, trLabels)
	print("Saving the SVM ...")
	joblib.dump(cfier, vDir + 'SVM/classifier.pkl')
#end if


#NOTE: to call later:
#	cfier = joblib.load(vDir + 'classifier.pkl')


print('\nDone. ... for now\n')