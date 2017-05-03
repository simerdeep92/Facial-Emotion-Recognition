import numpy as np
import random
import cv2
import glob
import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

np.random.seed(123)

Emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
bin_n = 128

def hog(img):
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
	# quantizing binvalues in (0...16)
	bins = np.int32(bin_n*ang/(2*np.pi))
	# Divide to 4 sub-squares
	bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
	mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)
	return hist

def CreateDataset():
	trainX = []
	trainY = []
	testX = []
	testY = []
	for emotion in Emotions:
		print emotion
		files = glob.glob("dataset/%s/*" %emotion)
		random.shuffle(files)
		ntrain = int(len(files) * 0.8)
		ntest = len(files) - ntrain		
		for file in files[: ntrain]:
			trainX.append(hog(cv2.resize(cv2.imread(file) , (96,96))))
			trainY.append(Emotions.index(emotion))
		for file in files[-ntest:]:
			testX.append(hog(cv2.resize(cv2.imread(file) , (96,96))))
			testY.append(Emotions.index(emotion))
	return trainX,trainY,testX,testY

trainX,trainY,testX,testY = CreateDataset();
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
scaler = StandardScaler()
testX = scaler.fit_transform(testX)
print(len(trainX))
print(len(testX))

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


def Create_Model_LinearSVM():
	print "SVM Started"
	clf = svm.SVC(kernel = 'linear')
	clf.fit(trainX, trainY) 
	predicted = clf.predict(testX)
	# get the accuracy
	print "Liner Kernel"
	print accuracy_score(testY, predicted)

def Create_Model_RadialSVM():
	clf = svm.SVC(kernel = 'poly')
	clf.fit(trainX, trainY) 
	predicted = clf.predict(testX)
	# get the accuracy
	print "Radial Kernel"
	print accuracy_score(testY, predicted)

Create_Model_RadialSVM()


def CreateCSVDataset():
	trainX = []
	trainY = []
	testX = []
	testY = []
	for emotion in Emotions:
		print emotion
		files = glob.glob("dataset/%s/*" %emotion)			
		for file in files[0:1]:
			trainX.append(hog.compute(cv2.imread(file)))
			trainY.append(Emotions.index(emotion))
	# with open("Features_data.csv", "wb") as f:
	#     writer = csv.writer(f)
	#     writer.writerows(trainX)	
	# with open("Label_data.csv", "wb") as f:
	#     writer = csv.writer(f)
	#     writer.writerows(trainY)

