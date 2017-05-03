import numpy as np
import random
import cv2
import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
from PIL import Image

from resizeimage import resizeimage

np.random.seed(123)

Emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

def CreateDataset():
	trainX = []
	trainY = []
	testX = []
	testY = []
	for emotion in Emotions:
		files = glob.glob("dataset/%s/*" %emotion)
		random.shuffle(files)
		ntrain = int(len(files) * 0.8)
		ntest = len(files) - ntrain
		#print ntrain
		#print ntest
		#print len(files)
		for file in files[: ntrain]:
			trainX.append(cv2.resize(cv2.imread(file),(96,96)))
			trainY.append(Emotions.index(emotion))
		for file in files[-ntest:]:
			testX.append(cv2.resize(cv2.imread(file),(96,96)))
			testY.append(Emotions.index(emotion))
	trainY = np_utils.to_categorical(trainY, len(Emotions))
	testY = np_utils.to_categorical(testY, len(Emotions))
	trainX = np.mean(np.array(trainX),axis =3)
	testX = np.mean(np.array(testX),axis =3)
	trainX = trainX.reshape(trainX.shape[0], 1, 96, 96)
	testX = testX.reshape(testX.shape[0], 1, 96, 96)
	trainX = trainX.astype('float32')
	testX = testX.astype('float32')
	trainX /= 255
	testX /= 255
	print(trainX.shape)
	return trainX,trainY,testX,testY

trainX,trainY,testX,testY = CreateDataset();

def Evaluate_Save_Model(conv_structure,kernelsize,poolsize,dense_layer,dropout,epoch,batchsize):	# 7. Define model architecture
	model = Sequential()	
	# first conv layer compulsory
	model.add(Conv2D(conv_structure[0][0], kernel_size =(kernelsize, kernelsize), padding="same",activation='relu',data_format='channels_first', input_shape=(1,96,96)))
	#1st layer 
	for i in range(conv_structure[0][1] - 1):
		model.add(Conv2D(conv_structure[i][0], padding="same",kernel_size =(kernelsize, kernelsize), activation='relu'))
	model.add(MaxPooling2D(pool_size=(poolsize,poolsize)))

	if (len(conv_structure) > 1):
		## Other than first Convolution layer if any
		for conv in conv_structure[1:]:
			for c in range(conv[1]):
				model.add(Conv2D(conv_structure[c][0], padding="same",kernel_size =(kernelsize, kernelsize), activation='relu'))
			model.add(MaxPooling2D(pool_size=(poolsize,poolsize)))
			 
	model.add(Flatten())	
	for d in dense_layer:
		model.add(Dense(d, activation='relu'))
		if (dropout):
			model.add(Dropout(dropout))
	
	model.add(Dense(trainY.shape[1], activation='softmax'))
	model.summary()
	 
	# 8. Compile model
	model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	 
	# 9. Fit model on training data
	model.fit(trainX, trainY, batch_size=batchsize, epochs=epoch, verbose=1)
	 
	# 10. Evaluate model on test data
	score = model.evaluate(testX, testY, verbose=0)
	with open("results.txt", 'a') as file:
		file.write("#################################")
		file.write("\n")
		file.write("conv_structure:")
		file.write(str(conv_structure))
		file.write("\n")
		file.write("kernelsize:")
		file.write(str(kernelsize))
		file.write("\n")
		file.write("poolsize:")
		file.write (str(poolsize))
		file.write("\n")
		file.write("dense_layer:")
		file.write(str(dense_layer))
		file.write("\n")
		file.write("dropout:")
		file.write(str(dropout))
		file.write("\n")
		file.write("epoch:")
		file.write(str(epoch))
		file.write("\n")
		file.write("\n")
		file.write("score:")
		file.write(str(score))
		file.write("\n")
		file.write("#################################")
		file.write("\n")
		file.write("\n")
	return score
	

#Evaluate_Save_Model(conv_structure,kernelsize,poolsize,dense_layer,dropout,epoch,batchsize):	# 7. Define model architecture
	
#Evaluate_Save_Model([[32,1]],3,2,[64],0.5,1,32)
#Evaluate_Save_Model([[32,2],[64,2]],3,2,[64],0.5,1,32)
#Evaluate_Save_Model([[32,2],[64,2],[128,2]],3,2,[64],0.5,1,32)
# Evaluate_Save_Model([[32,1],[64,1],[128,1]],5,2,[64],0.75,10,32)
# Evaluate_Save_Model([[64,1],[128,1],[256,1]],5,2,[64],0.75,10,32)
# Evaluate_Save_Model([[32,1],[64,1],[128,1]],5,2,[64,64],0.75,10,32)
# Evaluate_Save_Model([[32,1],[64,1],[128,1]],5,2,[64],0.50,10,32)

# Evaluate_Save_Model([[32,1],[64,1],[128,1]],5,2,[64],0.20,10,32)
# Evaluate_Save_Model([[32,1],[64,1],[128,1]],3,2,[64],0.20,10,32)
# Evaluate_Save_Model([[32,1],[64,1],[128,1]],3,3,[64],0.20,10,32)
# Evaluate_Save_Model([[32,2],[64,2],[128,2]],5,2,[64],0.30,10,32)
# Evaluate_Save_Model([[32,2],[64,2],[128,2]],5,2,[64],0.25,10,32)
# Evaluate_Save_Model([[32,3],[64,3],[128,3]],3,2,[128],0.25,20,32)
# # Evaluate_Save_Model([[128,3],[256,3]],5,2,[128],0.25,10,32)
# #Evaluate_Save_Model([[32,4],[64,4],[128,4]],5,2,[64],0.25,10,32)
# Evaluate_Save_Model([[32,3],[64,3],[128,3]],3,2,[64],0.15,20,32)
# Evaluate_Save_Model([[32,1],[64,1],[128,1]],3,2,[64],0.20,20,32)
# Evaluate_Save_Model([[64, 1], [128, 1]],3,2,[64],0.10,80,32)
# Evaluate_Save_Model([[32,1],[64,1],[128,1]],3,2,[64],0.10,60,32)
# Evaluate_Save_Model([[32, 3], [64, 3], [128, 3]],3,2,[64],0.20,50,32)
# Evaluate_Save_Model([[32, 3], [64, 3], [128, 3]],3,2,[64],0.10,50,32)
# Evaluate_Save_Model([[32, 3], [64, 3], [128, 3]],5,3,[64],0.20,50,32)
# Evaluate_Save_Model([[32, 3], [64, 3], [128, 3]],3,2,[128],0.10,20,32)
# Evaluate_Save_Model([[32, 5], [64, 5], [128, 5]],3,2,[128],0.10,20,32)


