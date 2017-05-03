import glob
from shutil import copyfile
import random
import mahotas
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
np.random.seed(123)
from keras.models import Sequential, Model
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Merge, TimeDistributed, Masking
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras import preprocessing
from keras.preprocessing import sequence
import PIL
from PIL import Image as Im
from resizeimage import resizeimage
import os, sys
from mahotas import imresize
import skimage
from skimage import data, io, filters, transform



Emotions = [ "anger", "disgust", "fear", "happy", "sadness", "surprise"]
imageLenTrain = []
imageLenTest=[]



def CreateDataset():
	trainX = []
	trainY = []
	testX = []
	testY = []
	imageSeq=[]
	sessionsLen=[]
	size = 100, 100

	for emotion in Emotions:
		emoPart=glob.glob("/Users/bita/Desktop/sequentialdataset/%s/*" %emotion)
		random.shuffle(emoPart)
#		print(emoPart)
		trainLen=int(len(emoPart)*0.8)
		testLen=len(emoPart)-trainLen
		for part in emoPart[:trainLen]:
			#print(part)
			images=sorted(glob.glob("/%s/*" %part))
			imageLenTrain.append(len(images))
			for imageFile in images:
				im1=mahotas.imread(imageFile)
				im1=skimage.transform.resize(im1, (100,100))
				imageSeq.append(im1)

			trainX.append(imageSeq)
			trainY.append(Emotions.index(emotion))
			imageSeq=[]

		
		for part in emoPart[trainLen:]:
			images=sorted(glob.glob("/%s/*" %part))
			imageLenTest.append(len(images))
			for imageFile in images:
				im1=mahotas.imread(imageFile)
				im1=skimage.transform.resize(im1, (100,100))
				imageSeq.append(im1)
			testX.append(imageSeq)
			testY.append(Emotions.index(emotion))
			imageSeq=[]#
	#print(max(imageLenTest))

	trainX=np.array(trainX)
	testX=np.array(testX)

	paddImageTrain=preprocessing.sequence.pad_sequences(trainX, maxlen=max(max(imageLenTrain),max(imageLenTest)), padding='post', truncating='pre', value=0.0)
	paddImageTest=preprocessing.sequence.pad_sequences(testX, maxlen=max(max(imageLenTrain),max(imageLenTest)), padding='post', truncating='pre', value=0.0)
	trainY = np_utils.to_categorical(trainY, len(Emotions))
	testY = np_utils.to_categorical(testY, len(Emotions))

	return paddImageTrain,trainY,paddImageTest,testY




trainX,trainY,testX,testY=CreateDataset()
print(trainX.shape)
print(testX.shape)
seq_len=max(max(imageLenTrain),max(imageLenTest))

def CreateModel(LSTM1,LSTM2,nEpoch,batchSize):
	model = Sequential()
	model.add(TimeDistributed(Convolution2D(32, 5, 5,border_mode='valid', activation='relu'), input_shape=(seq_len,100, 100,1)))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	model.add(TimeDistributed(Convolution2D(64, 5, 5, border_mode='valid', activation='relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	model.add(TimeDistributed(Convolution2D(128, 5, 5, border_mode='valid', activation='relu')))
	#model.add(TimeDistributed(Dropout(0.2)))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	#model.add(TimeDistributed(Dropout(0.2)))
	model.add(TimeDistributed(Flatten()))
	model.add(LSTM(LSTM1, return_sequences = True))
	#model.add(Dropout(0.2))
	model.add(LSTM(LSTM2))
	model.add(Dense(6, init='uniform'))
	model.add(Activation('softmax'))


	model.summary()

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


	# 9. Fit model on training data
	model.fit(trainX, trainY, batch_size=batchSize, epochs=nEpoch, verbose=1)
	 
	# 10. Evaluate model on test data
	score = model.evaluate(testX, testY, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	with open("results.txt", 'a') as file:
		file.write("#################################")
		file.write("\n")
		file.write("]LSTM1:")
		file.write(str(LSTM1))
		file.write("\n")
		file.write("LSTM2:")
		file.write(str(LSTM2))
		file.write("\n")
		file.write("Epoch:")
		file.write (str(nEpoch))
		file.write("\n")
		file.write("batch_size:")
		file.write(str(batchSize))
		file.write("\n")
		file.write("\n")
		file.write("score:")
		file.write(str(score))
		file.write("\n")
		file.write("#################################")
		file.write("\n")
		file.write("\n")
	return score

#score1=CreateModel(30,15,10,32)
#score2=CreateModel(30,15,10,64)
#score3=CreateModel(100,50,10,32)
score4=CreateModel(100,50,30,32)
# if(score1 >= max(score2,score3,score4)):
# 	score11=CreateModel(30,15,30,32)
# if(score2 >= max(score1,score3,score4)):
# 	score22=CreateModel(30,15,30,64)
# if(score3>= max(score2,score1,score4)):
# 	score33=CreateModel(100,50,30,32)
# if(score4 >= max(score2,score3,score1)):
# 	score44=CreateModel(20,10,30,32)










