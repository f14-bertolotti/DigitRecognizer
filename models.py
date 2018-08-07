import keras
from keras.layers import Activation, Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential

class models:

	def CNN2D_type0(traindata, trainlabels, testdata, testlabels):
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
		model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(10, activation='softmax'))
		return  model,\
				traindata.reshape((len(traindata),28,28,1)),\
				trainlabels.reshape((len(trainlabels),10)),\
				testdata.reshape((len(testdata),28,28,1)),\
				testlabels.reshape((len(testlabels),10))

	def MLPNN_type0(traindata, trainlabels, testdata, testlabels):
		model = Sequential()
		model.add(Dense(512, activation='relu', input_shape=(784,)))
		model.add(Dropout(0.2))
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(10, activation='softmax'))
		return  model,\
				traindata.reshape((len(traindata),784)),\
				trainlabels.reshape((len(trainlabels),10)),\
				testdata.reshape((len(testdata),784)),\
				testlabels.reshape((len(testlabels),10))

	def MLPNN_type1(traindata, trainlabels, testdata, testlabels):
		model = Sequential()
		model.add(Dense(64, activation='relu', input_shape=(784,)))
		model.add(Dropout(0.2))
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.2))				
		model.add(Dense(10, activation='softmax'))
		return  model,\
				traindata.reshape((len(traindata),784)),\
				trainlabels.reshape((len(trainlabels),10)),\
				testdata.reshape((len(testdata),784)),\
				testlabels.reshape((len(testlabels),10))