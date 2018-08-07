import os
import keras
import dataset
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import Activation, Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.utils import plot_model
from optparse import OptionParser
from itertools import count
from models import models

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=2)


'''
	creates Neural Networks models as MLPNN, CNN and committees of both,
	by selecting params from CLI.  

	examples are:
	python3 trainer.py -s 512 -m "MLPNN_type0" -e 100 -c 20 -b
	python3 trainer.py -s 128 -m "CNN2D_type0" -e 20 -c 10 -b
	python3 trainer.py -s 512 -m "MLPNN_type0" -e 100 -c 20
	python3 trainer.py -s 128 -m "CNN2D_type0" -e 20 -c 10
	python3 trainer.py -s 512 -m "MLPNN_type0" -e 100
	python3 trainer.py -s 128 -m "CNN2D_type0" -e 20
	python3 trainer.py

'''

if __name__ == '__main__':


	parser = OptionParser()
	parser.add_option("-d", "--directory", dest="dirname", default="",
	                  help="where to store training info and n.n. models")
	parser.add_option("--datasetname", dest="datasetname", default="mnist",
	                  help="name of the dataset, listed here: mnist")
	parser.add_option("-s", "--batch", dest="batch_size", default=128, type ="int",
	                  help="set the batch size for the training")
	parser.add_option("-e", "--epochs", dest="epochs", default=20, type ="int",
	                  help="sets the epochs for the training")
	parser.add_option("-b", "--balance", dest="balance", default=False,
	                  help="balance data before training", action="store_true")
	parser.add_option("-c", "--committee", dest="committee", default=1, type ="int",
	                  help="number of model to generate")
	parser.add_option("-m", "--modelname", dest="modelname", default="CNN2D_type0",
                 	  help="modelname to train, listed here: CNN2D_type0")
	(options, args) = parser.parse_args()


	if options.dirname == "":
		options.dirname = "./"+options.datasetname+"_models/"
		options.dirname += options.modelname+"-batch"+str(options.batch_size)
		if options.balance: options.dirname+="-balanced"
		if options.committee != 1: options.dirname+="-committee"+str(options.committee)


	data = getattr(dataset.dataset, options.datasetname)()
	train_data = data[0]
	train_labels = data[1]
	test_data = data[2]
	test_labels = data[3]
	train_data_balanced = data[4]
	train_labels_balanced = data[5]

	if not os.path.exists("./"+options.datasetname+"_models"): 
		os.makedirs("./"+options.datasetname+"_models")
	if not os.path.exists(options.dirname): 
		os.makedirs(options.dirname)

	trains = train_data_balanced if options.balance else train_data
	labels = train_labels_balanced if options.balance else train_labels


	for nn in range(options.committee):

		data = getattr(models,options.modelname)(trains,labels,test_data,test_labels)
		model = data[0]
		trains = data[1]
		labels = data[2]
		t_trains = data[3]
		t_labels = data[4]

		print(model.summary())
		plot_model(model, to_file=options.dirname+'/model.png',show_shapes=True)

		model.compile(optimizer='rmsprop',
		              loss='categorical_crossentropy',
		              metrics=['accuracy'])

		filename = options.dirname+"/model"+str(nn)+".best.hdf5"
		checkpoint = ModelCheckpoint(filename,save_best_only=True,mode="auto")


		history = model.fit(trains, labels, 
				  batch_size = options.batch_size,
				  validation_data = (t_trains,t_labels),
				  epochs = options.epochs,
				  callbacks = [checkpoint])

		plt.figure(1)
		plt.plot(history.history['acc'])  
		plt.plot(history.history['val_acc'])  
		plt.title('model accuracy')  
		plt.ylabel('accuracy')  
		plt.xlabel('epoch')  
		plt.legend(['train', 'test'], loc='upper left')  
		plt.savefig(options.dirname+"/"+str(nn)+"train_loss_acc.png")
		plt.close()

		plt.figure(2)
		plt.plot(history.history['loss'])  
		plt.plot(history.history['val_loss'])  
		plt.title('model loss')  
		plt.ylabel('loss')  
		plt.xlabel('epoch')  
		plt.legend(['train', 'test'], loc='upper left')  
		plt.savefig(options.dirname+"/"+str(nn)+"test_loss_acc.png") 
		plt.close()