import numpy as np

class dataset:

	def mnist():

		train_data = None
		train_labels = None
		test_data = None
		test_labels = None
		train_data_balanced = None
		train_labels_balanced = None

		with open("./MNIST_dataset/train-labels-idx1-ubyte", "rb") as f:

			magic_number_bytes = np.array(f.read(4))
			magic_number = magic_number_bytes.view(dtype=np.dtype('>i4'))
			
			labels_number_byte = np.array(f.read(4))
			labels_number = labels_number_byte.view(dtype=np.dtype('>i4'))	

			labels_byte = f.read(labels_number)
			labels_uint = np.frombuffer(labels_byte, dtype=">u1")
			train_labels = np.array([[1 if e == i else 0 for e in range(10)] for i in labels_uint])

		with open("./MNIST_dataset/train-images-idx3-ubyte", "rb") as f:

			magic_number_bytes = np.array(f.read(4))
			magic_number = magic_number_bytes.view(dtype=np.dtype('>i4'))
			
			images_number_bytes = np.array(f.read(4))
			images_number = images_number_bytes.view(dtype=np.dtype('>i4'))	

			rows_number_bytes = np.array(f.read(4))
			rows_number = rows_number_bytes.view(dtype=np.dtype('>i4'))	
			
			cols_number_bytes = np.array(f.read(4))
			cols_number = cols_number_bytes.view(dtype=np.dtype('>i4'))	
			
			img_size = rows_number*cols_number

			images_byte = f.read(images_number*img_size)
			images_uints = np.frombuffer(images_byte, dtype=">u1")/255
			train_data = images_uints.reshape((60000,28,28,1))

		with open("./MNIST_dataset/t10k-labels-idx1-ubyte", "rb") as f:

			magic_number_bytes = np.array(f.read(4))
			magic_number = magic_number_bytes.view(dtype=np.dtype('>i4'))
			
			labels_number_byte = np.array(f.read(4))
			labels_number = labels_number_byte.view(dtype=np.dtype('>i4'))	

			labels_byte = f.read(labels_number)
			labels_uint = np.frombuffer(labels_byte, dtype=">u1")
			test_labels = np.array([[1 if e == i else 0 for e in range(10)] for i in labels_uint])


		with open("./MNIST_dataset/t10k-images-idx3-ubyte", "rb") as f:

			magic_number_bytes = np.array(f.read(4))
			magic_number = magic_number_bytes.view(dtype=np.dtype('>i4'))
			
			images_number_bytes = np.array(f.read(4))
			images_number = images_number_bytes.view(dtype=np.dtype('>i4'))	

			rows_number_bytes = np.array(f.read(4))
			rows_number = rows_number_bytes.view(dtype=np.dtype('>i4'))	
			
			cols_number_bytes = np.array(f.read(4))
			cols_number = cols_number_bytes.view(dtype=np.dtype('>i4'))	
			
			img_size = rows_number*cols_number

			images_byte = f.read(images_number*img_size)
			images_uints = np.frombuffer(images_byte, dtype=">u1")/255
			test_data = images_uints.reshape((10000,28,28,1))

			b_lbls = []
			b_imgs = []
			for digit in range(10):
				i = 0
				for img,lbl in zip(train_data,train_labels):
					if list(lbl).index(1) == digit and i < 5420:
						b_imgs.append(img)
						b_lbls.append(lbl)
						i += 1

			train_data_balanced = np.array(b_imgs).reshape((54200,28,28,1))
			train_labels_balanced = np.array(b_lbls).reshape((54200,10))

			return  train_data,\
					train_labels,\
					test_data,\
					test_labels,\
					train_data_balanced,\
					train_labels_balanced
