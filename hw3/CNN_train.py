import numpy as np
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten,AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import regularizers
from  keras.callbacks import ModelCheckpoint,Callback
#categorical_crossentropy

def load_data():
	file = open(sys.argv[1])
	content = file.readlines()
	x_train = []
	y_train = []
	
	for line in content:
		line = line.replace('\n','').split(',')
		y_train.append(line[0])
		x_train.append(line[1].split(' '))
	y_train = np.array(y_train[1:]).astype(np.int)
	x_train = np.array(x_train[1:]).astype(np.int)

	# xdata = np.empty((28709,1,48,48))
	# for i in range(28709):
	# 	xdata[i,1,:,:]=x_train[i*48:i*48+48]

	file.close()

	#double x_train
	
	cn = 0
	for line in x_train:
		xdata = np.reshape(line,(48,48))
		if cn == 0:
			xmirror_train = [xdata[...,::-1].reshape(48*48)]
			cn += 1
			continue
		cn += 1
		print (cn)
		xxdata = [xdata[...,::-1].reshape(48*48)]
		xmirror_train = np.append(xmirror_train,xxdata, axis=0)
		#print (xmirror_train)
	#print (xmirror_train.shape)
	x_train = np.append(x_train,xmirror_train,axis=0)
	#print (x_train.shape)
	y_train = np.append(y_train,y_train)

 # convert class vectors to binary class matrices
	y_train = np_utils.to_categorical(y_train, 7)
	x_train = x_train/255.0

	return (x_train, y_train)

(x_train,y_train)=load_data()

havemodel = 0
train = 1


if train:
	model2 = Sequential()
	model2.add(Conv2D(32,(3,3),input_shape=(48,48,1)))
	model2.add(Activation('relu'))
#	model2.add(Dropout(0.5))

	model2.add(Conv2D(32,(3,3)))
	model2.add(Activation('relu'))
	model2.add(MaxPooling2D((2,2)))
	model2.add(Dropout(0.25))

	model2.add(Conv2D(64,(3,3), border_mode='valid'))
	model2.add(Activation('relu'))

	model2.add(Conv2D(64,(3,3)))
	model2.add(Activation('relu'))
	model2.add(MaxPooling2D((2,2)))
	model2.add(Dropout(0.25))

	model2.add(Conv2D(128,(3,3)))
	model2.add(Activation('relu'))

	model2.add(Conv2D(128,(3,3)))
	model2.add(Activation('relu'))
	model2.add(AveragePooling2D(pool_size=(2, 2)))  # S6
	model2.add(Dropout(0.25))

	model2.add(Flatten())
	model2.add(Dense(units=512,activation='relu'))#, kernel_regularizer=regularizers.l2(0.01))),kernel_regularizer=regularizers.l2(0.01)
	model2.add(Dropout(0.5))

	model2.add(Dense(units=7,activation='softmax'))
	model2.summary()


	x_train = x_train.reshape(x_train.shape[0],48,48,1)


#sparse_categorical_crossentropy
	opt = SGD(lr=0.01,decay=1e-6, momentum=0.9, nesterov=True)
	model2.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
	result = model2.fit(x_train,y_train,batch_size=70,epochs=200,validation_split=0.1,shuffle=True)#train model，hisrory,
	score = model2.evaluate(x_train,y_train)
	model2.save_weights('./model.h5')
	print ('\nTrain Acc:', score[1])
