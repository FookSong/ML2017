import numpy as np
# import matplotlib as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten,AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import regularizers
from  keras.callbacks import ModelCheckpoint,Callback
import os
#categorical_crossentropy

class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_accs=[]
        self.val_accs=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))
def dump_history(store_path,logs):
    with open(os.path.join(store_path,'train_loss'),'a') as f:
        for loss in logs.tr_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'train_accuracy'),'a') as f:
        for acc in logs.tr_accs:
            f.write('{}\n'.format(acc))
    with open(os.path.join(store_path,'valid_loss'),'a') as f:
        for loss in logs.val_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'valid_accuracy'),'a') as f:
        for acc in logs.val_accs:
            f.write('{}\n'.format(acc))

def load_data():
	valid = 5600

	file = open('train.csv')
	content = file.readlines()
	x_train = []
	y_train = []
	
	for line in content:
		line = line.replace('\n','').split(',')
		y_train.append(line[0])
		x_train.append(line[1].split(' '))
	y_train = np.array(y_train[1:valid]).astype(np.int)
	x_train = np.array(x_train[1:valid]).astype(np.int)

	# xdata = np.empty((28709,1,48,48))
	# for i in range(28709):
	# 	xdata[i,1,:,:]=x_train[i*48:i*48+48]

	file.close()

	#double x_train
	# cn = 0
	# for line in x_train:
	# 	xdata = np.reshape(line,(48,48))
	# 	if cn == 0:
	# 		xmirror_train = [xdata[...,::-1].reshape(48*48)]
	# 		cn += 1
	# 		continue
	# 	cn += 1
	# 	print (cn)
	# 	xxdata = [xdata[...,::-1].reshape(48*48)]
	# 	xmirror_train = np.append(xmirror_train,xxdata, axis=0)
	# 	#print (xmirror_train)
	# print (xmirror_train.shape)
	# x_train = np.append(x_train,xmirror_train,axis=0)
	# print (x_train.shape)
	# y_train = np.append(y_train,y_train)

	file = open('train.csv')
	content = file.readlines()
	x_test = []
	# id_test = []
	for line in content:
		line = line.replace('\n','').split(',')
		# id_test.append(line[0])
		x_test.append(line[1].split(' '))
	# id_test = np.array(id_test[valid:]).astype(np.int)
	x_test = np.array(x_test[valid:]).astype(np.int)
	file.close()


	file = open('train100label.csv')
	content = file.readlines()
	idd_test = []
	for line in content:
		line = line.replace('\n','').split(',')
		idd_test.append(line[1])
	idd_test = np.array(idd_test[1:]).astype(np.int)
	file.close()



	x_train = np.append(x_train,x_test,axis=0)
	y_train = np.append(y_train,idd_test,axis=0)

	file = open('test.csv')
	content = file.readlines()
	x_newtest = []
	id_newtest = []
	for line in content:
		line = line.replace('\n','').split(',')
		id_newtest.append(line[0])
		x_newtest.append(line[1].split(' '))
	id_newtest = np.array(id_newtest[1:]).astype(np.int)
	x_newtest = np.array(x_newtest[1:]).astype(np.int)
	file.close()
	# cn=0
	# for line in x_test:
	# 	xdata = np.reshape(line,(48,48))
	# 	if cn == 0:
	# 		xmirror_test = [xdata[...,::-1].reshape(48*48)]
	# 		cn += 1
	# 		continue
	# 	cn += 1
	# 	print (cn)
	# 	xxdata = [xdata[...,::-1].reshape(48*48)]
	# 	xmirror_test = np.append(xmirror_test,xxdata, axis=0)
	# 	#print (xmirror_train)
	# x_test = np.append(x_test,xmirror_test,axis=0)



 # convert class vectors to binary class matrices
	y_train = np_utils.to_categorical(y_train, 7)
	x_train = x_train/255.0
	x_newtest = x_newtest/255.0

	return (x_train, y_train), (x_newtest,id_newtest)

(x_train,y_train),(x_test,id_test)=load_data()

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
	x_test = x_test.reshape(x_test.shape[0],48,48,1)


#sparse_categorical_crossentropy
	opt = SGD(lr=0.01,decay=1e-6, momentum=0.9, nesterov=True)
	model2.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

	# best_model_file = "./20%/checkpoint-{epoch:02d}-{loss:.4f}-{acc:.4f}.h5"
	# best_model = ModelCheckpoint(best_model_file, verbose=1,save_best_only=True)
	if havemodel:
		model2.load_weights('./200%/checkpoint-138-1.0016-0.6794.h5')
	store_path="./200%"
	hisrory = History()

	result = model2.fit(x_train,y_train,batch_size=70,epochs=80,shuffle=True,validation_split=0.1, callbacks=[hisrory])#train model
	model2.save_weights('./200%/model1.h5')
	dump_history(store_path,hisrory)
	score = model2.evaluate(x_train,y_train)
	print ('\nTrain Acc:', score[1])



# score2 = model2.predict(x_test, batch_size=70, verbose=0)
# score3 = score2[:7178,:]+score2[7178:,:]
# score3 =  np.argmax(score3, axis=1)
# f = open(str('CNN501_best.csv').rstrip(),'w')
# f.write('id,label\n')	
# i = 0
# while i < score3.shape[0]:
# 	f.write(str(i) + ',' + str(score3[i]) + '\n')	
# 	i += 1
# f.close()

score2 = model2.predict(x_test, batch_size=70, verbose=0)
score2 =  np.argmax(score2, axis=1)
f = open(str('train200.csv').rstrip(),'w')
f.write('id,label\n')	
i = 0
while i < score2.shape[0]:
	f.write(str(i) + ',' + str(score2[i]) + '\n')	
	i += 1
f.close()
