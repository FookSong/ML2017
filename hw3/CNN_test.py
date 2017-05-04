import numpy as np
import sys
from keras.models import Sequential,load_model
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
	x_test = []
	id_test = []
	for line in content:
		line = line.replace('\n','').split(',')
		id_test.append(line[0])
		x_test.append(line[1].split(' '))
	id_test = np.array(id_test[1:]).astype(np.int)
	x_test = np.array(x_test[1:]).astype(np.int)
	file.close()

	cn=0
	for line in x_test:
		xdata = np.reshape(line,(48,48))
		if cn == 0:
			xmirror_test = [xdata[...,::-1].reshape(48*48)]
			cn += 1
			continue
		cn += 1
		#print (cn)
		xxdata = [xdata[...,::-1].reshape(48*48)]
		xmirror_test = np.append(xmirror_test,xxdata, axis=0)
		#print (xmirror_train)
	x_test = np.append(x_test,xmirror_test,axis=0)

 # convert class vectors to binary class matrices
	x_test = x_test/255.0

	return (x_test,id_test)


model2 = load_model('bestmodel.h5')
(x_test,id_test)=load_data()
x_test = x_test.reshape(x_test.shape[0],48,48,1)



score2 = model2.predict(x_test, batch_size=70, verbose=0)
score3 = score2[:7178,:]+score2[7178:,:]
score3 =  np.argmax(score3, axis=1)
f = open(str(sys.argv[2]).rstrip(),'w')
f.write('id,label\n')	
i = 0
while i < score3.shape[0]:
	f.write(str(i) + ',' + str(score3[i]) + '\n')	
	i += 1
f.close()
