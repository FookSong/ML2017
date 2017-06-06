import numpy as np 
from keras.models import Sequential
from keras.layers import Reshape, Merge, Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras import regularizers
import pandas as pd
import sys
##################path##############
#train_path = '../train.csv'

test_path = sys.argv[1]+'test.csv'
    
output_path = sys.argv[2]

test = True
dot = False
nb_epoch = 1000
batch_size = 64
k_factors = 120#120
p_dropout = 0.25
split_ratio = 0.1
seed = 1446557
##################def##############
def load_data(path,training):
	x_train = []
	y_train = []

	with open(path,'r') as f:
		f.readline()#delete tag
		for line in f:
			line = line.replace('\n','').split(',')
			x_train.append(line[1:3])
			if training:
				y_train.append(line[3])

		x_train = np.array(x_train).astype(np.int)
		if training:
			y_train = np.array(y_train).astype(np.float)
	return (x_train,y_train)

def sort_category(id):
	patch = []
	for num in id:
		if num not in patch:
			patch.append(num)
	return patch

def RMSE(y_true,y_pred):
	return K.sqrt(K.mean(K.square(y_pred-y_true)))

def split_data(X,Y,split_ratio):
	indices = np.arange(X.shape[0])  
	np.random.shuffle(indices) 

	X_data = X[indices]
	Y_data = Y[indices]

	num_validation_sample = int(split_ratio * X_data.shape[0] )

	X_train = X_data[num_validation_sample:]
	Y_train = Y_data[num_validation_sample:]

	X_val = X_data[:num_validation_sample]
	Y_val = Y_data[:num_validation_sample]

	return (X_train,Y_train),(X_val,Y_val)

##################train##############
#(x_train,y_train) = load_data(train_path,True)
(x_test,_) = load_data(test_path,False)

#norm ratio
# y_mean = np.mean(y_train[:])
# y_std = np.std(y_train[:])
# y_train[:] = (y_train[:] - y_mean)/y_std
# print(y_std)
# print(y_mean)
# #shuffle
# (X_train,Y_train),(X_val,Y_val) = split_data(x_train,y_train,split_ratio)

# train_userid = X_train[:,0]
# train_movieid = X_train[:,1]

# val_userid = X_val[:,0]
# val_movieid = X_val[:,1]

# print('load_data finished')
# userid_cat = sort_category(train_userid)
# movieid_cat = sort_category(train_movieid)


# test_userid_cat = sort_category(test_userid)
# test_movieid_cat = sort_category(test_movieid)

#X_train = np.zeros((np.max(train_userid),np.max(train_movieid)))
#
#for i in range(y_train.shape[0]):
#	X_train[train_userid[i]-1,train_movieid[i]-1] = y_train[i]


#u,s,v = np.linalg.svd(X_train,full_matrices=False)


#################model###############
P = Sequential()
P.add(Embedding(6040, k_factors, input_length=1))
P.add(Reshape((k_factors,)))
Q = Sequential()
Q.add(Embedding(3952, k_factors, input_length=1))
Q.add(Reshape((k_factors,)))
model = Sequential()
###########################DOT
if dot:
	merged = Merge([P, Q], mode='dot', dot_axes=1)
	model.add(merged)

###########################DNN
else:
	merged = Merge([P, Q], mode='concat')
	model.add(merged)
	model.add(Dropout(p_dropout))
#	model.add(Dense(2*k_factors, activation='relu'))#,kernel_regularizer=regularizers.l2(0.01)
#	model.add(Dropout(p_dropout))
	model.add(Dense(k_factors, activation='relu'))#,kernel_regularizer=regularizers.l2(0.01)
	model.add(Dropout(p_dropout))
	model.add(Dense(1, activation='linear'))


model.summary()


adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
opt = SGD(lr=0.01,decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mse',
              optimizer='adamax',
              #metrics=['accuracy'])
               metrics=[RMSE])
earlystopping = EarlyStopping(monitor='val_RMSE', patience = 15, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath='best.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                            # monitor='val_acc',
                             monitor='val_RMSE',
                             mode='min')
if not test:
    history = model.fit([train_userid,train_movieid], Y_train, 
                    validation_data=([val_userid,val_movieid],Y_val), 
                     verbose=1,
                     epochs=nb_epoch, 
                     batch_size=batch_size,
                     callbacks=[earlystopping,checkpoint])

##################################plot
# loss = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
#                      'training': [ np.sqrt(loss) for loss in history.history['loss'] ],
#                      'validation': [ np.sqrt(loss) for loss in history.history['val_loss'] ]})
# ax = loss.ix[:,:].plot(x='epoch', figsize={7,10}, grid=True)
# ax.set_ylabel("root mean squared error")
# ax.set_ylim([0.0,3.0]);

# min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
# print ('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(np.sqrt(min_val_loss)))
##################test##############

test_userid = x_test[:,0]
test_movieid = x_test[:,1]
model.load_weights('rnn-best.hdf5')
Y_pred = model.predict([test_userid,test_movieid])
y_std = 1.116897661146206
y_mean = 3.5817120860388076
Y_pred = Y_pred*y_std+y_mean
Y_pred[np.where(Y_pred<1)]=1
Y_pred[np.where(Y_pred>5)]=5
#Y_pred = np.round(Y_pred)
with open(str(output_path).rstrip(),'w') as f:
	f.write('TestDataID,Rating\n')	
	i = 0
	while i < Y_pred.shape[0]:
		f.write(str(i+1) + ',' + str(Y_pred[i,0]) + '\n')	
		i += 1

