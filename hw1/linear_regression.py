import numpy as np
import csv
import sys
import random

def savee(n,b,w1):
	f = open(str(n)+'model.csv', 'w')
	f.close()
	f=open(str(n)+'model.csv','a+')
	i=0
	while i < w1.shape[1]:
		f.write( 'w1 = ,' + str(w1[0,i]) + '\n')	
		i += 1
	#i=0
	#while i < w2.shape[1]:
		#f.write( 'w2 = ,' + str(w2[0,i]) + '\n')	
		#i += 1

	f.write('b = ,'+str(float(b)) + '\n')

	f.close()
#read train data
data = np.loadtxt(sys.argv[1] , dtype=np.str, delimiter=",")
datapm = data[10::18,3:].astype(np.float).reshape(5760)
ydata = np.zeros([datapm.shape[0]-9*12,1]) #5751
iteration = 20000
lr = 10
idealloss = 0
havemodel = 0
train = 1

length = np.array(datapm).shape[0]#5760
feature = np.zeros([length - 9*12,9]) #5751*9

i = 0
store = 0
while i<=length - 9:	#0 - 5750
	if i % 480 in range(471,480):
		i += 1
		continue
	feature[store,0:9] = datapm[i:i+9]
	ydata[store] = datapm[i+9]  #ydata 5652
	i += 1
	store += 1
# ydata = b + w1 * datapm +w2*datapm*datapm

if havemodel:
	model = np.loadtxt('model.csv' , dtype=np.str, delimiter=",")
	w1 = model[0:9,1].astype(np.float).reshape(9)
	#w2 = model[9:18,1].astype(np.float).reshape(9)
	b = model[18,1].astype(np.float)
	w1sum = model[154:307,1].astype(np.float).reshape(153)
	#w2sum = 
	#bsum = model[307,1].astype(np.float)
else:
	w1 = np.random.rand(1,9)	# initial w1
	#2 = np.random.rand(1,9)	# initial w2
	b = random.random() # initial b
	w1sum = np.zeros([1,9])
	bsum = np.zeros([1,1])
if train:
	# use Adagrad to Iterations
	for n in range(iteration):
	    # Store parameters for plotting
		temp = ydata.T[0] - (np.sum(w1 * feature,axis=1) + float(b))#+ np.sum(w2 * (feature**2),axis=1)#1*5751
		LOSS = (temp**2).sum()/5652.0
		b_grad = (-2*(temp)*(1)).sum()
		bsum += b_grad**2
		w1_grad = (-2*(temp)*(feature.T)).sum(axis=1)
		w1sum += w1_grad**2 
		#2_grad = (-2*(temp)*((feature.T)**2)).sum(axis=1)
		print 'iter: '+str(n)+' Loss: ' + str(LOSS)
		b = b - lr /np.sqrt(bsum +1e-8) * b_grad
		w1 = w1 - lr /np.sqrt(w1sum +1e-8)* w1_grad
		#w2 = w2 - lr * w2_grad
		
		if LOSS < idealloss:
			break
	#savee(n,b,w1)
#test data
data = np.loadtxt(sys.argv[2] , dtype=np.str, delimiter=",")   
testid = data[9::18,0]
testfea10 = data[9::18,2:].astype(np.float)
#compute the result
testdata=testfea10
result = float(b) + np.sum(w1 * (testdata),axis=1)# + np.sum(w2 * ((testdata)**2),axis=1)

f = open(str(sys.argv[3]).rstrip(),'w')
f.write('id,value\n')	
i = 0
while i < result.shape[0]:
	f.write(testid[i] + ',' + str(result[i]) + '\n')	
	i += 1
f.close()
