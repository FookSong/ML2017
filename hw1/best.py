#coding: big5 
import numpy as np
import csv
import sys
import random

def savee(b,w1,w2,namda):
	f = open(str(namda)+'model.csv', 'w')
	f.close()
	f=open(str(namda)+'model.csv','a+')
	i=0
	while i < w1.shape[1]:
		f.write( 'w1 = ,' + str(w1[0,i]) + '\n')	
		i += 1
	i=0
	while i < w2.shape[1]:
		f.write( 'w2 = ,' + str(w2[0,i]) + '\n')	
		i += 1

	f.write('b = ,'+str(float(b)) + '\n')
	f.write('namda = ,'+str(namda) + '\n')

	f.close()
def testt(b,w1,w2,namda):
	#test data
	data = np.loadtxt(sys.argv[2], dtype=np.str, delimiter=",")   
	testid = data[9::18,0]
	testfea10 = data[9::18,2:].astype(np.float)
	#compute the result
	testdata=testfea10
	result = float(b) + np.sum(w1 * (testdata),axis=1) + np.sum(w2 * ((testdata)**2),axis=1)
	f = open(str(sys.argv[3]).rstrip(),'w')
	f.write('id,value\n')	
	i = 0
	while i < result.shape[0]:
		f.write(testid[i] + ',' + str(result[i]) + '\n')	
		i += 1
	f.close()
	#savee(b,w1,w2,namda)
#read train data
data = np.loadtxt(sys.argv[1] , dtype=np.str, delimiter=",")   
datapm = data[10::18,3:].astype(np.float).reshape(5760)

ydata = np.zeros([datapm.shape[0]-9 * 12,1]) #5751
iteration =1000000
lr = 1.0
idealloss = 0
havemodel = 0
train = 1
s
a = 1.0	#regularization

length = np.array(datapm).shape[0]#5760
feature = np.zeros([length - 9 * 12,9]) #5751*9

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
	model = np.loadtxt('bestmodel.csv' , dtype=np.str, delimiter=",")
	w1 = model[0:9,1].astype(np.float).reshape(9)
	w2 = model[9:18,1].astype(np.float).reshape(9)
	b = model[18,1].astype(np.float)
	#w1sum = model[154:307,1].astype(np.float).reshape(153)
	#w2sum = 
	#bsum = model[307,1].astype(np.float)
	namda = model[19,1].astype(np.float)
ii = 0
for ii in range(1):
	namda = 0.1
	w1 = np.random.rand(1,9)	# initial w1
	w2 = np.random.rand(1,9)	# initial w2
	b = random.random() # initia
	w1sum = np.zeros([1,9])
	w2sum = np.zeros([1,9])
	bsum = np.zeros([1,1])
	# use Adagrad to Iterations
	n=0
	for n in range(iteration):
	    # Store parameters for plotting
		temp = ydata.T[0] - (np.sum(w1 * feature,axis=1)  +float(b) + np.sum(w2 * (feature**2),axis=1) )
		#print str(temp)
		#+ np.sum(w2 * (feature**2),axis=1) + b)#1*5751
		LOSS = ((temp**2).sum() + namda * a * (w1**2).sum() + namda * a * (w2**2).sum())/5652.0		
		b_grad = (-2*(temp)*(1)).sum()
		w1_grad = (-2*(temp)*(feature.T)).sum(axis=1) + 2 * namda * w1 *a
		w2_grad = (-2*(temp)*((feature.T)**2)).sum(axis=1) + 2 * namda * w2 *a
		w1sum += w1_grad**2
		w2sum += w2_grad**2
		bsum +=  b_grad**2
		
		print 'iter: '+str(n)+' Loss: ' + str(LOSS) +'   ii :' + str (ii)

		# Update parameters.
		b = b - lr /np.sqrt(bsum + 1e-8)* b_grad
		w1 = w1 - lr /np.sqrt(w1sum + 1e-8)* w1_grad
		w2 = w2 - lr /np.sqrt(w2sum + 1e-8)* w2_grad
		if LOSS < idealloss:
			break

	testt(b,w1,w2,namda)
