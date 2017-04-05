#coding: big5 
import numpy as np
import csv
import sys
import random

def savee(b,w1):
	f = open('model.csv', 'w')
	f.close()
	f=open('model.csv','a+')
	i=0
	while i < w1.shape[1]:
		f.write( 'w1 = ,' + str(w1[0,i]) + '\n')	
		i += 1
	f.write('b = ,'+str(float(b)) + '\n')

	f.close()
def testt(b,w1): 
	#test data
	data = np.loadtxt(sys.argv[5], dtype=np.str, delimiter=",",skiprows = 1).astype(np.float)     
	feature = np.row_stack((data[:,0]/np.max(data[:,0]),data[:,1]/np.max(data[:,1]),data[:,2],data[:,3]/np.max(data[:,3]),data[:,4]/np.max(data[:,4]),data[:,5]/np.max(data[:,5])))
	feature = np.column_stack((feature.T,data[:,6:]))
	#compute the temp
	z = np.sum(w1 * feature,axis=1)  +float(b) #+ np.sum(w2 * (feature**2),axis=1) )
	temp = 1.0 / (1.0+np.exp(-z))
	temp = np.array([np.int(i + 0.5)/1 for i in temp])
	f = open((str(sys.argv[6])).rstrip(),'w')
	f.write('id,label\n')	
	i = 0
	while i < temp.shape[0]:
		f.write(str(i+1) + ',' + str(temp[i]) + '\n')	
		i += 1
	f.close()
	#savee(b,w1,n)
#read train data
data = np.loadtxt(sys.argv[3] , dtype=np.str, delimiter="," ,skiprows = 1).astype(np.float)   
ydata =  np.loadtxt(sys.argv[4] , dtype=np.str, delimiter=",").astype(np.float)

iteration =100000
lr = 0.001
idealloss = 0
havemodel = 1
train = 0
num=291000
a = 1.0	#regularization

feature = np.row_stack(((data[:,0]-np.min(data[:,0]))/(np.max(data[:,0])-np.min(data[:,0])),(data[:,1]-np.min(data[:,1]))/(np.max(data[:,1])-np.min(data[:,1])),data[:,2],(data[:,3]-np.min(data[:,3]))/(np.max(data[:,3])-np.min(data[:,3])),(data[:,4]-np.min(data[:,4]))/(np.max(data[:,4])-np.min(data[:,4])),(data[:,5]-np.min(data[:,5]))/(np.max(data[:,5])-np.min(data[:,5]))))
feature = np.column_stack((feature.T,data[:,6:]))

#print str(feature )
# ydata = sigmoid(wx+b)
if havemodel:
	model = np.loadtxt('model.csv' , dtype=np.str, delimiter=",")
	w1 = model[0:106,1].astype(np.float).reshape(106)
	#w2 = model[9:18,1].astype(np.float).reshape(9)
	b = model[106,1].astype(np.float)
	#namda = model[19,1].astype(np.float)
ii = 0
if train:#Adam
	for ii in range(1):
		namda = 0.1
		w1 = np.random.rand(1,feature.shape[1])	# initial w1
		#w2 = np.random.rand(1,9)	# initial w2
		b = random.random() # initia
		#w1sum = np.zeros([1,106])
		#w2sum = np.zeros([1,9])
		#bsum = np.zeros([1,1])
		# use Adagrad to Iterations
		n=0
		beta1 = 0.9
		beta2 = 0.999
		mg = np.zeros([1,feature.shape[1]])
		vg = np.zeros([1,feature.shape[1]])
		mb = np.zeros([1,1])
		vb = np.zeros([1,1])
		for n in range(iteration):
		    # Store parameters for plotting

			#randIndex = np.random.random_integers((feature.shape[0]-1)/4, size=(500))+np.int(feature.shape[0]/4)*(n%4)
			#itefea = feature[randIndex,:]
			#itey = ydata[randIndex]

			z = np.sum(w1[0] * feature,axis=1)  +float(b) #+ np.sum(w2 * (feature**2),axis=1) )
			#print str(np.sum(w1[0] * feature,axis=1).max() )
			temp = 1.0 / (1.0+np.exp(-z))
			temp = np.clip(temp,0.000000000000001,0.999999999999999)
			#+ np.sum(w2 * (feature**2),axis=1) + b)#1*5751
			LOSS = (-np.sum([np.log(temp).T * ydata + (1 - ydata) * np.log(1 - temp).T ] ) + (namda * a * (w1 **2)).sum())/feature.shape[0] 
			b_grad = (temp - ydata).sum()/feature.shape[0]
			w1_grad = (((temp - ydata)* feature.T).sum(axis=1) + 2 * namda * w1 *a) / feature.shape[0] 
			#w2_grad = (-2*(temp)*((feature.T)**2)).sum(axis=1) + 2 * namda * w2 *a
			#w1sum += w1_grad**2
			#w2sum += w2_grad**2
			mg = beta1 * mg + (1 - beta1) * w1_grad
			vg = beta2 * vg + (1 - beta2) * (w1_grad**2)
			mg_head = mg/(1- beta1**(n+1))
			vg_head = vg/(1- beta2**(n+1))

			mb = beta1 * mb+ (1 - beta1) * b_grad
			vb = beta2 * vb+ (1 - beta2) * (b_grad**2)
			mb_head = mb/(1- beta1**(n+1))
			vb_head = vb/(1- beta2**(n+1))
			if  n%100==0:
				print 'iter: '+str(n)+' Loss: ' + str(LOSS) #+'   err :' + str (ii)
			#blr = 1.0
			# Update parameters.
			b = b - lr/(np.sqrt(vb_head)+1e-8) * mb_head
			w1 = w1 - lr /(np.sqrt(vg_head)+1e-8) * mg_head
			#w2 = w2 - lr /np.sqrt(w2sum + 1e-8)* w2_grad
			if LOSS < idealloss:
				break

testt(b,w1)
