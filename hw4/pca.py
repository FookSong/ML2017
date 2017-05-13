import numpy as np
import sklearn.cluster as skc
import sys

def test(result):
	f = open(str(sys.argv[2]).rstrip(),'w')
	f.write('SetId,LogDim\n')	
	i = 0
	while i < result.shape[0]:
		f.write(str(i) + ',' + str(result[i]) + '\n')	
		i += 1
	f.close()


if __name__ == '__main__':
	r = np.load(sys.argv[1])
	varr=[]
	n=0
	for arr in r:
		varr.append(np.var(r[arr]))
		#print (n)
		n=n+1
	print ("pre-process ok")
	varr = np.array(np.transpose([varr]))
	print(varr.shape)
	k=60
	clf = skc.KMeans(n_clusters=k, n_init=10, verbose=1).fit(varr) 

	ind = np.argsort(clf.cluster_centers_, axis=None)
	clflabel = np.zeros((1,200))
	n=2
	for index in ind:
		#print(np.where(clf.labels_==index))
		clflabel[0,np.where(clf.labels_==index)] = n
		n += 1
	# print(ind)
	# print(clf.labels_)
	# print(np.log(clflabel))
	# print(np.sort(clf.cluster_centers_,axis=None))

	test(np.log(clflabel[0]))