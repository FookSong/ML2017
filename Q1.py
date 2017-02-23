import numpy as np
import sys

data = []
A = np.loadtxt(sys.argv[1],dtype='int',delimiter = ',')
B = np.loadtxt(sys.argv[2],dtype='int',delimiter = ',')
cul = np.dot(A,B)

if cul.ndim != 1:
	for line in cul:
		data.extend(line)
		finaldata = np.sort(data)
else:
	finaldata = np.sort(cul)

np.savetxt("ans_one.txt",finaldata,fmt='%d')