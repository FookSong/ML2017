import os
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt

def load_data():
	path = "./q1/"
	batch_size = 15
	listFiles = [f for f in os.listdir(path) if f[-4:] == ".bmp"]
	imgdata = []
	for bmpFile in listFiles:
		if bmpFile[0]=='K':
			break
		if  float(bmpFile[-6:-4]) < 10:
			#print(bmpFile)
			bmp = imread(os.path.join(path,bmpFile))
			imgdata.append(bmp.flatten())

	imgdata=np.array(imgdata)
	return imgdata

def plot_filt(image):
	fig = plt.figure(figsize=(64,64)) # 大小可自行決定
	for i in range(100): # 畫出每一個filter
		ax = fig.add_subplot(10,10,i+1) # 每16個小圖一行
		ax.imshow(image[i].reshape(64,64),cmap=plt.get_cmap('gray')) 
		plt.xticks(np.array([]))
		plt.yticks(np.array([]))
		#plt.xlabel('whatever subfigure title you want') # 如果你想在子圖下加小標的話
		#plt.tight_layout()
	#fig.suptitle('Whatever title you want')
	fig.savefig('e1.3.jpg')
	#fig.show()
data = load_data()


img_mean= data.mean(axis = 0,keepdims = True)
img_ctr = data-img_mean


u,s,v = np.linalg.svd(img_ctr,full_matrices=False)#v 4096*4096
for u in range(100):
    k=u+1
    v5=v[0:k]
    x_reduced=np.empty((k,100))
    n=0
    for i in v5:
        a=np.dot(img_ctr,i)
        x_reduced[n]=a
        n+=1
    reconstruct=np.dot(np.transpose(x_reduced),v5) +img_mean
                      
    RMSE = np.sqrt((np.sum((data-reconstruct)**2))/(4096*100*65535))
    print(RMSE)
    if RMSE<0.01:
        print(k)
        break
#plot_filt(reconstruct)
# plt.imshow(img_mean.reshape(64,64),cmap=plt.get_cmap('gray'))
# plt.show()