#!/usr/bin/env python
# -- coding: utf-8 --

import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from vis.visualization import visualize_saliency

base_dir = "G:\学习\Machine_ Learning\hw3"
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
model_dir = os.path.join(base_dir, 'model')


def histeq(im,nbr_bins=256):

	#get image histogram
	imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
	cdf = imhist.cumsum() #cumulative distribution function
	cdf = 255 * cdf / cdf[-1] #normalize

	#use linear interpolation of cdf to find new pixel values
	im2 = np.interp(im.flatten(),bins[:-1],cdf)

	return im2

def load_data():
	file = open('train.csv')
	content = file.readlines()
	x_train = []
	y_train = []
	
	for line in content:
		line = line.replace('\n','').split(',')
		y_train.append(line[0])
		x_train.append(line[1].split(' '))
	y_train = np.array(y_train[1:]).astype(np.int)
	x_train = np.array(x_train[1:]).astype(np.int)

	file.close()

	#y_train = np_utils.to_categorical(y_train, 7)
	#x_train = x_train/255.0


	return (x_train, y_train)


def main():

	(x_train,y_train)=load_data()
	#x_train = x_train.reshape(x_train.shape[0],48,48,1)

	parser = argparse.ArgumentParser(prog='plot_saliency.py',description='ML-Assignment3 visualize attention heat map.')
	parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=80)
	args = parser.parse_args()
	model_name = "checkpoint-151-1.0418-0.6745.h5"
	# model_path = os.path.join(model_dir, model_name)
	emotion_classifier = load_model(model_name)
	print(colored("Loaded model from {}".format(model_name), 'yellow', attrs=['bold']))
	print('Model loaded.')


	# write_file=open('1.pkl','wb')
	# a=pickle.dump([[x_train,y_train]],write_file,-1)  
	# write_file.close()
	# read_file=open('1.pkl','rb')  
	# private_pixels=pickle.load(read_file)
	# read_file.close() 
	# private_pixels = [ np.fromstring(private_pixels[i], dtype=float, sep=' ').reshape((1, 48, 48, 1))  for i in range(len(private_pixels)) ]

	private_pixels = x_train.reshape(x_train.shape[0],48,48,1)
	input_img = emotion_classifier.input
	img_ids = [0]
	layer_name = 'dense_2'
	layer_idx = [idx for idx, layer in enumerate(emotion_classifier.layers) if layer.name == layer_name][0]
	nb_filter = 7

	for idx in img_ids:

		plt.figure()
		plt.imshow(x_train[idx].reshape(48,48),cmap = plt.cm.gray)
		plt.colorbar()
		plt.tight_layout()
		fig = plt.gcf()
		plt.draw()
	#	plt.show()
		# while 1:
		# 	pas=1
		fig.savefig(os.path.join(cmap_dir, 'origin{}.png'.format(idx)), dpi=100)

		val_proba = emotion_classifier.predict(private_pixels[idx].reshape(1,48,48,1))
		pred = val_proba.argmax(axis=-1)
		target = K.mean(emotion_classifier.output[:, pred])
		grads = K.gradients(target, input_img)[0]

		fn = K.function([input_img, K.learning_phase()], [grads])

		gradient = np.array(fn([x_train[idx].reshape(1,48,48,1),0]))[0,0]
		#print(gradient)
		# The name of the layer we want to visualize
		# (see model definition in vggnet.py)
		heatmap =[]
		for j in range(nb_filter):
			heatmap.append( visualize_saliency(emotion_classifier, layer_idx, [j], gradient, alpha=0.5))
			#print(heatmap.shape)
			# heatmap = histeq(private_pixels[idx].reshape(48*48))
			# heatmap = (private_pixels[idx].reshape(48,48) - np.min(heatmap))/(np.max(heatmap)-np.min(heatmap))
			'''
		    Implement your heatmap processing here!
		    hint: Do some normalization or smoothening on grads

		    '''
			fig = plt.figure(figsize=(14, 8))
		for j in range(nb_filter): # 畫出每一個filter
			ax = fig.add_subplot(1,16,j+1) # 每16個小圖一行
			ax.imshow(heatmap[j],cmap='BuGn') # image為某個filter的output或最能activate某個filter的input image
			plt.xticks(np.array([]))
			plt.yticks(np.array([]))
			plt.xlabel(j) # 如果你想在子圖下加小標的話
			plt.tight_layout()
		fig.suptitle('Whatever title you want')
		fig.savefig('filter.jpg') #將圖片儲存至disk




		thres = 0.40
		see = private_pixels[idx].reshape(48, 48)
		see[np.where(heatmap[:,:,1]/255 <= thres)] = 0

		plt.figure()
		plt.imshow(heatmap/255, cmap=plt.cm.jet)
		plt.colorbar()
		plt.tight_layout()
		fig = plt.gcf()
		plt.draw()
		fig.savefig(os.path.join(cmap_dir, 'privateTest{}.png'.format(idx)), dpi=100)

		plt.figure()
		plt.imshow(see,cmap='gray')
		plt.colorbar()
		plt.tight_layout()
		fig = plt.gcf()
		plt.draw()
		fig.savefig(os.path.join(partial_see_dir, 'privateTest{}.png'.format(idx)), dpi=100)
		# plt.show()
		# while 1:
		# 	a=1
if __name__ == "__main__":
	main()