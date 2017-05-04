#!/usr/bin/env python
# -- coding: utf-8 --

import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from utils import *
#from marcos import *
import numpy as np

# def plot_filt(image,nb_filter)
# 	fig = plt.figure(figsize=(14,8)) # 大小可自行決定
#     for i in range(nb_filter): # 畫出每一個filter
#         ax = fig.add_subplot(nb_filter/16,16,i+1) # 每16個小圖一行
#         ax.imshow(image,cmap='BuGn') # image為某個filter的output或最能activate某個filter的input image
#         plt.xticks(np.array([]))
#         plt.yticks(np.array([]))
#         plt.xlabel('whatever subfigure title you want') # 如果你想在子圖下加小標的話
#         plt.tight_layout()
#     fig.suptitle('Whatever title you want')
#     fig.savefig('filter.jpg') #將圖片儲存至disk

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def deprocess_image(x):
	# normalize tensor: center on 0., ensure std is 0.1
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	# clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)

	# convert to RGB array
	x *= 255
	if K.image_data_format() == 'channels_first':
		x = x.transpose((1, 2, 0))
	x = np.clip(x, 0, 255).astype('uint8')
	return x

def grad_ascent(num_step,input_image_data,iter_func):
	"""
	Implement this function!
	"""
	# run gradient ascent for 20 steps

	for i in range(160):
		loss_value, grads_value = iter_func([input_image_data,0])
		input_image_data += grads_value * num_step

		print('Current loss value:', loss_value)
		# if loss_value <= 0.:
		# # some filters get stuck to 0, we can skip them
		# 	break

	if loss_value > 0:
		input_image_data = deprocess_image(input_image_data)
	filter_images=((input_image_data, loss_value))
	return filter_images

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
	return (x_train, y_train)
def main():
	# (x_train,y_train)=load_data()
	emotion_classifier = load_model('checkpoint-151-1.0418-0.6745.h5')

	# get_layer_output = K.function([emotion_classifier.layers[0].input, K.learning_phase()],
	# 											[emotion_classifier.layers[3].output])

	# # output in test mode = 0
	# layer_output = get_layer_output([X, 0])[0]

	layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
	input_img = emotion_classifier.input

	print (layer_dict)
	name_ls = ["conv2d_2"]#,"conv2d_6"],"conv2d_4","conv2d_5","conv2d_6"
	nb_filter =32
	img_store = np.empty((1,nb_filter,48,48))
	filter_loss = np.empty((1,nb_filter,1))
	collect_layers = [ layer_dict[name].output for name in name_ls ]
	NUM_STEPS = 1
	RECORD_FREQ = 1 
	filter_dir =  "G:\学习\Machine_ Learning\hw3\Q5"
	store_path = "img"
	for cnt, c in enumerate(collect_layers): # level
		filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
		for filter_idx in range(nb_filter):
			input_img_data = np.random.random((1, 48, 48, 1)) # random noise
			input_img_data = (input_img_data - 0.5) * 20 + 128
			target = K.mean(c[:, :, :, filter_idx])
			grads = normalize(K.gradients(target, input_img)[0])
			iterate = K.function([input_img,K.learning_phase()], [target, grads])

			num_step = 0.1
			###
			#"You need to implement it."
			filter_imgs = grad_ascent(num_step, input_img_data, iterate)
			img_store[0,filter_idx,:,:] = filter_imgs[0].reshape((1, 1, 48, 48))
			###
			filter_loss[0,filter_idx,0] = filter_imgs[1]

			#print (filter_loss[0,filter_idx,0] )

		for it in range(NUM_STEPS//RECORD_FREQ):
			fig = plt.figure(figsize=(14, 8))
			for i in range(nb_filter):
				ax = fig.add_subplot(nb_filter/16, 16, i+1)
				ax.imshow(img_store[it,i], cmap='BuGn')#img
				plt.xticks(np.array([]))
				plt.yticks(np.array([]))
				plt.xlabel('{:.3f}'.format(float(filter_loss[it,i])))#loss
				plt.tight_layout()

			fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], it*RECORD_FREQ))
			img_path = os.path.join(filter_dir, '{}-{}'.format(store_path, name_ls[cnt]))
			if not os.path.isdir(img_path):
				os.mkdir(img_path)
			fig.savefig(os.path.join(img_path,'e{}'.format(it*RECORD_FREQ)))

if __name__ == "__main__":

	main()