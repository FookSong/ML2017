#!/usr/bin/env python
# -- coding: utf-8 --

import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from utils import *
import numpy as np

def load_data():
    file = open('train.csv')
    content = file.readlines()
    x_train = []
    #y_train = []

    for line in content:
        line = line.replace('\n','').split(',')
       # y_train.append(line[0])
        x_train.append(line[1].split(' '))
    #y_train = np.array(y_train[1:]).astype(np.int)
    x_train = np.array(x_train[1:]).astype(np.int)
    print ("load finished")
    file.close()
    return x_train

def main():
    emotion_classifier = load_model('checkpoint-151-1.0418-0.6745.h5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    private_pixels = load_data()
    input_img = emotion_classifier.input
    name_ls = ["conv2d_2","conv2d_3","conv2d_4","conv2d_5","conv2d_6"]
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    # private_pixels = load_pickle('../fer2013/privateTest_pixels.pkl')
    # private_pixels = [ np.fromstring(private_pixels[i], dtype=float, sep=' ').reshape((1, 48, 48, 1)) 
    #                    for i in range(len(private_pixels)) ]
    filter_dir =  "G:\学习\Machine_ Learning\hw3\Q5"
    store_path = "q5_3img"
    choose_id = 28000
    photo = private_pixels[choose_id].reshape(1,48,48,1)
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt+2, choose_id))
        img_path = os.path.join(filter_dir, store_path)
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path,'layer{}'.format(cnt+2)))
if __name__ == "__main__":

    main()