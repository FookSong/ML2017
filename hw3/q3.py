#!/usr/bin/env python
# -- coding: utf-8 --

from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
def load_data():
    

    file = open('train.csv')
    content = file.readlines()
    x_test = []
    id_test = []
    for line in content:
        line = line.replace('\n','').split(',')
        id_test.append(line[0])
        x_test.append(line[1].split(' '))
    id_test = np.array(id_test[1:]).astype(np.int)
    x_test = np.array(x_test[1:]).astype(np.int)
    file.close()
    x_test = x_test[25839:]
    id_test = id_test[25839:]
   
    x_test = x_test/255.0

    return  (x_test,id_test)

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    (x_test,y_test)=load_data()
    x_test = x_test.reshape(x_test.shape[0],48,48,1)

    emotion_classifier = load_model('checkpoint-44-1.0679-0.6284.h5')
    np.set_printoptions(precision=2)
    dev_feats = x_test

    predictions = emotion_classifier.predict_classes(dev_feats)
    te_labels = y_test
    conf_mat = confusion_matrix(te_labels,predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    plt.show()
 
main()