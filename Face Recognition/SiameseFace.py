# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:12:18 2019

@author: Kunal
"""

import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import Convolution2D,MaxPooling2D
from keras.layers import BatchNormalization
from keras import regularizers



#num_classes=40
#epochs = 60

def gen_range(n, start,end):
    return list(range(start, n)) + list(range(n+1, end))

def create_pairs(faces):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs=[]
    y=[]
    for k in range(40):
        for i in range(0,10):
            for j in range(i,10):
                pairs.append([faces[k,i],faces[k,j]])
                y.append(1)
                
    for i in range(40):
        r=gen_range(i,0,40)
        for j in range(10):
            for k in range(15):
                p=random.choice(r)
                q=random.randint(0,9)
                pairs.append([faces[p,q],faces[i,j]])
                y.append(0)
    
    pairs=np.array(pairs)
    y=np.array(y)
    return pairs,y

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)

def create_base_network2(input_shape):
    x = Input(shape=input_shape)
    y=Convolution2D(32,(5,5),padding='same',activation='relu')(x)
    y=BatchNormalization()(y)
    y=MaxPooling2D(pool_size=2,strides=2,padding='same')(y)
    
    y=Convolution2D(32,(5,5),padding='same',activation='relu')(y)
    y=BatchNormalization()(y)
    y=MaxPooling2D(pool_size=2,padding='same',strides=2)(y)
    
    y=Convolution2D(32,(5,5),padding='same',activation='relu')(y)
    y=BatchNormalization()(y)
    y=MaxPooling2D(pool_size=2,padding='same',strides=2)(y)
    
    
    y = Flatten()(y)
    
    y = Dense(1024, activation="relu",kernel_regularizer=regularizers.l2(0.1))(y)
    
    y = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.1))(y)
    
    return Model(x,y)


def create_base_network3(input_shape):
    x = Input(shape=input_shape)
    
    y=Convolution2D(64,(10,10),activation='relu', kernel_regularizer=regularizers.l2(2e-4))(x)
    y=BatchNormalization()(y)
    y=MaxPooling2D(pool_size=2,strides=2)(y)
    
    y=Convolution2D(128,(7,7),activation='relu', kernel_regularizer=regularizers.l2(2e-4))(x)
    y=BatchNormalization()(y)
    y=MaxPooling2D(pool_size=2,strides=2)(y)
    
    y=Convolution2D(64,(7,7),activation='relu', kernel_regularizer=regularizers.l2(2e-4))(x)
    y=BatchNormalization()(y)
    y=MaxPooling2D(pool_size=2,strides=2)(y)
    
    y=Convolution2D(32,(7,7),activation='relu', kernel_regularizer=regularizers.l2(2e-4))(x)
    y=BatchNormalization()(y)
    y=MaxPooling2D(pool_size=2,strides=2)(y)
    
    
    y = Flatten()(y)
    
    y = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(1e-3))(y)
    
    #y = Dense(256,activation='sigmoid',kernel_regularizer=regularizers.l2(1e-3))(y)
    
    return Model(x,y)
    

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

        






'''
def display(img,msg="Image"):
    cv2.imshow(msg,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#"E:\!KUNAL\ML\Face Recognition\DataSet\orl_faces"
faces=[]
ID_faces=[]
for i in range(1,41):
    face=[]
    for j in range(1,11):
        img=cv2.imread("DataSet/orl_faces/s"+str(i)+"/"+str(j)+".pgm",-1)
        face.append(img)
        ID_faces.append(i)
    faces.append(face)

display(faces[0][5])
faces=np.array(faces)
faces=faces.astype('float32')
faces /= 255
faces=np.expand_dims(faces,axis=4)
#faces=np.transpose(faces,(1,2,3,4,0))
train_pairs,train_y=create_pairs(faces)
X_train,X_test,y_train,y_test = train_test_split(train_pairs,train_y,test_size=0.2)

X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

import h5py
hf = h5py.File('facetrainingdata.h5', 'w')
hf.create_dataset('X_train', data=X_train)
hf.create_dataset('y_train', data=y_train)
hf.create_dataset('X_test', data=X_test)
hf.create_dataset('y_test', data=y_test)
hf.close()
'''
import h5py
hf=h5py.File('facetrainingdata.h5', 'r')
X_train=hf.get('X_train')
X_test=hf.get('X_test')
y_train=hf.get('y_train')
y_test=hf.get('y_test')

input_shape=X_train[20,1].shape

base_network = create_base_network3(input_shape)


input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])
output=Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(0.1))(distance)
model = Model([input_a, input_b], output)
#model.add(Dense(1),activation='sigmoid',kernel_regularizer=regularizers.l2(0.1))

rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

model.fit([X_train[:, 0], X_train[:, 1]], y_train,
          batch_size=32,
          epochs=50,
          validation_data=([X_test[:, 0], X_test[:, 1]], y_test))

y_pred = model.predict([X_train[:, 0], X_train[:, 1]])
tr_acc = compute_accuracy(y_train, y_pred)
y_pred = model.predict([X_test[:, 0], X_test[:, 1]])
te_acc = compute_accuracy(y_test, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

model.save("28-04-2019evenlargermodel.h5")

'''

winname='Image'
#cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)
cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''