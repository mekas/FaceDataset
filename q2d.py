""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
learning_rate = 0.003
batch_size = 512
n_epochs = 90
n_train = 60000
n_test = 10000

# Step 1: Read in data
mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)
#rint(type(train))
feature,label = train
#print(len(label))

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # if you want to shuffle your data
train_data = train_data.batch(batch_size)

# create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.shuffle(10000)
test_data = test_data.batch(batch_size)
#############################
########## TO DO ############
#############################


# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
#w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer(0, 0.01))
L = 200
M = 100
N = 60
O = 30

w = tf.get_variable("w", shape=(784,L), initializer=tf.random_normal_initializer(0,0.01))
b = tf.get_variable("b", shape=(1,L), initializer=tf.zeros_initializer())

w2 = tf.get_variable("w2", shape=(L,M), initializer=tf.random_normal_initializer(0,0.01))
b2 = tf.get_variable("b2", shape=(1,M), initializer=tf.zeros_initializer())

w3 = tf.get_variable("w3", shape=(M,N), initializer=tf.random_normal_initializer(0,0.01))
b3 = tf.get_variable("b3", shape=(1,N), initializer=tf.zeros_initializer())

w4 = tf.get_variable("w4", shape=(N,O), initializer=tf.random_normal_initializer(0,0.01))
b4 = tf.get_variable("b4", shape=(1,O), initializer=tf.zeros_initializer())

w5 = tf.get_variable("w5", shape=(O,10), initializer=tf.random_normal_initializer(0,0.01))
b5 = tf.get_variable("b5", shape=(1,10), initializer=tf.zeros_initializer())

pkeep = tf.get_variable("pkeep", initializer=tf.constant(1.0))
# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer

Y1 = tf.nn.relu(tf.matmul(img,w) + b)
Y1d = tf.nn.dropout(Y1, pkeep)
Y2 = tf.nn.relu(tf.matmul(Y1d,w2) + b2)
Y2d = tf.nn.dropout(Y2, pkeep)
Y3 = tf.nn.relu(tf.matmul(Y2d,w3) + b3)
Y3d = tf.nn.dropout(Y3, pkeep)
Y4 = tf.nn.relu(tf.matmul(Y3d,w4) + b4)
Y4d = tf.nn.dropout(Y4, pkeep)
logits = tf.matmul(Y4, w5) + b5
#logits = tf.matmul(img, w) + b
preds = tf.nn.softmax(logits)

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
#apply l2_regularization to the loss function with lambda=0.01
loss = tf.reduce_mean(entropy) * batch_size #+ 0.01*tf.nn.l2_loss(w)

#loss = tf.nn.l2_loss(entropy)
#############################
########## TO DO ############
#############################


# Step 6: define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
decay_rate = 0.98
global_step = tf.get_variable(name = "global_step", initializer=tf.constant(0))
decay_step = 10000
learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_step, decay_rate, staircase=True)
#############################
########## TO DO ############
#############################


# Step 7: calculate accuracy with test set
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
preds_clone = tf.identity(preds)
label_clone = tf.identity(label)
conf_mat = tf.contrib.metrics.confusion_matrix(tf.argmax(preds_clone,1), tf.argmax(label_clone,1))

#writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
   
    start_time = time.time()
    sess.run(tf.global_variables_initializer())    
    # train the model n_epochs times
    for i in range(n_epochs): 	
        sess.run(train_init, {pkeep: 0.5})	# drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('n_batches: {0}'.format(n_batches))            
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)			# drawing samples from test_data
    total_correct_preds = 0    
    total_cmat = np.zeros((10,10),dtype="int32")
    try:
        while True:
            accuracy_batch, cmat = sess.run([accuracy, conf_mat],{pkeep: 1.0})
            total_correct_preds += accuracy_batch
            
            total_cmat += cmat
    except tf.errors.OutOfRangeError:
        pass    
    print('Accuracy {0}'.format(total_correct_preds/n_test))
    print('Confusion Matrix \n {0}'.format(total_cmat))
#writer.close()
