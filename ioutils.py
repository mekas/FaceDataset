import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import mnist2.mnist as mn
import numpy as np
import tensorflow as tf

def huber_loss(labels, predictions, delta=14.0):
    residual = tf.abs(labels - predictions)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, f1, f2)

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def get_mnist_dataset(batch_size, mnist_folder='convert_MNIST'):
    # Step 1: Read in data    
    #download_mnist(mnist_folder)
    #train, val, test = read_mnist(mnist_folder, flatten=False)    
    #(data, label) = test    
    mnist = mn.read_data_sets(mnist_folder, one_hot=True, num_classes=35, validation_size=0, channels=3)
    train_img = mnist.train.images    
    train_label = mnist.train.labels
    test_img = mnist.test.images    
    test_label = mnist.test.labels
    train = (train_img, train_label)
    test = (test_img, test_label)

    # Step 2: Create datasets and iterator
    train_data = tf.data.Dataset.from_tensor_slices(train)
    train_data = train_data.shuffle(10000) # if you want to shuffle your data
    train_data = train_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices(test)
    test_data = test_data.batch(batch_size)

    return train_data, test_data