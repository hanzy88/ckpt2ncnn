#coding:utf-8
import os
import numpy as np 
import tensorflow as tf
from ops import *
#import win_unicode_console
#win_unicode_console.enable()



def model(inputs, class_num):
    """
    vggExp for ExpGan
    input: the output of the pre-trained vggFace
    """
    with tf.variable_scope('comExp') as scope:
        
        conv0 = tf.nn.relu(conv2d(inputs, 16, 'conv0', kernel_size=5))
        conv1 = tf.nn.relu(conv2d(conv0, 16, 'conv1', kernel_size=5))
        max_pool0 = tf.nn.max_pool(conv1, [1,2,2,1], [1,2,2,1], padding='SAME')

        conv2 = tf.nn.relu(conv2d(max_pool0, 32, 'conv2', kernel_size=5))
        conv3 = tf.nn.relu(conv2d(conv2, 32, 'conv3', kernel_size=5))
        max_pool1 = tf.nn.max_pool(conv3, [1,2,2,1], [1,2,2,1], padding='SAME')

        conv4 = tf.nn.relu(conv2d(max_pool1, 64, 'conv4', kernel_size=5))
        conv5 = tf.nn.relu(conv2d(conv4, 64, 'conv5', kernel_size=5))
        max_pool2 = tf.nn.max_pool(conv5, [1,2,2,1], [1,2,2,1], padding='SAME')
        
        """
        #flatten = tf.contrib.layers.flatten(max_pool2)
        flatten = tf.reshape(max_pool2, [-1, 3136])
        fc1 = tf.nn.relu(fullyConnect(flatten, 64, name='fc_1'))
        fc2 = tf.nn.relu(fullyConnect(fc1, 64, name='fc_2'))
        
        output = fullyConnect(fc2, class_num, name='output')
        """
        output = conv2d(max_pool2, class_num, 'ouput', kernel_size = 7, padding='valid')
       
    return output
   
