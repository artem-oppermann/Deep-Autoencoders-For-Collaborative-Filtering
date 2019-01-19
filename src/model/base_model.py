# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:15:00 2018

@author: Artem Oppermann
"""

import tensorflow as tf

class BaseModel(object):
        
    def __init__(self, FLAGS):
        
        self.weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2)
        self.bias_initializer=tf.zeros_initializer()
        self.FLAGS=FLAGS
    
    def _init_parameters(self):
        
        with tf.name_scope('weights'):
            self.W_1=tf.get_variable(name='weight_1', shape=(self.FLAGS.num_v,256), 
                                     initializer=self.weight_initializer)
            self.W_2=tf.get_variable(name='weight_2', shape=(256,128), 
                                     initializer=self.weight_initializer)
            self.W_3=tf.get_variable(name='weight_3', shape=(128,256), 
                                     initializer=self.weight_initializer)
            self.W_4=tf.get_variable(name='weight_4', shape=(256,self.FLAGS.num_v), 
                                     initializer=self.weight_initializer)
        
        with tf.name_scope('biases'):
            self.b1=tf.get_variable(name='bias_1', shape=(256), 
                                    initializer=self.bias_initializer)
            self.b2=tf.get_variable(name='bias_2', shape=(128), 
                                    initializer=self.bias_initializer)
            self.b3=tf.get_variable(name='bias_3', shape=(256), 
                                    initializer=self.bias_initializer)
    
    def inference(self, x):
        ''' Making one forward pass. Predicting the networks outputs.
        @param x: input ratings
        
        @return : networks predictions
        '''
        
        with tf.name_scope('inference'):
             a1=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, self.W_1),self.b1))
             a2=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a1, self.W_2),self.b2))
             a3=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a2, self.W_3),self.b3))   
             a4=tf.matmul(a3, self.W_4) 
        return a4