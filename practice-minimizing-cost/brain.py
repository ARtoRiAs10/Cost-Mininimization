# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 01:41:10 2024

@author: gaurav
"""

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

#Building THE BRAINS

class Brain(object):
    
    def __init__(self, learning_rate = 0.001,number_actions = 5 ):
        self.learning_rate = learning_rate
        states = Input(shape = (3,))
        x = Dense(units = 64, activation = 'sigmoid')(states)
        y = Dense(units = 32, activation = 'sigmoid')(x)
        q_values = Dense(units = number_actions , activation= 'softmax')(y)
        
        self.model = Model(inputs = states, outputs = q_values)
        
        self.model.compile(loss = 'mse', optimizer = Adam(lr = learning_rate))
