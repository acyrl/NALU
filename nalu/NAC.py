import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.layers import *
from keras.models import *

class NAC(Layer):
    
    def __init__(self,
                 units,
                 W_hat_initializer='glorot_uniform',
                 M_hat_initializer='glorot_uniform',
                 name=None,
                 **kwargs
                ):
        
        super(NAC, self).__init__(**kwargs)
        
        self.units = units
        self.W_hat_initializer = initializers.get(W_hat_initializer)
        self.M_hat_initializer = initializers.get(M_hat_initializer)
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.W_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.W_hat_initializer,
                                     name='W_hat')
        self.M_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.M_hat_initializer,
                                     name='M_hat')
        self.built = True
    
    def call(self, inputs):
        W = K.tanh(self.W_hat) * K.sigmoid(self.M_hat)
        return K.dot(inputs, W)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'W_hat_initializer': initializers.serialize(self.W_hat_initializer),
            'M_hat_initializer': initializers.serialize(self.M_hat_initializer)
        }
        base_config = super(NAC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))