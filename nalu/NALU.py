import keras.backend as K
import tensorflow as tf
from keras.layers import Layer
from keras import initializers

class NALU(Layer):
    
    def __init__(self, units,
                 W_hat_initializer = 'glorot_uniform',
                 M_hat_initializer = 'glorot_uniform',
                 G_initializer = 'glorot_uniform',
                 epsilon=1e-7,
                 name=None,
                 **kwargs):
        
        super(NALU, self).__init__(**kwargs)
        
        self.units = units
        self.W_hat_initializer = initializers.get(W_hat_initializer)
        self.M_hat_initializer = initializers.get(M_hat_initializer)
        self.G_initializer = initializers.get(G_initializer)
        self.epsilon = epsilon
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.W_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.W_hat_initializer,
                                     name='W_hat')
        self.M_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.M_hat_initializer,
                                     name='M_hat')
        self.G = self.add_weight(shape=(input_dim, self.units),
                             initializer=self.G_initializer,
                             name='G')
        self.built = True
    
    def call(self, inputs):
        W = K.tanh(self.W_hat) * K.sigmoid(self.M_hat)
        a = K.dot(inputs, W)
        m = K.exp(K.dot(K.log(K.abs(inputs) + self.epsilon), W))
        g = K.sigmoid(K.dot(inputs, self.G))
        y = g*a + (1-g)*m
        
        return y

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
            'M_hat_initializer': initializers.serialize(self.M_hat_initializer),
            'G_initializer': initializers.serialize(self.M_hat_initializer),
            'epsilon': self.epsilon,
        }
        base_config = super(NALU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))