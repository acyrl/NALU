import tensorflow as tf

def nac(input_layer, num_outputs):
    """ Calculate the Neural Accumulator (NAC).
    
    Arguments:
    input_layer - the input vector we want to the NAC of. 
    num_outputs - dimension of the output vector.
    
    Returns:
    y - vector of dimension (X.shape.dims[0], num_outputs) 
    """
    
    shape = (input_layer.shape.dims[-1].value, num_outputs)
    
    with tf.name_scope("NAC"):
        W_hat = tf.Variable(tf.truncated_normal(shape, stddev=5), name="W_hat")
        M_hat = tf.Variable(tf.truncated_normal(shape, stddev=5), name="M_hat")
        W = tf.multiply(tf.tanh(W_hat), tf.sigmoid(M_hat))
        a = tf.matmul(input_layer, W)
        return a