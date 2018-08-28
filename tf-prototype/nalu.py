import tensorflow as tf

def nalu(input_layer, num_outputs, epsilon=1e-6):
    """ Calculate the Neural Arithmetic Logic Unit (NALU).
    
    Arguments:
    input_layer - the input vector we want to the NALU of. 
    num_outputs - dimension of the output vector.
    epsilon - small shift to prevent log(0) 
    
    Returns:
    y - vector of dimension (X.shape.dims[0], num_outputs) 
    """
    
    shape = (input_layer.shape.dims[-1].value, num_outputs)
    
    with tf.name_scope("NALU"):
        W_hat = tf.Variable(tf.truncated_normal(shape, stddev=5), name="W_hat")
        M_hat = tf.Variable(tf.truncated_normal(shape, stddev=5), name="M_hat")
        G = tf.Variable(tf.truncated_normal(shape, stddev=0.02), name="G")
        
        W = tf.multiply(tf.tanh(W_hat), tf.sigmoid(M_hat)) 
        m = tf.exp(tf.matmul(tf.log(tf.abs(input_layer) + epsilon), W))

        a = tf.matmul(input_layer, W)
        g = tf.sigmoid(tf.matmul(input_layer, G))
        y = tf.multiply(g, a) + tf.multiply(1-g, m)
        
        return y