import tensorflow as tf
import numpy as np

def drop_out(input, keep_prob, is_train):
    if is_train:
        out = tf.nn.dropout(input, keep_prob)
    else:
        keep_prob = 1
        out = tf.nn.dropout(input, keep_prob)

    return out

def rnn_layer(bottom_sequence,sequence_length,rnn_size,scope):
    """Build bidirectional (concatenated output) RNN layer"""
    weight_initializer = tf.truncated_normal_initializer(stddev=0.01)

    cell_fw = tf.contrib.rnn.LSTMCell( rnn_size, initializer=weight_initializer)
    cell_bw = tf.contrib.rnn.LSTMCell( rnn_size , initializer=weight_initializer)
    rnn_output,_ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, bottom_sequence,
        sequence_length=sequence_length,
        time_major=True,
        dtype=tf.float32,
        scope=scope)

    rnn_output_stack = tf.concat(rnn_output,2,name='output_stack')
    return rnn_output_stack

def _norm(input, is_train, reuse=True, norm=None):
    assert norm in ['instance', 'batch', 'layer', None]
    if norm == 'instance':
        with tf.variable_scope('instance_norm', reuse=reuse):
            eps = 1e-5
            mean, sigma = tf.nn.moments(input, [1, 2], keep_dims=True)
            normalized = (input - mean) / (tf.sqrt(sigma) + eps)
            out = normalized
    elif norm == 'batch':
        with tf.variable_scope('batch_norm', reuse=reuse):
            out = tf.layers.batch_normalization(input, training=is_train, reuse=reuse)
    elif norm == 'layer':
        with tf.variable_scope('layer_norm', reuse=reuse):
            out = tf.contrib.layers.layer_norm(input)
    else:
        out = input
    return out

def _activation(input, activation=None):
    assert activation in ['relu', 'leaky', 'tanh', 'sigmoid', 'prelu', 'softplus', 'selu', None]
    if activation == 'relu':
        return tf.nn.relu(input)
    elif activation == 'leaky':
        return tf.contrib.keras.layers.LeakyReLU(0.2)(input)
    elif activation == 'tanh':
        return tf.tanh(input)
    elif activation == 'sigmoid':
        return tf.sigmoid(input)
    elif activation == 'prelu':
        alphas = tf.get_variable('alpha', input.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(input)
        neg = alphas * (input - abs(input)) * 0.5
        return pos + neg
    elif activation == 'softplus':
        return tf.nn.softplus(input)
    elif activation == 'selu':
        return tf.nn.selu(input)
    else:
        return input

def pooling(input, k_size, stride, mode):
    assert mode in ['MAX', 'AVG']
    if mode == 'MAX':
        aa = tf.nn.max_pool(value=input,
                          ksize=[1, k_size[0], k_size[1], 1],
                          strides=[1, stride[0], stride[1], 1],
                          padding='SAME',
                          name='max_pooling')
    else:
        aa = tf.nn.avg_pool(value=input,
                            ksize=[1, k_size[0], k_size[1], 1],
                            strides=[1, stride[0], stride[1], 1],
                            padding='SAME',
                            name='avg_pooling')
    return aa

def flatten(input):
    return tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])




def conv2d(input, num_filters, filter_size, stride, reuse=False, pad='SAME', dtype=tf.float32, bias=True):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, input.get_shape()[3], num_filters]
    w = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    if pad == 'REFLECT':
        p = (filter_size - 1) // 2
        x = tf.pad(input, [[0,0],[p,p],[p,p],[0,0]], 'REFLECT')
        conv = tf.nn.conv2d(x, w, stride_shape, padding='VALID')
    else:
        assert pad in ['SAME', 'VALID']
        conv = tf.nn.conv2d(input, w, stride_shape, padding=pad)
    b = tf.get_variable('b', [1,1,1,num_filters], initializer=tf.constant_initializer(0.0))
    conv = conv + b
    return conv

def conv_block(input, name, num_filters, k_size, stride, is_train, reuse, norm, activation, pad='SAME', bias=False):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d(input, num_filters, k_size, stride, reuse, pad, bias=bias)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out

def residual(input,  name, num_filters,  is_train, reuse, norm, activation, pad='SAME', bias=True):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('res1', reuse=reuse):
            out = conv2d(input, num_filters, 3, 1, reuse, pad, bias=bias)
            out = _norm(out, is_train, reuse, norm)
            out = _activation(out, activation)

        with tf.variable_scope('res2', reuse=reuse):
            out = conv2d(out, num_filters, 3, 1, reuse, pad, bias=bias)
            out = _norm(out, is_train, reuse, norm)
            out = _activation(out + input, activation)
        return out



def detail_conv2d(input, num_filters, filter_size, stride, reuse=False, pad='SAME', dtype=tf.float32, bias=True):
    stride_shape = [1, stride[0], stride[1], 1]
    filter_shape = [filter_size[0], filter_size[1], input.get_shape()[3], num_filters]
    w = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    if pad == 'REFLECT':
        ph = (filter_shape[0] - 1) // 2
        pw = (filter_shape[1] - 1) // 2
        x = tf.pad(input, [[0,0],[ph,ph],[pw,pw],[0,0]], 'REFLECT')
        conv = tf.nn.conv2d(x, w, stride_shape, padding='VALID')
    elif pad == 'SAME' or pad == 'VALID':
        conv = tf.nn.conv2d(input, w, stride_shape, padding=pad)
    else:
        ph = pad[0]
        pw = pad[1]
        x = tf.pad(input, [[0,0],[ph,ph],[pw,pw],[0,0]], 'CONSTANT')
        conv = tf.nn.conv2d(x, w, stride_shape, padding='VALID')
    b = tf.get_variable('b', [1,1,1,num_filters], initializer=tf.constant_initializer(0.0))
    conv = conv + b
    return conv

def detail_conv_block(input, name, num_filters, k_size, stride, is_train, reuse, norm, activation, pad='SAME', bias=False):
    with tf.variable_scope(name, reuse=reuse):
        out = detail_conv2d(input, num_filters, k_size, stride, reuse, pad, bias=bias)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out




def conv2d_transpose(input, num_filters, filter_size, stride, reuse, pad='SAME', dtype=tf.float32):
    n, h, w, c = input.get_shape().as_list()
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, num_filters, c]
    #output_shape = [n, h * stride, w * stride, num_filters]

    input_shape = tf.shape(input)
    try:  # tf pre-1.0 (top) vs 1.0 (bottom)
        output_shape = tf.pack([input_shape[0], stride * input_shape[1], stride * input_shape[2], num_filters])
    except Exception as e:
        output_shape = tf.stack([input_shape[0], stride * input_shape[1], stride * input_shape[2], num_filters])


    w = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    deconv = tf.nn.conv2d_transpose(input, w, output_shape, stride_shape, pad)
    return deconv

def deconv_block(input, name, num_filters, k_size, stride, is_train, reuse, norm, activation):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d_transpose(input, num_filters, k_size, stride, reuse)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out



def detail_conv2d_transpose(input, num_filters, filter_size, stride, reuse, pad='SAME', dtype=tf.float32):
    n, h, w, c = input.get_shape().as_list()
    stride_shape = [1, stride[0], stride[1], 1]
    filter_shape = [filter_size[0], filter_size[1], num_filters, c]
    #output_shape = [n, h * stride, w * stride, num_filters]

    input_shape = tf.shape(input)
    try:  # tf pre-1.0 (top) vs 1.0 (bottom)
        output_shape = tf.pack([input_shape[0], stride[0] * input_shape[1], stride[1] * input_shape[2], num_filters])
    except Exception as e:
        output_shape = tf.stack([input_shape[0], stride[0] * input_shape[1], stride[1] * input_shape[2], num_filters])

    w = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    deconv = tf.nn.conv2d_transpose(input, w, output_shape, stride_shape, pad)
    return deconv

def detail_deconv_block(input, name, num_filters, k_size, stride, is_train, reuse, norm, activation):
    with tf.variable_scope(name, reuse=reuse):
        out = detail_conv2d_transpose(input, num_filters, k_size, stride, reuse)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out




def dilated_conv(input, name, num_filters, k_size, stride, is_train, reuse, norm, activation):
    with tf.variable_scope(name, reuse=reuse):
        _, _, _, in_channels = input.get_shape().as_list()
        filter_shape = [k_size, k_size, in_channels, num_filters]

        filter = tf.get_variable("filter", filter_shape, dtype=tf.float32,  initializer=tf.random_normal_initializer(0, 0.02))
        out = tf.nn.atrous_conv2d(input, filter, stride, padding='SAME', name=name)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out

def dilated_deconv(input, name, filters, k_size, stride, is_train, reuse, norm, activation):
    with tf.variable_scope(name, reuse=reuse):
        filter_shape = [k_size, k_size, filters, filters]
        input_shape = tf.shape(input)
        try:  # tf pre-1.0 (top) vs 1.0 (bottom)
            output_shape = tf.pack([input_shape[0], stride * input_shape[1], stride * input_shape[2], filters])
        except Exception as e:
            output_shape = tf.stack([input_shape[0], stride * input_shape[1], stride * input_shape[2], filters])

        filter = tf.get_variable("filter", filter_shape, dtype=tf.float32, initializer=tf.random_normal_initializer(0.0, 0.02))
        out = tf.nn.atrous_conv2d_transpose(input, filter, output_shape, stride, padding='SAME', name=name)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out


def mlp(input, out_dim, name, is_train, reuse, norm=None, activation=None, dtype=tf.float32, bias=True):
    with tf.variable_scope(name, reuse=reuse):
        _, n = input.get_shape()
        w = tf.get_variable('w', [n, out_dim], dtype, tf.random_normal_initializer(0.0, 0.02))
        out = tf.matmul(input, w)

        b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))
        out = out + b
        out = _activation(out, activation)
        out = _norm(out, is_train, reuse, norm)
        return out

def dense_to_sparse(dense_tensor, sparse_val=0):
    """Inverse of tf.sparse_to_dense.

    Parameters:
        dense_tensor: The dense tensor. Duh.
        sparse_val: The value to "ignore": Occurrences of this value in the
                    dense tensor will not be represented in the sparse tensor.
                    NOTE: When/if later restoring this to a dense tensor, you
                    will probably want to choose this as the default value.
    Returns:
        SparseTensor equivalent to the dense input.
    """
    with tf.name_scope("dense_to_sparse"):
        sparse_inds = tf.where(tf.not_equal(dense_tensor, sparse_val),
                               name="sparse_inds")
        sparse_vals = tf.gather_nd(dense_tensor, sparse_inds,
                                   name="sparse_vals")
        dense_shape = tf.shape(dense_tensor, name="dense_shape",
                               out_type=tf.int64)
        return tf.SparseTensor(sparse_inds, sparse_vals, dense_shape)


