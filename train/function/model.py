from function.ops import *
from tensorflow.contrib.layers import variance_scaling_initializer

def Segmenter(image, width, is_train):
    with tf.variable_scope("Cutter"):
        s = image
        s = detail_conv_block(s, 'conv1', 6, [5, 5], [2, 1], is_train, False, None, 'relu', pad=[2, 2])
        s = detail_conv_block(s, 'conv2', 6, [5, 5], [2, 1], is_train, False, None, 'relu', pad=[2, 2])
        s = detail_conv_block(s, 'conv3', 6, [7, 3], [2, 1], is_train, False, None, 'relu', pad=[3, 1])
        s = detail_conv_block(s, 'conv4', 6, [5, 5], [2, 1], is_train, False, None, 'relu', pad=[2, 2])
        s = detail_conv_block(s, 'conv5', 6, [3, 3], [2, 1], is_train, False, None, 'relu', pad=[1, 1])

        s = tf.squeeze(s, 1)
        s = tf.transpose(s, perm=[1, 0, 2], name='time_major')
        s = rnn_layer(s, width, 8, 'bdrnn1')
        s = rnn_layer(s, width, 8, 'bdrnn2')
        s = tf.layers.dense(s, 1,
                            activation=None,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            bias_initializer=tf.constant_initializer(
                            value=0.0),
                            name='logits')
        s = tf.transpose(s, perm=[1, 0, 2], name='batch_major')
        logit = tf.squeeze(s, 2)
        return logit


def sig_ce_loss(logit, label):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = label, logits = logit)
    loss = tf.reduce_mean(loss)
    return loss

def policy_gradient(l2_reg, logit, opt, reward, residu, cutter_param, learning_rate, l2_lambda):
    logit = tf.clip_by_value(logit, 1e-6, 1-1e-6)
    log_logit_1 = tf.log(logit) * -reward
    log_logit_2 = tf.log(1-logit) * -residu
    policy_gradient = opt.compute_gradients(log_logit_1 + log_logit_2 + l2_lambda * l2_reg, cutter_param)
    return policy_gradient

def l2_regularization(name):
    glovars = tf.global_variables()
    vars = [var for var in glovars if name in var.name]
    weight_norm = [tf.nn.l2_loss(v) for v in vars if 'b' not in v.name ]
    weight_norm = tf.add_n(weight_norm)
    return weight_norm


def get_session_config():
    """Setup session config to soften device placement"""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

def get_checkpoint(model):
    """Get the checkpoint path from the given model output directory"""
    ckpt = tf.train.get_checkpoint_state(model)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = ckpt.model_checkpoint_path
    else:
        raise RuntimeError('No checkpoint file found')

    return ckpt_path

def get_init_trained(name):
    """Return init function to restore trained model from a given checkpoint"""
    saver_reader = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name))
    init_fn = lambda sess, ckpt_path: saver_reader.restore(sess, ckpt_path)
    return init_fn


flat_size = 1 * 1 * 120
def Recognizer_kor(image, is_train):
    """Recognizer for char image"""
    with tf.variable_scope("Recognizer_kor"):
        s = image #32x44
        s = detail_conv_block(s, 'conv1', 18, [5, 5], [1, 1], is_train, False, None, 'relu', pad=[0, 0]) #28x40
        s = pooling(s, [2,2], [2,2], 'MAX') #14x20
        s = detail_conv_block(s, 'conv2', 48, [5, 5], [1, 1], is_train, False, 'batch', 'relu', pad=[0, 0]) #10x16
        s = pooling(s, [2,2], [2,2], 'MAX') #5x8
        s = detail_conv_block(s, 'conv3', 360, [5, 8], [1, 1], is_train, False, 'batch', 'relu', pad=[0, 0]) #1x1

        s = tf.reshape(s, [-1, flat_size * 3])

        s_cho = mlp(s, 84, 'cho_fc5', is_train, False, 'batch', 'relu')
        cho = mlp(s_cho, 20,'cho_fc6', is_train, False, None, None)

        s_jung = mlp(s, 84, 'jung_fc5', is_train, False, 'batch', 'relu')
        jung = mlp(s_jung, 22,'jung_fc6', is_train, False, None, None)

        s_jong = mlp(s, 84, 'jong_fc5', is_train, False, 'batch', 'relu')
        jong = mlp(s_jong, 29,'jong_fc6', is_train, False, None, None)

        return cho, jung, jong


def Recognizer_eng(image, is_train):
    """Recognizer for char image"""
    with tf.variable_scope("Recognizer_eng"):
        s = image #32x44
        s = detail_conv_block(s, 'conv1', 6, [5, 5], [1, 1], is_train, False, None, 'relu', pad=[0, 0]) #28x40
        s = pooling(s, [2,2], [2,2], 'AVG') #14x20
        s = detail_conv_block(s, 'conv2', 16, [5, 5], [1, 1], is_train, False, 'batch', 'relu', pad=[0, 0]) #10x16
        s = pooling(s, [2,2], [2,2], 'AVG') #5x8
        s = detail_conv_block(s, 'conv3', 120, [5, 8], [1, 1], is_train, False, 'batch', 'relu', pad=[0, 0]) #1x1

        s = tf.reshape(s, [-1, flat_size])

        s = mlp(s, 84, 'fc5', is_train, False, 'batch', 'relu')
        result = mlp(s, 53,'fc6', is_train, False, None, None)

        return result



def Recognizer_spe(image, is_train):
    """Recognizer for char image"""
    with tf.variable_scope("Recognizer_spe"):
        s = image #32x44
        s = detail_conv_block(s, 'conv1', 6, [5, 5], [1, 1], is_train, False, None, 'relu', pad=[0, 0]) #28x40
        s = pooling(s, [2,2], [2,2], 'AVG') #14x20
        s = detail_conv_block(s, 'conv2', 16, [5, 5], [1, 1], is_train, False, 'batch', 'relu', pad=[0, 0]) #10x16
        s = pooling(s, [2,2], [2,2], 'AVG') #5x8
        s = detail_conv_block(s, 'conv3', 120, [5, 8], [1, 1], is_train, False, 'batch', 'relu', pad=[0, 0]) #1x1

        s = tf.reshape(s, [-1, flat_size])

        s = mlp(s, 84, 'fc5', is_train, False, 'batch', 'relu')
        result = mlp(s, 47,'fc6', is_train, False, None, None)

        return result



def Recognizer_chi(image, is_train):
    """Recognizer for char image"""
    with tf.variable_scope("Recognizer_chi"):
        s = image #32x44
        s = detail_conv_block(s, 'conv1', 12, [5, 5], [1, 1], is_train, False, None, 'relu', pad=[0, 0]) #28x40
        s = pooling(s, [2,2], [2,2], 'AVG') #14x20
        s = detail_conv_block(s, 'conv2', 32, [5, 5], [1, 1], is_train, False, 'batch', 'relu', pad=[0, 0]) #10x16
        s = pooling(s, [2,2], [2,2], 'AVG') #5x8
        s = detail_conv_block(s, 'conv3', 240, [5, 8], [1, 1], is_train, False, 'batch', 'relu', pad=[0, 0]) #1x1

        s = tf.reshape(s, [-1, flat_size * 2])

        s = mlp(s, 168, 'fc5', is_train, False, 'batch', 'relu')
        result = mlp(s, 1818,'fc6', is_train, False, None, None)

        return result

def Switch(image, is_train):
    """Recognizer for char image"""
    with tf.variable_scope("Switch"):
        s = image
        s = conv_block(s, 'conv1',  32, 3, 1, is_train, False, None, 'relu')
        s = pooling(s, [2,2], [2,2], 'MAX')

        s = conv_block(s, 'conv2',  64, 3, 1, is_train, False, 'batch', 'relu')
        s = pooling(s, [2,2], [2,2], 'MAX')

        s = conv_block(s, 'conv3', 128, 3, 1, is_train, False, 'batch', 'relu')
        s = pooling(s, [2,2], [2,2], 'MAX')

        s = conv_block(s, 'conv4', 256, 3, 1, is_train, False, 'batch', 'relu')
        s = pooling(s, [2,2], [2,2], 'MAX')

        s = conv_block(s, 'conv7', 512, 1, 1, is_train, False, 'batch', 'relu') #2,3,512

        s = tf.reshape(s, [-1, 2 * 3 * 512])
        s = mlp(s, 1024, 'fc7', is_train, False, None, 'relu')
        s = mlp(s, 256, 'fc8', is_train, False, None, 'relu')
        logit = mlp(s, 4,'fc9', is_train, False, None, None)
        return logit


