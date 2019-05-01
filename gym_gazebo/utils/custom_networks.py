import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import CnnPolicy, FeedForwardPolicy, LstmPolicy, nature_cnn
from stable_baselines.a2c.utils import conv, linear, conv_to_fc



def norm_layer(input_tensor, scope, is_training=True, trainable=True, momentum=0.99, name="BatchNorm"):
    """
    Perform batch normalization on a layer
    :param input_tensor: (TensorFlow Tensor) The input tensor for the normalization
    :return: (Tensorflow Tensor) The normalized tensor
    """
    with tf.variable_scope(scope):
        return tf.layers.batch_normalization(input_tensor, training=is_training)
        # return  tf.keras.layers.BatchNormalization(epsilon=1E-5, center=True, scale=True, momentum=momentum, trainable=trainable, name=name)(input_tensor)
        # add_elements_to_collection(layer.updates, tf.GraphKeys.UPDATE_OPS)
        # return layer.apply(input_tensor, training=is_training)

def navigation_cnn(observation, **kwargs):
    """
    Modified CNN (Nature).

    :param observation: (TensorFlow Tensor) Image and relative distance to goal input placeholders
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """

    # np.shape(observation) = (1,84,85,1)
    print("Shape observation: ", np.shape(observation))
    scaled_images = observation[:, :, 0:-1, :]
    # With LIDAR
    scalar = observation[:, :, -1, :]
    # navigation_info = scalar[:, :, 0]  # Reshape in order to concatenate the arrays
    # WITHOUT LIDAR
    navigation_info = scalar[:, 0:3, :]  # Uncomment if you don't want use the lidar
    navigation_info = navigation_info[:, :, 0]  # Reshape in order to concatenate the arrays
    # TODO: navigation_info needs to be normalized in [0,1] like the scaled images
    # navigation_info = navigation_info * 255  / 3.14 # Denormalize the vector multiplying by ob_space.high and normalise on the bearing.high (3.14)
    print("Scaled images: ", np.shape(scaled_images))
    print("Scalar: ", np.shape(scalar))
    print("NavigationInfo: ", np.shape(navigation_info))

    activ = tf.nn.elu
    layer_1 = norm_layer(activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs)), 'c1_norm')
    print("Layer1: ", np.shape(layer_1))
    layer_2 = norm_layer(activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs)), 'c2_norm')
    print("Layer2: ", np.shape(layer_2))
    layer_3 = norm_layer(activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs)), 'c3_norm')
    print("Layer3: ", np.shape(layer_3))
    # layer_1 = (activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs)))
    # layer_2 = (activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs)))
    # layer_3 = (activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs)))
    # layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    # layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    # layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    print("Layer3: ", np.shape(layer_3))
    #layer_3 = tf.nn.sigmoid(layer_3)  # To squeeze values in [0,1]
    #print("L3: ", np.shape(layer_3))
    #print("NI: ", np.shape(navigation_info))
    fc_1 = tf.concat([layer_3, navigation_info], axis=1)
    #print("L3: ", np.shape(layer_3))
    # filter_summaries = tf.summary.merge([tf.summary.image("raw_observation", scaled_images, max_outputs=32),
    #                                           tf.summary.image("filters/conv1", layer_1,
    #                                                            max_outputs=32)])

    return layer_3, tf.nn.relu(linear(fc_1, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

class NavigationMlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction=None, **kwargs):
        super(NavigationMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256,
                                                reuse=reuse, net_arch=[256], act_fun=tf.nn.relu, feature_extraction= "none", **kwargs)

class NavigationCnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the navigation CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(NavigationCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        cnn_extractor=navigation_cnn, feature_extraction="cnn", **_kwargs)

class NavigationCnnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using LSTMs with a CNN feature extraction (the navigation CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(NavigationCnnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                        layer_norm=False, cnn_extractor=navigation_cnn, feature_extraction="cnn", **_kwargs)