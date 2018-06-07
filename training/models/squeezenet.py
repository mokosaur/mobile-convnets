from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import nn

slim = tf.contrib.slim


def fire_module(inputs, filters, name='fire_module'):
    with tf.name_scope(name):
        x = slim.conv2d(inputs, int(filters * .125), [1, 1])
        x1 = slim.conv2d(x, filters // 2, [1, 1], activation_fn=None)
        x2 = slim.conv2d(x, filters // 2, [3, 3], padding='same', activation_fn=None)
        x = tf.concat([x1, x2], 3)
        return nn.relu(x)


@slim.add_arg_scope
def squeezenet(inputs, num_classes=100, is_training=True, dropout_keep_prob=0.5, scope='squeezenet'):
    with tf.variable_scope(scope, 'squeezenet', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 96, [7, 7], 2, scope='base_conv_1')
            net = slim.max_pool2d(net, [3, 3], 2)
            net = fire_module(net, 128)
            net = fire_module(net, 128)
            net = fire_module(net, 256)
            net = slim.max_pool2d(net, [3, 3], 2)
            net = fire_module(net, 256)
            net = fire_module(net, 384)
            net = fire_module(net, 384)
            net = fire_module(net, 512)
            net = slim.max_pool2d(net, [3, 3], 2)
            net = fire_module(net, 512)
            net = slim.dropout(net, is_training=is_training)
            net = slim.conv2d(net, num_classes, [1, 1], 1, scope='base_conv_2')
            net = tf.reduce_mean(net, [1, 2])

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points


class NoOpScope(object):
    """No-op context manager."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def safe_arg_scope(funcs, **kwargs):
    """Returns `slim.arg_scope` with all None arguments removed.

    Arguments:
      funcs: Functions to pass to `arg_scope`.
      **kwargs: Arguments to pass to `arg_scope`.

    Returns:
      arg_scope or No-op context manager.

    Note: can be useful if None value should be interpreted as "do not overwrite
      this parameter value".
    """
    filtered_args = {name: value for name, value in kwargs.items()
                     if value is not None}
    if filtered_args:
        return slim.arg_scope(funcs, **filtered_args)
    else:
        return NoOpScope()


def training_scope(is_training=True, stddev=0.09, dropout_keep_prob=0.5, bn_decay=0.997):
    batch_norm_params = {
        'decay': bn_decay,
        'is_training': is_training
    }
    if stddev < 0:
        weight_intitializer = slim.initializers.xavier_initializer()
    else:
        weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_initializer=weight_intitializer,
            normalizer_fn=slim.batch_norm), \
         safe_arg_scope([slim.batch_norm], **batch_norm_params), \
         safe_arg_scope([slim.dropout], is_training=is_training,
                        keep_prob=dropout_keep_prob) as s:
        return s
