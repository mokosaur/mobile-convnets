from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import nn
from training.models.squeezenet import *

slim = tf.contrib.slim


@slim.add_arg_scope
def small_squeezenet(inputs, num_classes=100, is_training=True, dropout_keep_prob=0.5, scope='small_squeezenet'):
    with tf.variable_scope(scope, 'small_squeezenet', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 96, [7, 7], 2, scope='base_conv_1')
            net = slim.max_pool2d(net, [3, 3], 2)
            net = fire_module(net, 128)
            net = fire_module(net, 256)
            net = slim.max_pool2d(net, [3, 3], 2)
            net = fire_module(net, 256)
            net = fire_module(net, 384)
            net = fire_module(net, 512)
            net = slim.max_pool2d(net, [3, 3], 2)
            net = fire_module(net, 512)
            net = slim.dropout(net, is_training=is_training)
            net = slim.conv2d(net, num_classes, [1, 1], 1, scope='base_conv_2')
            net = tf.reduce_mean(net, [1, 2])

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points


@slim.add_arg_scope
def tiny_squeezenet(inputs, num_classes=100, is_training=True, dropout_keep_prob=0.5, scope='tiny_squeezenet'):
    with tf.variable_scope(scope, 'tiny_squeezenet', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 96, [7, 7], 2, scope='base_conv_1')
            net = slim.max_pool2d(net, [3, 3], 2)
            net = fire_module(net, 128)
            net = slim.max_pool2d(net, [3, 3], 2)
            net = fire_module(net, 256)
            net = slim.max_pool2d(net, [3, 3], 2)
            net = fire_module(net, 512)
            net = slim.dropout(net, is_training=is_training)
            net = slim.conv2d(net, num_classes, [1, 1], 1, scope='base_conv_2')
            net = tf.reduce_mean(net, [1, 2])

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points
