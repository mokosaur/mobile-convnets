from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def fire_module(inputs, filters, name='fire_module'):
    with tf.name_scope(name):
        x = slim.conv2d(inputs, int(filters * .125), [1, 1])
        x1 = slim.conv2d(x, filters // 2, [1, 1], activation_fn=None)
        x2 = slim.conv2d(x, filters // 2, [3, 3], padding='same', activation_fn=None)
        x = tf.concat([x1, x2], 3)
        return tf.nn.relu(x)


def squeezenet(inputs, num_classes=100, is_training=True, dropout_keep_prob=0.5, scope='squeezenet'):
    with tf.variable_scope(scope, 'squeezenet', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 96, [7, 7], 2)
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
            net = tf.layers.dropout(net, dropout_keep_prob, training=is_training)
            net = slim.conv2d(net, 1000, [1, 1], 1)
            net = tf.layers.flatten(net, name='flatten')
            net = tf.layers.dense(inputs=net, units=num_classes)

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points


def cnn_architecture(inputs, module, outputs, training_mode, num_class=4, learning_rate=0.001):
    x = inputs
    x = tf.layers.conv2d(x, 96, [7, 7], 2, activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, [3, 3], 2)
    x = module(x, 128)
    x = module(x, 128)
    x = module(x, 256)
    x = tf.layers.max_pooling2d(x, [3, 3], 2)
    x = module(x, 256)
    x = module(x, 384)
    x = module(x, 384)
    x = module(x, 512)
    x = tf.layers.max_pooling2d(x, [3, 3], 2)
    x = module(x, 512)
    x = tf.layers.dropout(x, training=training_mode)
    x = tf.layers.conv2d(x, 1000, [1, 1], 1, activation=tf.nn.relu)
    x = tf.layers.flatten(x, name='flatten')
    with tf.name_scope('logits'):
        logits = tf.layers.dense(inputs=x, units=num_class)

    with tf.name_scope('predictions'):
        predictions = {
            "classes": tf.argmax(logits, axis=1),
            "probabilities": tf.nn.softmax(logits)
        }

    with tf.name_scope('loss'):
        onehot_labels = tf.one_hot(indices=tf.cast(outputs, tf.int32), depth=num_class)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(predictions['classes'], tf.cast(outputs, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    loss_summary = tf.summary.scalar("loss_%s" % module.__name__, loss)
    acc_summary = tf.summary.scalar("accuracy_%s" % module.__name__, accuracy)

    return {
        "predictions": predictions,
        "loss": loss,
        "accuracy": accuracy,
        "optimizer": optimizer,
        "train_op": train_op,
        "loss_summary": loss_summary,
        "acc_summary": acc_summary
    }
