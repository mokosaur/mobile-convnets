import tensorflow as tf
import numpy as np
import os
import cv2
import math
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)

from datasets.dataset import load_data


def fire_module(inputs, filters, name='fire_module'):
    with tf.name_scope(name):
        x = tf.layers.conv2d(inputs, int(filters * .125), [1, 1], activation=tf.nn.relu)
        x1 = tf.layers.conv2d(x, filters // 2, [1, 1])
        x2 = tf.layers.conv2d(x, filters // 2, [3, 3], padding='same')
        x = tf.concat([x1, x2], 3)
        return tf.nn.relu(x)


def separable_conv(inputs, filters, batch_norm=True, name='separable_conv'):
    with tf.name_scope(name):
        n_channels = inputs.shape[3].value
        W = tf.Variable(tf.truncated_normal([3, 3, n_channels, 1], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[n_channels]))
        x = tf.nn.depthwise_conv2d(inputs, W, [1, 1, 1, 1], 'SAME') + b
        if batch_norm:
            x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters, [1, 1])
        if batch_norm:
            x = tf.layers.batch_normalization(x)
        return tf.nn.relu(x)


def conv(inputs, filters, name='convolution'):
    with tf.name_scope(name):
        return tf.layers.conv2d(inputs, filters, [3, 3], padding='same', activation=tf.nn.relu)


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


def augment(images, resize=None, angle=30):
    if resize:
        images = tf.image.resize_bilinear(images, resize)
    # batch_size = tf.shape(images)[0]
    # shp = tf.shape(images)
    # batch_size, height, width = shp[0], shp[1], shp[2]
    # width = tf.cast(width, tf.float32)
    # height = tf.cast(height, tf.float32)

    # angle_rad = angle * math.pi / 180
    # angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
    # images = tf.contrib.image.rotate(images, angles, interpolation='BILINEAR')
    # rotation = tf.contrib.image.angles_to_projective_transforms(angles, height, width)
    # images = tf.contrib.image.transform(images, rotation, interpolation='BILINEAR')
    # images = tf.contrib.image.rotate(images, angle * math.pi / 180, interpolation='BILINEAR')
    # images = tf.contrib.image.rotate(images, tf.random_uniform([batch_size], -angle, angle), interpolation='BILINEAR')

    images = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
    images = tf.map_fn(lambda img: tf.image.per_image_standardization(img), images)
    return images


# import tensorflow.contrib.slim.nets as nets
#
# with tf.Session() as sess:
#     x_input = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='images')
#     y_input = tf.placeholder(tf.float32, shape=[None], name='labels')
#
#     inception_net = nets.inception.inception_v3(x_input, num_classes=12)
#     # saver = tf.train.import_meta_graph('../../models/mobilenet_v1_0.5_224/mobilenet_v1_0.5_224.ckpt.meta',
#     #                                    clear_devices=True)
#     # print([n.name for n in tf.get_default_graph().as_graph_def().node])
#     # # print(tf.get_default_graph().as_graph_def())
#     # saver.restore(sess, '../../models/mobilenet_v1_0.5_224/mobilenet_v1_0.5_224.ckpt')

with tf.Session() as sess:
    x_input = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='images')
    y_input = tf.placeholder(tf.float32, shape=[None], name='labels')
    training_mode = tf.placeholder(tf.bool)

    x_input = augment(x_input)

    # tf.global_variables_initializer().run(session=sess)
    #
    # X_test, y_test = next(load_data("cladonia_preprocessed", 16, type='test', shuffled=False))
    # result = sess.run(output, feed_dict={x_input: X_test, y_input: y_test})
    #
    # # print(result)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.imshow(result[0])
    # plt.show()

    # cnn_conv = cnn_architecture(
    #     x_input,
    #     conv,
    #     y_input,
    #     training_mode,
    #     10
    # )

    cnn_squeeze = cnn_architecture(
        x_input,
        fire_module,
        y_input,
        training_mode,
        12
    )

    # cnn_mobile = cnn_architecture(
    #     x_input,
    #     separable_conv,
    #     y_input,
    #     training_mode,
    #     10
    # )

    tf.global_variables_initializer().run(session=sess)
    batch_size = 16
    epoch_size = 80 * 12 // batch_size

    fw = tf.summary.FileWriter('summary', sess.graph)
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2, max_to_keep=10)

    batch_generator = load_data("cladonia_preprocessed", batch_size)
    X_test, y_test = next(load_data("cladonia_preprocessed", 256, type='test'))

    best_ac = 0
    estimator, estimator_type = cnn_squeeze, "squeezenet"
    saver.export_meta_graph('tmp/model-graph.meta')
    with tf.name_scope('train'):
        for e in range(5000):
            for i in range(epoch_size):
                X_batch, y_batch = next(batch_generator)
                _, ls, loss_summary = sess.run(
                    [estimator["train_op"], estimator["loss"], estimator["loss_summary"]],
                    feed_dict={x_input: X_batch, y_input: y_batch, training_mode: True}
                )
                fw.add_summary(loss_summary, e * epoch_size + i)


            ac, acc_summary = sess.run(
                [estimator["accuracy"], estimator["acc_summary"]],
                feed_dict={x_input: X_test, y_input: y_test, training_mode: False}
            )
            fw.add_summary(acc_summary, e)
            print("[%s] epoch: %s, train loss: %s, val accuracy: %s" % (estimator_type, e, ls, ac))

            if ac > best_ac:
                best_ac = ac
                saver.save(sess, "tmp/model", global_step=e, write_meta_graph=False)

# segmentacja pierwszego planu
# augmentacja
# confusion matrix
# pca
# YUV LAB
# toronto metric? sparse regularization of dense layer at the end
# COTRASTIVE LOSS + CROSS-ENTROPY LOSS
# pipeline trasformation