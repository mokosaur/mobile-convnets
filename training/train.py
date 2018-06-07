import tensorflow as tf
from training.models import squeezenet
from training.models import shallow_squeezenet
from training.models import alexnet
from training.models.mobilenet import mobilenet_v2
import math
import numpy as np
import cv2
import shutil
import os

tf.logging.set_verbosity(tf.logging.INFO)

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def eval_confusion_matrix(labels, predictions, num_classes=None):
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_classes)

        con_matrix_sum = tf.Variable(tf.zeros(shape=(num_classes, num_classes), dtype=tf.int32),
                                     trainable=False,
                                     name="confusion_matrix_result",
                                     collections=[tf.GraphKeys.LOCAL_VARIABLES])

        update_op = tf.assign_add(con_matrix_sum, con_matrix)

        return tf.convert_to_tensor(con_matrix_sum), update_op


def gpu_augmentation_fn(images):
    # images = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
    images = tf.map_fn(lambda img: tf.image.per_image_standardization(img), images)
    return images


def generate_model_fn(topology_fn, num_classes=12, preprocessor_fn=gpu_augmentation_fn, arg_scope=lambda x: [],
                      transfer_checkpoint_dir='', transfer_exclude=('Logits',), optimizer_name='adam',
                      lr_decay=None, initial_learning_rate=0.001, max_step=None):
    def model_fn(features, labels, mode):
        input = preprocessor_fn(features['x']) if preprocessor_fn else features['x']
        with tf.contrib.slim.arg_scope(arg_scope(mode == tf.estimator.ModeKeys.TRAIN)):
            logits, endpoints = topology_fn(input, num_classes=num_classes,
                                            is_training=mode == tf.estimator.ModeKeys.TRAIN)
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
            "logits": logits
        }

        variables_to_train = None
        if transfer_checkpoint_dir:
            print(tf.trainable_variables())
            variables_to_train = [v for v in tf.trainable_variables() if
                                  any('/%s/' % x in v.name for x in transfer_exclude)]
            variables_to_restore = [v for v in tf.trainable_variables() if
                                    not any('/%s/' % x in v.name for x in transfer_exclude)]
            print(variables_to_train)
            tf.train.init_from_checkpoint(transfer_checkpoint_dir,
                                          {v.name.split(':')[0]: v for v in variables_to_restore})

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        acc = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        top3acc = tf.metrics.mean(tf.nn.in_top_k(predictions=logits,
                                                 targets=labels, k=3))
        learning_rate = initial_learning_rate
        if lr_decay == 'wr':
            if not max_step: raise Exception("max_step must be set with lr_decay=wr")
            learning_rate = tf.train.cosine_decay_restarts(initial_learning_rate, tf.train.get_global_step(),
                                                           [max_step // 16, max_step // 16, max_step // 8,
                                                            max_step // 4, max_step // 2])
        elif lr_decay == 'exp':
            if not max_step: raise Exception("max_step must be set with lr_decay=exp")
            learning_rate = tf.train.exponential_decay(initial_learning_rate, tf.train.get_global_step(),
                                                       max_step, 0.96, staircase=True)

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", acc[1])
        tf.summary.scalar("top3accuracy", top3acc[1])
        tf.summary.scalar("learning rate", learning_rate)
        # tf.summary.image("input", input)

        if mode == tf.estimator.ModeKeys.TRAIN:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if optimizer_name == 'rmsprop':
                    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
                elif optimizer_name == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                else:
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step(),
                    var_list=variables_to_train)
            logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=5)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                              training_hooks=[logging_hook])

        confusion_matrix = eval_confusion_matrix(labels=labels, predictions=predictions["classes"],
                                                 num_classes=num_classes)

        eval_metric_ops = {
            "accuracy": acc,
            "top3accuracy": top3acc,
            "confusion_matrix": confusion_matrix
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return model_fn


def parse_fn(example, mode, channels):
    feature = {mode + '/image': tf.FixedLenFeature([], tf.string),
               mode + '/label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example, features=feature)
    image = tf.decode_raw(features[mode + '/image'], tf.float32)
    label = tf.cast(features[mode + '/label'], tf.int32)
    image = tf.reshape(image, [448, 448, channels])
    image = cpu_augmentation_fn(image, mode)
    return image, label


def generate_input_fn(dataset_name, mode='train', batch_size=1, shuffle_buffer=512, channels=3):
    def input_fn():
        dataset = tf.data.TFRecordDataset(os.path.join(__location__, '..', 'datasets',
                                                       '%s-%s.tfrecords' % (dataset_name, mode)))
        dataset = dataset.map(map_func=lambda x: parse_fn(x, mode, channels), num_parallel_calls=4)
        if mode == 'train':
            dataset = dataset.shuffle(buffer_size=shuffle_buffer)
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=1)
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()

        features = {'x': images}
        return features, labels

    return input_fn


def cpu_augmentation_fn(image, mode):
    with tf.device('/cpu:0'):
        if mode == 'train':
            image = tf.image.random_flip_left_right(image)
            image = tf.cond(
                tf.less(tf.random_uniform(()), 0.5),
                lambda: tf.contrib.image.rotate(
                    image,
                    tf.random_uniform((), -30 / 180 * math.pi, 30 / 180 * math.pi)
                ),
                lambda: tf.identity(image)
            )

            def crop(image):
                crop_size = tf.random_uniform((), 224, 448, tf.int32)
                x_offset = tf.random_uniform((), 0, 448 - crop_size, tf.int32)
                y_offset = tf.random_uniform((), 0, 448 - crop_size, tf.int32)
                return tf.image.crop_to_bounding_box(image, x_offset, y_offset, crop_size, crop_size)

            image = tf.cond(
                tf.less(tf.random_uniform(()), 0.5),
                lambda: crop(image),
                lambda: tf.identity(image)
            )
        image = tf.image.resize_images(image, (224, 224))
    return image


def cpu_augmentation_fn_old(image, mode):
    with tf.device('/cpu:0'):
        if mode == 'train':
            image = tf.image.random_flip_left_right(image)
            if np.random.uniform(0, 1.) < 0.5:
                angle = np.random.uniform(-30, 30) / 180 * math.pi
                image = tf.contrib.image.rotate(image, angle)
            if np.random.uniform(0, 1.) < 0.5:
                crop_size = np.random.random_integers(224, 448)
                x_offset = np.random.random_integers(0, 448 - crop_size)
                y_offset = np.random.random_integers(0, 448 - crop_size)
                image = tf.image.crop_to_bounding_box(image, x_offset, y_offset, crop_size, crop_size)
        image = tf.image.resize_images(image, (224, 224))
    return image


def copy_checkpoint(source, dest, global_step):
    source = os.path.join(__location__, source)
    dest = os.path.join(__location__, dest)
    checkpoint = '%s/model.ckpt-%s.' % (source, global_step)
    if not os.path.exists(dest):
        os.makedirs(dest)
    for format in ["data-00000-of-00001", "index", "meta"]:
        shutil.copy(checkpoint + format, dest + '/')


models = {
    'squeezenet': {
        'architecture': squeezenet.squeezenet,
        'scope': lambda x: squeezenet.training_scope(is_training=x)
    },
    'mobilenet_v2': {
        'architecture': mobilenet_v2.mobilenet,
        'scope': lambda x: mobilenet_v2.training_scope(is_training=x)
    },
    'alexnet': {
        'architecture': alexnet.alexnet_v2,
        'scope': lambda x: alexnet.alexnet_v2_arg_scope()
    },
    'small_squeezenet': {
        'architecture': shallow_squeezenet.small_squeezenet,
        'scope': lambda x: squeezenet.training_scope(is_training=x)
    },
    'tiny_squeezenet': {
        'architecture': shallow_squeezenet.tiny_squeezenet,
        'scope': lambda x: squeezenet.training_scope(is_training=x)
    },
}

dataset_name = 'cladonia'
model_name = 'squeezenet'
task_type = 'decay-learning'
num_epochs = 500
channels = 3

datasets_dir = os.path.join(__location__, '..', 'datasets')
train_size = sum(1 for _ in tf.python_io.tf_record_iterator(os.path.join(datasets_dir,
                                                                         '%s-train.tfrecords' % dataset_name)))
test_size = sum(1 for _ in tf.python_io.tf_record_iterator(os.path.join(datasets_dir,
                                                                        '%s-val.tfrecords' % dataset_name)))
batch_size = 16
epoch_size = train_size // batch_size
eval_every_n_epochs = 10

if __name__ == "__main__":
    if task_type == 'transfer-learning':
        model_fn = generate_model_fn(
            models[model_name]['architecture'], arg_scope=models[model_name]['scope'],
            transfer_checkpoint_dir=os.path.join(__location__, 'checkpoints', 'squeezenet'),
            transfer_exclude=('base_conv_2', 'Conv_23', 'Conv_22', 'Conv_21', 'Conv_20', 'Conv_19', 'Conv_18'),
            num_classes=10
        )
    elif task_type == 'decay-learning':
        model_fn = generate_model_fn(
            models[model_name]['architecture'], arg_scope=models[model_name]['scope'],
            lr_decay='wr',
            initial_learning_rate=1.,
            max_step=num_epochs * epoch_size
        )
    elif task_type == 'standard-learning':
        model_fn = generate_model_fn(
            models[model_name]['architecture'], arg_scope=models[model_name]['scope']
        )
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=os.path.join(__location__, "tmp", model_name),
        config=tf.estimator.RunConfig(save_summary_steps=epoch_size)
    )

    best_acc = 0.
    train_input_fn = generate_input_fn(dataset_name, batch_size=batch_size, shuffle_buffer=train_size,
                                       channels=channels)
    for e in range(num_epochs // eval_every_n_epochs):
        classifier.train(input_fn=train_input_fn, steps=eval_every_n_epochs * epoch_size)
        metrics = classifier.evaluate(input_fn=generate_input_fn(dataset_name, mode='val', channels=channels))
        np.save(os.path.join(__location__, "tmp", model_name, "cm-%s" % metrics['global_step']),
                metrics['confusion_matrix'])
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            copy_checkpoint('tmp/%s' % model_name, 'tmp/%s/best' % model_name, metrics['global_step'])
