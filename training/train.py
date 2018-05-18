import tensorflow as tf
from training.models import squeezenet
from training.models import alexnet
from training.models.mobilenet import mobilenet_v2
import math
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


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


def generate_model_fn(topology_fn, num_classes=12, preprocessor_fn=gpu_augmentation_fn, arg_scope=lambda x: []):
    def model_fn(features, labels, mode):
        input = preprocessor_fn(features['x']) if preprocessor_fn else features['x']
        with tf.contrib.slim.arg_scope(arg_scope(mode == tf.estimator.ModeKeys.TRAIN)):
            logits, endpoints = topology_fn(input, num_classes=num_classes,
                                            is_training=mode == tf.estimator.ModeKeys.TRAIN)
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        acc = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        top3acc = tf.metrics.mean(tf.nn.in_top_k(predictions=logits,
                                                 targets=labels, k=3))

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", acc[1])
        tf.summary.scalar("top3accuracy", top3acc[1])
        tf.summary.image("input", input)
        tf.summary.histogram("softmax", predictions['probabilities'])

        if mode == tf.estimator.ModeKeys.TRAIN:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
                optimizer = tf.train.AdamOptimizer()
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
            logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=5)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                              training_hooks=[logging_hook])

        eval_metric_ops = {
            "accuracy": acc,
            "top3accuracy": top3acc,
            "confusion_matrix": eval_confusion_matrix(labels=labels, predictions=predictions["classes"],
                                                      num_classes=num_classes)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return model_fn


def parse_fn(example, mode):
    feature = {mode + '/image': tf.FixedLenFeature([], tf.string),
               mode + '/label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example, features=feature)
    image = tf.decode_raw(features[mode + '/image'], tf.float32)
    label = tf.cast(features[mode + '/label'], tf.int32)
    image = tf.reshape(image, [448, 448, 3])
    image = cpu_augmentation_fn(image, mode)
    return image, label


def generate_input_fn(dataset_name, mode='train', batch_size=16, repeat=True):
    def input_fn():
        dataset = tf.data.TFRecordDataset('../datasets/%s-%s.tfrecords' % (dataset_name, mode))
        dataset = dataset.shuffle(buffer_size=512)
        dataset = dataset.map(map_func=lambda x: parse_fn(x, mode), num_parallel_calls=4)
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=512)
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()

        features = {'x': images}
        return features, labels

    return input_fn


def cpu_augmentation_fn(image, mode):
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


models = {
    'squeezenet': {
        'architecture': squeezenet.squeezenet,
        'scope': lambda x: []
    },
    'mobilenet_v2': {
        'architecture': mobilenet_v2.mobilenet,
        'scope': lambda x: mobilenet_v2.training_scope(is_training=x)
    },
    'alexnet': {
        'architecture': alexnet.alexnet_v2,
        'scope': lambda x: alexnet.alexnet_v2_arg_scope()
    }
}

model_name = 'mobilenet_v2'
num_epochs = 5000
train_size = sum(1 for _ in tf.python_io.tf_record_iterator('../datasets/cladonia-train.tfrecords'))
batch_size = 16
epoch_size = train_size // batch_size

classifier = tf.estimator.Estimator(
    model_fn=generate_model_fn(models[model_name]['architecture'], arg_scope=models[model_name]['scope']),
    model_dir="./tmp/" + model_name,
    config=tf.estimator.RunConfig(save_summary_steps=epoch_size)
)

for e in range(num_epochs // 50):
    classifier.train(input_fn=generate_input_fn('cladonia', batch_size=batch_size), steps=epoch_size * 50)
    classifier.evaluate(input_fn=generate_input_fn('cladonia', mode='val', repeat=False))