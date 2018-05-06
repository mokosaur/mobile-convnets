import tensorflow as tf
from training.models import squeezenet
from training.models.mobilenet import mobilenet_v2

tf.logging.set_verbosity(tf.logging.INFO)


def generate_model_fn(topology_fn):
    def model_fn(features, labels, mode):
        logits, _ = topology_fn(features['x'], num_classes=12, is_training=mode == tf.estimator.ModeKeys.TRAIN)
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    return model_fn


def parse_fn(example):
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example, features=feature)
    image = tf.decode_raw(features['train/image'], tf.float32)
    label = tf.cast(features['train/label'], tf.int32)
    image = tf.reshape(image, [224, 224, 3])
    # image = _augment_helper(image)  # augments image using slice, reshape, resize_bilinear
    return image, label


def input_fn():
    # files = tf.data.Dataset.list_files("../datasets/cladonia-train.tfrecords")
    # dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = tf.data.TFRecordDataset('../datasets/cladonia-train.tfrecords')
    dataset = dataset.shuffle(buffer_size=512)
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=4)
    dataset = dataset.batch(batch_size=16)
    dataset = dataset.prefetch(buffer_size=512)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    features = {'x': images}
    return features, labels


classifier = tf.estimator.Estimator(
    model_fn=generate_model_fn(squeezenet.squeezenet),
    model_dir="./tmp/squeezenet"
)

classifier.train(input_fn=input_fn, steps=2)