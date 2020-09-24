import os
import numpy as np
import tensorflow as tf
import glob

class tfrecord_reader(object):
    @staticmethod
    def N_of_tfrecord(filepath):
        tf_records_filenames = glob.glob(filepath + '/*.tfrecord')
        c = 0
        for fn in tf_records_filenames:
            for record in tf.python_io.tf_record_iterator(fn):
                c += 1
        return c

    @staticmethod
    def _image_preprocess(image):
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.subtract(image, 0.5)
        return image

    @staticmethod
    def _var_preprocess(feature):
        feature = tf.cast(feature, tf.int32)
        feature = tf.sparse_tensor_to_dense(feature)
        return feature

    @staticmethod
    def _fix_preprocess(feature):
        feature = tf.cast(feature, tf.int32)
        return feature

    @staticmethod
    def _parser(record):
        features = tf.parse_single_example(record,
                                           features={'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                                                     'image/cho': tf.VarLenFeature(tf.int64),
                                                     'image/jung': tf.VarLenFeature(tf.int64),
                                                     'image/jong': tf.VarLenFeature(tf.int64),
                                                     'image/width': tf.FixedLenFeature([1], tf.int64, default_value=1),
                                                     'text/length': tf.FixedLenFeature([1], tf.int64, default_value=1),
                                                     'image/boundary': tf.VarLenFeature(tf.int64),
                                                     'text/language': tf.VarLenFeature(tf.int64)
                                                     })

        features = (tfrecord_reader._image_preprocess(features['image/encoded']),
                    tfrecord_reader._var_preprocess(features['image/cho']),
                    tfrecord_reader._var_preprocess(features['image/jung']),
                    tfrecord_reader._var_preprocess(features['image/jong']),
                    tfrecord_reader._fix_preprocess(features['image/width']),
                    tfrecord_reader._fix_preprocess(features['text/length']),
                    tfrecord_reader._var_preprocess(features['image/boundary']),
                    tfrecord_reader._var_preprocess(features['text/language']))

        return features


    @staticmethod
    def build_random_batch(base_dir, batch_size):
        # List of lists ...

        def element_length_fn(a, b, c, d, e, f, g, h):
            return tf.shape(a)[1]

        files = tf.data.Dataset.list_files(base_dir+ '/*.tfrecord')
        dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(100),
                                   cycle_length=8)
        dataset = dataset.map(lambda record: tfrecord_reader._parser(record), num_parallel_calls=64)
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat(10)

        boundaries = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        dataset = dataset.apply(
                tf.data.experimental.bucket_by_sequence_length(
                    element_length_fn,
                    bucket_boundaries = boundaries,
                    bucket_batch_sizes = [batch_size] * (len(boundaries) + 1)
                    )
                )
        dataset = dataset.prefetch(1)


        iterator = dataset.make_one_shot_iterator()
        samples = iterator.get_next()
        return samples[0], samples[1], samples[2], samples[3], samples[4], samples[5], samples[6], samples[7]
