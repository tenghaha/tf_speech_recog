# -*-coding:utf-8-*-
import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset
import numpy as np
import os
import json
import random

SHUFFLE_BUFFER = 501


# 数据集处理
class DataSet(TFRecordDataset):

    def __init__(self, data_path, vocabulary, batch_size):
        filenames = [file for file in os.listdir(data_path) if '.tfrecords' in file]
        random.shuffle(filenames)
        super(DataSet, self).__init__(filenames)

        self.batch_size = batch_size
        with open(vocabulary, 'r', encoding='utf8') as f:
            self.vocabulary = json.loads(f.read())

    def batch_iterator(self):
        dataset = self.map(self.__parse_function)
        batch_shuffled = dataset.shuffle(buffer_size=SHUFFLE_BUFFER)    # batch随机化
        padded_shapes = {'seq': [-1, ]}
        batch_shuffled_padded = batch_shuffled.padded_batch(self.batch_size, padded_shapes)  # padding input seqs
        b_iterator = batch_shuffled_padded.make_one_shot_iterator()
        # 转换batch为CTC所需形式
        '''
        for batch in b_iterator:
            b_seq_shape = tuple(np.shape(batch['seq'][0]))
            b_size_real = len(batch['seq'])
            inputs = np.zeros([b_size_real, b_seq_shape[0], b_seq_shape[1]])
            targets = []

            for k in range(b_size_real):
                inputs[k, :] = batch['seq'][k]
                targets.append(batch['label'][k])
            seq_len = np.ones(inputs.shape[0]) * b_seq_shape[0]

            yield inputs, targets, seq_len
        '''
        return b_iterator

    @staticmethod
    def __parse_function(example_proto):

        feat_dicts = {'seq_shape': tf.FixedLenFeature(shape=(2, ), dtype=tf.int64),
                      'label_len': tf.FixedLenFeature(shape=(1, ), dtype=tf.int64),
                      'seq': tf.VarLenFeature(dtype=tf.float32),
                      'label': tf.VarLenFeature(dtype=tf.int64)}
        parsed_example = tf.parse_single_example(example_proto, features=feat_dicts)

        #parsed_example[1]['seq'] = tf.sparse_tensor_to_dense(parsed_example[1]['seq'])
        #parsed_example[1]['seq'] = tf.reshape(parsed_example[1]['seq'], parsed_example[0]['seq_shape'])
        #parsed_example['label_len'] = parsed_example['label_len'][0]

        return parsed_example


if __name__ == '__main__':
    data = DataSet('data/train', 'data/lang/phones.json', 32)
    a = data.batch_iterator()
    for i in a:
        print(i)


"""


dataset = tf.contrib.data.TFRecordDataset(os.listdir('data/train/'))

def parse_function(example_proto):
    context_dicts = {'seq_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
                     'label_len': tf.FixedLenFeature(shape=(1,), dtype=tf.int64)}
    seq_dicts = {'seq': tf.VarLenFeature(dtype=tf.float32),
                 'label': tf.VarLenFeature(dtype=tf.int64)}
    parsed_example = tf.parse_single_sequence_example(example_proto,
                                                      context_features=context_dicts,
                                                      sequence_features=seq_dicts)

    # parsed_example[1]['seq'] = tf.sparse_tensor_to_dense(parsed_example[1]['seq'])
    # parsed_example[1]['seq'] = tf.reshape(parsed_example[1]['seq'], parsed_example[0]['seq_shape'])
    parsed_example[0]['label_len'] = parsed_example[0]['label_len'][0]

    return parsed_example

dataset = dataset.map(parse_function)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
"""