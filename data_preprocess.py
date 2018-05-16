# -*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
from collections import Counter
import os
import json
import time
import argparse
import librosa


parser = argparse.ArgumentParser()
parser.add_argument('wav_list', help='wav list file')
parser.add_argument('text_file', help='text file')
parser.add_argument('-s', '--split', type=int, default=1000, help='features per TFRecords file, default 1000')
args = parser.parse_args()


# 读取wav.scp
def get_wav_files(wav_file):

    wav_dict = {}
    try:
        with open(wav_file, 'r',  encoding='utf8') as f:
            for wav in f:
                wav_id = wav.split('\t', 1)[0]
                wav_path = wav.split('\t', 1)[1]
                wav_dict[wav_id] = wav_path.strip('\n')
    except FileNotFoundError:
        print("ERROR: wav list file not found!")
        exit(1)

    return wav_dict


# 读取wav对应的label
def get_wav_label(wav_dict, label_file):

    labels_dict = {}
    try:
        with open(label_file, 'r', encoding='utf8') as f:
            for label in f:
                label = label.strip('\n')
                label_id = label.split('\t', 1)[0]
                label_text = label.split('\t', 1)[1]
                labels_dict[label_id] = label_text
    except FileNotFoundError:
        print("ERROR: text file not found!")
        exit(1)

    labels_dict_new = {}
    for idx in wav_dict.keys():
        if idx in labels_dict.keys():
            labels_dict_new[idx] = (labels_dict[idx])

    return labels_dict_new


# 获取词汇表并生成label seq
def get_label_seqs(labels_dict):

    all_words = []
    print("生成词汇表...")
    for label in labels_dict.values():
        curr_words = label.strip().split(' ')
        all_words += [word for word in curr_words]
    counter = Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    # 加入静音符号sil
    count_pairs_sil = [('sil', 0)]
    count_pairs_sil.extend(list(count_pairs))

    # print(count_pairs)

    words, _ = zip(*count_pairs_sil)
    words_size = len(words)
    print("词汇表大小:", words_size)

    label_seqs_dict = {}
    print("开始文本向量化...")
    word_num_map = dict(zip(words, range(len(words))))
    to_num = lambda word: word_num_map.get(word, len(words))
    for idx in labels_dict.keys():
        label_seqs_dict[idx] = list(map(to_num, labels_dict[idx].strip().split(' ')))
    print("文本向量化完成！")

    return label_seqs_dict, word_num_map


# 提取MFCC特征并保存为TFRecords格式
def get_and_save_feats(wav_dict, label_seqs_dict):

    total_start = time.time()
    start = time.time()
    tf_file_name = 'data/train/train_data_%.5d-of-%.5d.tfrecords'
    sp = 1
    num_file = int(len(wav_dict) / args.split) + 1 * bool(len(wav_dict) % args.split)
    num_wav = 0
    curr_file = tf_file_name % (sp, num_file)

    print("开始提取特征并保存至data/train目录...")
    try:
        os.makedirs('data/train')
    except FileExistsError:
        pass

    writer = tf.python_io.TFRecordWriter(curr_file)
    for idx in wav_dict.keys():
        # 每写入split个指向下一文件
        if (num_wav % args.split == 0 and num_wav != 0) or (num_wav == len(wav_dict)):
            writer.close()
            print("File: {} done. Time: {:.3f}s".format(curr_file, time.time() - start))
            sp += 1
            curr_file = tf_file_name % (sp, num_file)
            start = time.time()
            writer = tf.python_io.TFRecordWriter(curr_file)

        try:
            wav, sr = librosa.load(wav_dict[idx], mono=True)
        except FileNotFoundError:
            print("wav file not found or failed to load: ", wav_dict[idx])
            continue
        feat = np.transpose(librosa.feature.mfcc(wav, sr), [1, 0])
        feat_shape = list(np.shape(feat))
        feat_flat = feat.reshape([-1,])

        label = np.asarray(label_seqs_dict[idx])
        label_len = list(np.shape(label))

        feature = {'seq_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=feat_shape)),
                   'label_len': tf.train.Feature(int64_list=tf.train.Int64List(value=label_len)),
                   'seq': tf.train.Feature(float_list=tf.train.FloatList(value=feat_flat)),
                   'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        tf_serialized = example.SerializeToString()
        writer.write(tf_serialized)

        num_wav += 1

    writer.close()
    print("特征提取完成！ Total time: {:.3f}s".format(time.time() - total_start))

    return


# 保存词汇表
def save_vocabulary(vocabulary):

    try:
        os.makedirs('data/lang')
    except FileExistsError:
        pass

    file_name = 'data/lang/phones.json'
    with open(file_name, 'w') as f:
        f.write(json.dumps(vocabulary))
    print("vocabulary save to %s" % file_name)

    return


def main(wav_file, label_file):

    print("Start data preprocess...")
    wav_dict = get_wav_files(wav_file=wav_file)
    labels_dict = get_wav_label(wav_dict, label_file=label_file)
    label_seqs_dict, vocabulary = get_label_seqs(labels_dict)

    save_vocabulary(vocabulary)

    get_and_save_feats(wav_dict, label_seqs_dict)

    print("Data preprocess done!")


if __name__ == '__main__':

    main(args.wav_list, args.text_file)
