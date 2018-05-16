# -*-coding:utf-8-*-
import tensorflow as tf
import os
import time
import configparser
from train_util import DataSet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# batch normalization: https://www.2cto.com/net/201707/653815.html

# 配置文件
cf = configparser.ConfigParser()
cf.read('config.cfg')


# 读取参数配置
tf_config = tf.ConfigProto()
tf_config.log_device_placement = int(cf.get('gpu', 'log_device_placement'))
tf_config.gpu_options.per_process_gpu_memory_fraction = float(cf.get('gpu', 'per_process_gpu_memory_fraction'))
tf_config.gpu_options.allow_growth = int(cf.get('gpu', 'allow_growth'))


# 读取常量配置
num_epochs = int(cf.get('model', 'num_epochs'))
num_hidden = int(cf.get('model', 'num_hidden'))
num_layers = int(cf.get('model', 'num_layers'))

INITIAL_LEARNING_RATE = float(cf.get('model', 'INITIAL_LEARNING_RATE'))
DECAY_STEPS = int(cf.get('model', 'DECAY_STEPS'))
LEARNING_RATE_DECAY_FACTOR = float(cf.get('model', 'LEARNING_RATE_DECAY_FACTOR'))  # The learning rate decay factor
MOMENTUM = float(cf.get('model', 'MOMENTUM'))  # 仅momentumOptmizer使用

BATCH_SIZE = int(cf.get('model', 'BATCH_SIZE'))
REPORT_STEPS = int(cf.get('model', 'REPORT_STEPS'))
SAVE_PER_EPOCH = int(cf.get('model', 'SAVE_PER_EPOCH'))


# 模型定义
def get_train_model(num_classes):

    inputs = tf.placeholder(tf.float32, [None, None, 20])
    inputs = tf.nn.l2_normalize(inputs, dim=0)  # 输入数据归一化处理，计划：引入Batch Normalization

    # 定义ctc_loss需要的稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32)

    # 1维向量 序列长度 [BATCH_SIZE,]
    seq_len = tf.placeholder(tf.int32, [None])

    # 定义LSTM网络
    def lstm_cell(hidden_size, keep_prob):
        cell = tf.contrib.rnn.LSTMCell(num_hidden, reuse=tf.get_variable_scope().reuse, state_is_tuple=True)     # 计划：增加可调参数
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)    # 计划：output_keep_prob加入可调参数
    stack = tf.contrib.rnn.MultiRNNCell([lstm_cell(num_hidden, keep_prob=0.5) for _ in range(num_layers)], state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    outputs = tf.reshape(outputs, [-1, num_hidden])
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")

    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [batch_s, -1, num_classes])
    logits = tf.transpose(logits, (1, 0, 2))

    return logits, inputs, targets, seq_len, W, b


# 训练
def train():

    def report_accuracy(decoded_list, test_targets, DataLoaderItem):

        #original_list = DataSet.decode_sparse_tensor(test_targets, DataLoaderItem.vocabulary)
        #detected_list = DataSet.decode_sparse_tensor(decoded_list, DataLoaderItem.vocabulary)
        true_numer = 0

        if len(original_list) != len(detected_list):
            print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                  " test and detect length desn't match")
            return
        print("T/F: original(length) <-------> detectcted(length)")
        for idx, number in enumerate(original_list):
            detect_number = detected_list[idx]
            hit = (number == detect_number)
            print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
            if hit:
                true_numer += 1
        print("Test Accuracy:", true_numer * 1.0 / len(original_list))

    def do_report(DataLoaderItem):

        test_inputs, test_targets, test_seq_len = next(DataLoaderItem.batch_iter())

        test_feed = {inputs: test_inputs,
                     targets: test_targets,
                     seq_len: test_seq_len}
        dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
        report_accuracy(dd, test_targets, DataLoaderItem)

    def do_batch(train_inputs, train_targets, train_seq_len, DataLoaderItem):

        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}

        b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, _ = session.run(
            [loss, targets, logits, seq_len, cost, global_step, optimizer], feed)

        # print b_loss
        # print b_targets, b_logits, b_seq_len
        # print(b_cost, steps)
        if steps > 0 and steps % REPORT_STEPS == 0:
            do_report(DataLoaderItem)
            # save_path = saver.save(session, "ocr.model", global_step=steps)
            # print(save_path)
        return b_cost, steps

    print("Traning start...\nLoading feature & label data...")
    data = DataSet(feats_index='feats/feats.json', label_file='lang/labels.json', batch_size=BATCH_SIZE)
    data.load()
    print("Data loading finished.")

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    num_classes = data.vocabulary_len + 1   # CTC blank

    logits, inputs, targets, seq_len, W, b = get_train_model(num_classes)

    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM)
    # optimizer = optimizer.minimize(loss, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(loss, global_step=global_step)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        for curr_epoch in range(num_epochs):
            print("Epoch.......", curr_epoch)
            train_cost = train_ler = 0
            start = time.time()
            for train_inputs, train_sparse_targets, train_seq_len in data.batch_iterator():
                c, steps = do_batch(train_inputs, train_sparse_targets, train_seq_len, data)
                train_cost += c * BATCH_SIZE
                seconds = time.time() - start
                print("Step:", steps, ", batch seconds:", seconds)

            train_cost /= data.num_train

            val_inputs, val_targets, val_seq_len = next(data.batch_iter())
            val_feed = {inputs: val_inputs,
                        targets: val_targets,
                        seq_len: val_seq_len}

            val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)

            log = "Epoch {}/{}, steps = {}, " \
                  "train_cost = {:.3f}, " \
                  "train_ler = {:.3f}, " \
                  "val_cost = {:.3f}, " \
                  "val_ler = {:.3f}, " \
                  "time = {:.3f}s, " \
                  "learning_rate = {}"

            print(log.format(curr_epoch,
                             num_epochs,
                             steps,
                             train_cost,
                             train_ler,
                             val_cost,
                             val_ler,
                             time.time() - start,
                             lr))

            if curr_epoch % SAVE_PER_EPOCH == 0:
                saver.save(session, 'model/rnn_e2e_epoch%d' % curr_epoch)


if __name__ == '__main__':
    train()