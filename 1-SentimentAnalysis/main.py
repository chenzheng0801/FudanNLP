from FeatureExtraction import DataProcess
from datetime import timedelta
from model.ClassficationModel import LrModel
import time
import tensorflow as tf


def get_time_dif(start_time):
    """获取已经使用的时间"""
    end_time = time.time()
    time_dif = end_time-start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(sess, data_process, batch_size):

    batch = data_process.batch_iter(batch_size)
    doc_len = len(data_process.no_repeat_word_docs)

    total = int(round(doc_len/batch_size))

    total_loss = 0
    total_acc = 0
    for batch_data, batch_one_hot in batch:
        accuracy, loss = sess.run([model.accuracy, model.cross_entropy],
                                  feed_dict={model.x: batch_data,
                                             model.y_: batch_one_hot})
        total_acc += accuracy
        total_loss += loss
    return total_acc/total, total_loss/total


def train(model, data_process, epoch_num, batch_size=32):
    last_imporoved = 0
    best_acc_val = 0.0
    start_time = time.time()
    total_batch = 0
    saver = tf.train.Saver
    print_per_batch = 100

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoch_num+1):
            batch = data_process.batch_iter(batch_size)
            for batch_data, batch_one_hot in batch:
                if total_batch % print_per_batch == 0:
                    accuracy, loss = evaluate(sess, data_process, batch_size)
                    if accuracy > best_acc_val:
                        best_acc_val = accuracy
                        saver.save("model.ckpt")
                    time_dif = get_time_dif(start_time)
                    print("")
                    print(time_dif)
                    print("epoch %d: total batch:%d loss %f, accuracy:%f"
                          % (epoch, total_batch, loss, accuracy))
                sess.run(model.train_op, feed_dict={model.x: batch_data,
                                                    model.y_: batch_one_hot})
                total_batch += 1


if __name__ == "__main__":
    data_process = DataProcess("train.tsv")
    model = LrModel(data_process.feature_dim, classes_num=5)
    epoch_num = 10
    train(model, data_process, epoch_num)