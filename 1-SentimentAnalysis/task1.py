from sklearn.feature_extraction.text import CountVectorizer
from DataLoader import DataProcess
from model.ClassficationModel import LrModel
from datetime import timedelta
import tensorflow as tf
import time
import numpy as np
from sklearn.linear_model import LogisticRegression


def get_time_dif(start_time):
    """获取已经使用的时间"""
    end_time = time.time()
    time_dif = end_time-start_time
    return timedelta(seconds=int(round(time_dif)))


def train_by_sklearn():
    text_list, label_list = DataProcess.clean_data("train.tsv")
    count_vec = CountVectorizer()

    x_train, y_train = text_list, label_list

    x_count_train = count_vec.fit_transform(x_train)
    logist = LogisticRegression(penalty="none")
    logist.fit(x_count_train, y_train)
    x_test = x_count_train
    predicted = logist.predict(x_test)
    print(np.mean(predicted == y_train))

    # logistic = LogisticRegression(penalty="none")
    # text_list, label_list = DataProcess.clean_data("train.tsv")
    #
    # new_label_list = []
    # for labels in label_list:
    #     new_label_list.append(labels[0])
    #
    # feature_extraction = DataProcess.FeatureExtraction(text_list, label_list)
    # batch = feature_extraction.batch_iter(new_label_list, text_list, batch_size=8529)
    # for batch_x, batch_y in batch:
    #     logistic.fit(batch_x, batch_y)
    #
    # x_test = feature_extraction.fit_x(text_list)
    # y_test = new_label_list
    # predicted = logistic.predict(x_test)
    # print(np.mean(predicted == y_test))


def evaluate(sess, label_list, text_list, model, feature_extraction):
    batch = feature_extraction.batch_iter(label_list, text_list, batch_size=128)
    total = 0
    total_loss = 0
    total_acc = 0
    for batch_data, batch_one_hot in batch:
        accuracy, loss = sess.run([model.accuracy, model.cross_entropy],
                                  feed_dict={model.x: batch_data,
                                             model.y_: batch_one_hot})
        total_acc += accuracy
        total_loss += loss
        total += 1
    return total_acc/total, total_loss/total


def train_tensorflow():
    text_list, label_list = DataProcess.clean_data("train.tsv")
    feature_extraction = DataProcess.FeatureExtraction(text_list, label_list)
    feature_dim = feature_extraction.cont_vec.vocabulary_.__len__()
    model = LrModel(feature_dim, classes_num=5)
    epoch_num = 20

    best_acc_val = 0.0
    start_time = time.time()
    total_batch = 0
    saver = tf.train.Saver()
    print_per_batch = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoch_num+1):
            batch = feature_extraction.batch_iter(label_list, text_list, batch_size=128)
            for batch_x, batch_y in batch:
                if total_batch % print_per_batch == 0:
                    accuracy, loss = evaluate(sess, label_list, text_list, model, feature_extraction)
                    if accuracy > best_acc_val:
                        best_acc_val = accuracy
                        saver.save(sess, "./model.ckpt")
                    time_diff = get_time_dif(start_time)
                    print("")
                    print(time_diff)
                    print("epoch %d: total batch:%d loss %f, accuracy:%f"
                          % (epoch, total_batch, loss, accuracy))
                sess.run(model.train_op, feed_dict={model.x: batch_x, model.y_: batch_y})
                total_batch += 1


train_tensorflow()
# train_by_sklearn()