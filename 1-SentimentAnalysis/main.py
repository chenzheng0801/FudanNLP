from FeatureExtraction import DataProcess
from model.ClassficationModel import LrModel
import time
import tensorflow as tf

def train(model, data_process):
    last_imporoved = 0
    best_acc_val = 0.0
    start_time = time.time()
    total_batch = 0

    saver = tf.train.Saver

    with tf.Session() as sess:
        pass


if __name__ == "__main__":
    data_process = DataProcess("train.tsv")
    model = LrModel(data_process.feature_dim, classes_num=5)
    train(model, data_process)