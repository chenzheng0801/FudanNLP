import numpy as np
import tensorflow as tf
import time
import os


class CharLSTM:
    def __init__(self, batch_size, num_layers, lstm_size, v_size,
                 learning_rate, grad_clip, n_steps, embedding_size):
        self.batch_size, self.n_steps = batch_size, n_steps
        self.num_classes = v_size
        self.grad_clip = grad_clip
        self.n_steps = n_steps
        self.grad_clip = grad_clip
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.lstm_size = lstm_size
        self.embedding_size = embedding_size
        self.build_graph()

    def build_graph(self):
        #输入层
        self.build_model_input()
        self.build_lstm_layer()
        self.build_model_output()
        self.build_loss()
        self.build_optimizer()

    def build_lstm_layer(self):

        def get_a_cell(lstm_size, keep_prob):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
            return drop_cell

        with tf.name_scope("lstm_layer"):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            self.initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs,
                                                                    initial_state=self.initial_state)

    def build_model_input(self):
        with tf.name_scope("inputs"):
            self.inputs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.n_steps],
                                         name="input")
            self.targets = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.n_steps],
                                          name="targets")
            self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")

            embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
            self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

    def build_model_output(self):
        with tf.name_scope("output_layer"):
            seq_output = tf.concat(self.lstm_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.lstm_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope("loss"):
            y_one_hots = tf.one_hot(self.targets, self.num_classes)
            y_one_hots = tf.reshape(y_one_hots, self.prediction.get_shape())

            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=y_one_hots)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, save_every_n, log_every_n,
              train_keep_prob, epoches, save_path=None):
        step = 0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            new_state = sess.run(self.initial_state)
            start_time = time.time()
            loss = 0
            for batch_x, batch_y in batch_generator:
                step += 1
                feed = {self.inputs: batch_x,
                        self.targets: batch_y,
                        self.keep_prob: train_keep_prob,
                        self.initial_state: new_state}
                embedding = sess.run([self.lstm_inputs], feed_dict=feed)
                batch_loss, new_state, _ = sess.run([self.loss,
                                                     self.final_state,
                                                     self.train_op],
                                                    feed_dict=feed)
                loss += batch_loss
                if step % log_every_n == 0:
                    end_time = time.time()
                    print('step:{}'.format(step),
                          # 'epoc: {}/{}'.format(epoch, epoches),
                          '{:.4f} sec/batch'.format((end_time - start_time)/log_every_n),
                          'loss: {:.4f}'.format(loss/log_every_n))
                    loss = 0
                    start_time = time.time()

                if step % save_every_n == 0:
                    saver.save(sess, os.path.join(save_path, 'model'), global_step=step)