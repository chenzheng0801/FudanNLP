import numpy as np
import tensorflow as tf
import time
import os


class CharLSTM:
    def __init__(self, batch_size, num_layers, lstm_size, v_size,
                 learning_rate, grad_clip, n_steps, embedding_size, ckpt_path=None):
        self.batch_size, self.n_steps = batch_size, n_steps
        self.num_classes = v_size
        self.grad_clip = grad_clip
        self.n_steps = n_steps
        self.grad_clip = grad_clip
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.lstm_size = lstm_size
        self.embedding_size = embedding_size
        self.session = None
        self.build_graph()
        self.saver = tf.train.Saver()
        if ckpt_path is not None:
            self.load(ckpt_path)

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
            batch_size = tf.shape(self.inputs)[0]
            self.initial_state = cell.zero_state(batch_size, dtype=tf.float32)
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs,
                                                                    initial_state=self.initial_state)

    def build_model_input(self):
        with tf.name_scope("inputs"):
            self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input")
            self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="targets")
            self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")

            embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
            self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

    def build_model_output(self):
        with tf.name_scope("output_layer"):
            x = tf.reshape(self.lstm_outputs, [-1, self.lstm_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope("loss"):
            y_one_hots = tf.one_hot(self.targets, self.num_classes)
            y_one_hots = tf.reshape(y_one_hots, [-1, self.num_classes])

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_one_hots)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, save_every_n, log_every_n,
              train_keep_prob, epoches, save_path=None, text_conveter=None):
        step = 0
        if self.session is None:
            self.session = sess = tf.Session()
        else:
            sess = self.session
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        loss = 0
        for epoch in range(1, epoches+1):
            batches = batch_generator()
            new_state = sess.run(self.initial_state,
                                 feed_dict={self.inputs: np.zeros(shape=(self.batch_size, 1))})
            for batch_x, batch_y in batches:
                step += 1
                feed = {self.inputs: batch_x,
                        self.targets: batch_y,
                        self.keep_prob: train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss,
                                                     self.final_state,
                                                     self.train_op],
                                                    feed_dict=feed)
                loss += batch_loss
                if step % log_every_n == 0:
                    end_time = time.time()
                    print('step:{}'.format(step),
                          'epoc: {}/{}'.format(epoch, epoches),
                          '{:.4f} sec/batch'.format((end_time - start_time)/log_every_n),
                          'loss: {:.4f}'.format(loss/log_every_n))
                    if text_conveter is not None:
                        pos = np.random.choice(self.batch_size, 1)[0]
                        x = batch_x[pos]
                        y = batch_y[pos]
                        print("****input sentence****\n{}".format(text_conveter.decode_seq(x)))
                        print("****target sentence****\n{}".format(text_conveter.decode_seq(y)))
                        sentence = self.inference("".join(text_conveter.decode_seq(x[0:50])), 50, text_conveter, sess)
                        print("****generated sentence 1****\n{}\n".format(sentence))
                        beg_chars = 'Everything was in confusion '
                        n_samples = 30
                        sentence = self.inference(beg_chars, n_samples, text_conveter)
                        print("****generated sentence 2****\n{}\n".format(sentence))
                    loss = 0
                    start_time = time.time()

                if step % save_every_n == 0 and save_path is not None:
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def inference(self, beginning_text, n_samples, text_converter, sess=None):

        samples = text_converter.encode_seq(beginning_text)
        vocab_size = text_converter.vocab_size
        if sess is None:
            sess = self.session

        new_state = sess.run(self.initial_state,
                             feed_dict={self.inputs: np.zeros(shape=(1, 1))})
        inp_x = np.array(samples).reshape((1, len(samples)))
        feed = {self.inputs: inp_x,
                self.keep_prob: 1.0,
                self.initial_state: new_state}
        predictions, new_state = sess.run([self.prediction,
                                           self.final_state],
                                          feed_dict=feed)
        c = pick_top_n(predictions, vocab_size)
        samples.append(c)
        for i in range(n_samples-1):
            inp_x = np.array([c]).reshape((1, 1))
            feed = {self.inputs: inp_x,
                    self.keep_prob: 1.0,
                    self.initial_state: new_state}
            predictions, new_state = sess.run([self.prediction,
                                               self.final_state],
                                              feed_dict=feed)
            c = pick_top_n(predictions, vocab_size)
            samples.append(c)
        text_seq = text_converter.decode_seq(samples)
        return ''.join(text_seq)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, tf.train.latest_checkpoint(checkpoint))
        print('Restored from: {}'.format(checkpoint))


def pick_top_n(predictions, vocab_size, top_n=5):
    preds = np.squeeze(predictions)
    if preds.ndim == 2:
        p = preds[-1, :]
    elif preds.ndim == 1:
        p = preds
    else:
        raise Exception('Incorrect prediction dimension!')

    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    res = np.random.choice(vocab_size, 1, p=p)[0]
    return res
