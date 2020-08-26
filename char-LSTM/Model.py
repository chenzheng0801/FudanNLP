import numpy as np
import tensorflow as tf


class CharLSTM:
    def __init__(self, batch_size, num_layers, hidden_size, v_size, learning_rate, grad_clip):
        #输入层
        with tf.name_scope("inputs"):
            self.inputs, self.targets, self.keep_prob = self.build_model_input(v_size)
            embed_vec = tf.Variable(tf.random_normal([v_size, hidden_size],
                                                     mean=0, stddev=0.3), name="embed_vec")
            embedding = tf.nn.embedding_lookup(embed_vec, self.inputs, name="embedding")

        with tf.name_scope("lstm_layer"):
            cell, self.initial_state = self.build_lstm_layer(batch_size, num_layers,
                                                             hidden_size, self.keep_prob)
            outputs, state = tf.nn.dynamic_rnn(cell, embedding, initial_state=self.initial_state)
            self.final_state = state

        with tf.name_scope("output_layer"):
            self.prediction = self.build_output(outputs, hidden_size, v_size)

        self.loss = self.build_loss(self.prediction, v_size)
        self.train_op = self.build_optimizer(learning_rate, grad_clip)

    def build_lstm_layer(self, batch_size, num_layers, hidden_size, keep_prob):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        drop_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([drop_cell for _ in range(num_layers)])
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        return cell, initial_state

    def build_model_input(self, v_size):
        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input")
        targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="targets")
        keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        return inputs, targets, keep_prob

    def build_output(self, lstm_outputs, hidden_size, v_size):
        seq_out = tf.concat(lstm_outputs, axis=1)
        x = tf.reshape(seq_out, [-1, hidden_size])

        with tf.variable_scope("softmax"):
            softmax_w = tf.Variable(tf.truncated_normal([hidden_size, v_size], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(hidden_size))

        logits = tf.matmul(x, softmax_w) + softmax_b
        prediction = tf.nn.softmax(logits, name='prediction')
        return prediction

    def build_loss(self, prediction, v_size):
        y_one_hots = tf.one_hot(self.targets, v_size)
        y_one_hots = tf.reshape(y_one_hots, prediction.get_shape())

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_one_hots)
        loss = tf.reduce_mean(loss)
        return loss

    def build_optimizer(self, learning_rate, grad_clip):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        return train_op
