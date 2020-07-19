import tensorflow as tf

class LrModel:
    def __init__(self, lr_config):
        self.config = lr_config
        self.logsitic_regression()

    def logsitic_regression(self):
        seq_length = self.config.seq_length
        classes_num = self.config.class_num
        self.x = tf.placeholder(tf.float32, [None, seq_length])
        W = tf.Variable(tf.random_normal([seq_length, classes_num]), name="weight")
        b = tf.Variable(tf.zeros([classes_num]), name="bias")
        y = tf.nn.softmax(tf.matmul(self.x, W) + b)

        self.pred_cls = tf.argmax(y,1)
        self.y_ = tf.placeholder(tf.float32, [None, classes_num])
        cross_entropy = tf.reduce_sum( self.y_ * tf.log(y))
