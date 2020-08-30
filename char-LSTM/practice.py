import tensorflow as tf
import numpy as np

inp_data0 = tf.placeholder(tf.int32, shape=(3,2))
inp_data1 = tf.placeholder(tf.int32, shape=(None))

embed_vec = tf.constant([[1,2,3,4], [5,6,7,8]], dtype=tf.float32)
embedding0 = tf.nn.embedding_lookup(embed_vec, inp_data0)
embedding1 = tf.nn.embedding_lookup(embed_vec, inp_data1)

data_shape = (tf.shape(inp_data1)[0], 4)
state = tf.zeros(data_shape, dtype=tf.float32)
final_out = state + embedding1

matrix = np.array( [
    [[1, 2], [1, 2]],
    [[2, 3], [2, 3]],
    [[4, 5], [4, 5]]
])
a = np.array([[1,2], [1,2]])
b = np.array([[2,3], [2,3]])
c = np.array([[4,5], [4,5]])
x = [a, b, c]
y = tf.concat(matrix, 0)

tmp = tf.reshape(matrix, shape=[-1, 3])

with tf.Session() as sess:
    ans = sess.run(tmp)
    x = [1]
    feed = {
        inp_data1: x,
        state: [[1,1,1,1]]
    }
    out = sess.run(final_out, feed_dict=feed)
    print(out)
