import tensorflow as tf
import time

from Model import CharLSTM
from DataProcessing import encode_seq
from DataProcessing import get_batches

vocab_to_int, int_to_vocab, encode_text = encode_seq("anna.txt")

epoches = 20
per_iteration = 200
keep_pro = 0.5

model = CharLSTM(batch_size=64,
                 hidden_size=512,
                 num_layers=2,
                 v_size=len(vocab_to_int),
                 learning_rate=0.1,
                 grad_clip=5)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    new_state = sess.run(model.initial_state)
    loss = 0
    iteration = 0

    for epoch in range(1, epoches+1):
        batches = get_batches(encode_text, 10, 20, vocab_to_int["\n"])

        for batch_x, batch_y in batches:

