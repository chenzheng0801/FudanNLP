import tensorflow as tf
import os
import codecs

from data_utils import TextConverter, batch_generator
from Model import CharLSTM
from DataProcessing import encode_seq
from DataProcessing import get_batches


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_integer('epoches', 20, 'number of seqs in one batch')
tf.flags.DEFINE_integer('n_steps', 100, 'length of one seq')
tf.flags.DEFINE_integer('hidden_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', 'anna.txt', 'utf8 encoded text file')
tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 20, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')
tf.flags.DEFINE_integer('batch_size', 100, 'batch size of the data')
tf.flags.DEFINE_float('grad_clip', 5, 'gradient clip threshold')


def main(_):

    model_path = os.path.join('model', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    # vocab_to_int, int_to_vocab, encode_text\
    #     = encode_seq(FLAGS.input_file)

    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()
    converter = TextConverter(text, FLAGS.max_vocab)
    arr = converter.text_to_arr(text)
    g = batch_generator(arr, FLAGS.batch_size, FLAGS.n_steps)

    model = CharLSTM(batch_size=FLAGS.batch_size,
                     hidden_size=FLAGS.hidden_size,
                     num_layers=FLAGS.num_layers,
                     v_size=converter.vocab_size,
                     learning_rate=FLAGS.learning_rate,
                     grad_clip=FLAGS.grad_clip,
                     n_steps=FLAGS.n_steps)
    model.train(g, FLAGS.save_every_n,
                FLAGS.log_every_n, FLAGS.train_keep_prob,
                FLAGS.epoches, save_path=model_path)


if __name__ == '__main__':
    tf.app.run()
