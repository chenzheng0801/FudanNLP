import tensorflow as tf
import os
import codecs

from Model import CharLSTM
from DataProcessing import TextConverter
from DataProcessing import get_batches


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_integer('epoches', 20, 'number of seqs in one batch')
tf.flags.DEFINE_integer('n_steps', 100, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', 'anna.txt', 'utf8 encoded text file')
tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 80, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')
tf.flags.DEFINE_integer('batch_size', 100, 'batch size of the data')
tf.flags.DEFINE_float('grad_clip', 5, 'gradient clip threshold')
tf.flags.DEFINE_string('mode', 'train', 'the train mode or the inference mode')
tf.flags.DEFINE_string('beg_chars', 'Everything was in confusion ', 'the given char sequence for inference')
tf.flags.DEFINE_string('ckpt_path', 'model/default', 'the checkpoint path for restoring model')

def main(_):
    mode = FLAGS.mode
    if mode == "train":
        model_path = os.path.join('model', FLAGS.name)
        if os.path.exists(model_path) is False:
            os.makedirs(model_path)
        with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
            text = f.read()
        converter = TextConverter(text)
        arr = converter.encode_seq(text)
        g = get_batches(arr, FLAGS.batch_size,
                        FLAGS.n_steps, converter.vocab_to_int['\n'])

        model = CharLSTM(batch_size=FLAGS.batch_size,
                         lstm_size=FLAGS.lstm_size,
                         num_layers=FLAGS.num_layers,
                         v_size=converter.vocab_size,
                         learning_rate=FLAGS.learning_rate,
                         grad_clip=FLAGS.grad_clip,
                         n_steps=FLAGS.n_steps,
                         embedding_size=FLAGS.embedding_size)
        model.train(g, FLAGS.save_every_n,
                    FLAGS.log_every_n, FLAGS.train_keep_prob,
                    FLAGS.epoches, save_path=model_path,
                    text_conveter=converter)
    elif mode == "inference":
        with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
            text = f.read()
        converter = TextConverter(text)
        beg_chars = FLAGS.beg_chars
        ckpt_path = FLAGS.ckpt_path
        n_samples = 20
        model = CharLSTM(batch_size=FLAGS.batch_size,
                         lstm_size=FLAGS.lstm_size,
                         num_layers=FLAGS.num_layers,
                         v_size=converter.vocab_size,
                         learning_rate=FLAGS.learning_rate,
                         grad_clip=FLAGS.grad_clip,
                         n_steps=FLAGS.n_steps,
                         embedding_size=FLAGS.embedding_size,
                         ckpt_path=ckpt_path)
        sentence = model.inference(beg_chars, n_samples, converter)
        print(sentence)


if __name__ == '__main__':
    tf.app.run()
