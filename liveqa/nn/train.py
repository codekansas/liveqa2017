"""Training code for the question generator model.

To run:
    python -m liveqa.nn.train
"""

from __future__ import absolute_import

import datetime
import os
import sys

from .model import QuestionGenerator, QuestionGAN

from .. import yahoo

import tensorflow as tf

# Defines command line flags.
tf.app.flags.DEFINE_integer('batch_size', 100,
                            'size of each training minibatch.')
tf.app.flags.DEFINE_integer('batches_per_epoch', 100,
                            'number of minibatches per training epoch')
tf.app.flags.DEFINE_integer('nb_epoch', 30000,  # Goes through all data.
                            'number of epochs to train the model for')
tf.app.flags.DEFINE_bool('rebuild_model', True,
                         'if set, reset the model weights')
tf.app.flags.DEFINE_string('logdir', 'model/',
                           'directory where model is saved')
tf.app.flags.DEFINE_bool('use_gan', False,
                         'if set, use the GAN model instead')
FLAGS = tf.app.flags.FLAGS

def main(_):
    BATCH_SIZE = FLAGS.batch_size
    BATCHES_PER_EPOCH = FLAGS.batches_per_epoch
    NB_EPOCH = FLAGS.nb_epoch
    REBUILD_MODEL = FLAGS.rebuild_model
    LOGDIR = FLAGS.logdir

    # Interactive session to avoid unpleasant parts.
    sess = tf.Session()

    if FLAGS.use_gan:
        model = QuestionGAN(sess,
                            yahoo.ANSWER_MAXLEN,
                            yahoo.QUESTION_MAXLEN,
                            yahoo.NUM_TOKENS,
                            embeddings=yahoo.get_word_embeddings(),
                            logdir=LOGDIR)
        model.build(discriminator_phase=5).load(ignore_missing=True)
    else:
        model = QuestionGenerator(sess,
                                  yahoo.ANSWER_MAXLEN,
                                  yahoo.QUESTION_MAXLEN,
                                  yahoo.NUM_TOKENS,
                                  embeddings=yahoo.get_word_embeddings(),
                                  logdir=LOGDIR)
        model.load(ignore_missing=True)

    # raw_input('Press <ENTER> to begin training (%s)'
    #           % ('with gan' if FLAGS.use_gan else 'without gan'))

    # Gets the data iterator.
    data_iter = yahoo.iterate_answer_to_question(BATCH_SIZE, False)
    sample_iter = yahoo.iterate_answer_to_question(1, True)  # For visualization.

    total_start_time = datetime.datetime.now()
    for epoch_idx in xrange(1, NB_EPOCH + 1):
        start_time = datetime.datetime.now()
        for batch_idx in xrange(1, BATCHES_PER_EPOCH + 1):
            qsamples, asamples, qlens, alens = data_iter.next()
            if FLAGS.use_gan:
                g, d = model.train(qsamples, asamples, qlens, alens)
                sys.stdout.write('epoch %d: %d / %d gen acc = %.3f, '
                                 'dis acc = %.3f       \r'
                                 % (epoch_idx,
                                    batch_idx,
                                    BATCHES_PER_EPOCH,
                                    g, d))
            else:
                loss = model.train(qsamples, asamples, qlens, alens)
                sys.stdout.write('epoch %d: %d / %d loss = %.3e       \r'
                                 % (epoch_idx, batch_idx,
                                    BATCHES_PER_EPOCH, loss))
            sys.stdout.flush()

        time_passed = (datetime.datetime.now() - start_time).total_seconds()
        total_time_passed = (datetime.datetime.now()
                             - total_start_time).total_seconds()

        # Saves the current model weights.
        model.save()

        for i in range(5):
            qsamples, asamples, qlens, alens, refs = sample_iter.next()
            qpred = model.sample(asamples, alens)

            s = []
            s.append('epoch %d / %d, seen %d' %
                     (epoch_idx, NB_EPOCH,
                      epoch_idx * BATCH_SIZE * BATCHES_PER_EPOCH))
            s.append('%d (%d) seconds' % (time_passed, total_time_passed))
            s.append('answer: "%s"' % yahoo.detokenize(asamples[0], refs[0],
                                                       show_missing=True))
            s.append('target: "%s"' % yahoo.detokenize(qsamples[0], refs[0],
                                                       show_missing=True))
            s.append('pred: "%s"' % yahoo.detokenize(qpred[0], refs[0],
                                                     argmax=True,
                                                     show_missing=True))
            sys.stdout.write('\n   '.join(s) + '\n')

            if FLAGS.use_gan:
                model.print_summary(qsamples, asamples, qlens, alens)


if __name__ == '__main__':
    tf.app.run()
