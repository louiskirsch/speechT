# Copyright 2016 Louis Kirsch. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np

import vocabulary
from speech_input import InputBatchLoader
from speech_model import Wav2LetterModel
from preprocess import SpeechCorpusReader

tf.app.flags.DEFINE_bool('relu', False, 'Use ReLU activation instead of tanh')
tf.app.flags.DEFINE_bool('power', False, 'Use a power spectrogram instead of mfccs as input')
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during evaluation.")
tf.app.flags.DEFINE_string("data_dir", "data/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "train/", "Training directory")
tf.app.flags.DEFINE_string("log_dir", "log/", "Logging directory for summaries")
tf.app.flags.DEFINE_string("run_name", "", "Give this training a name to appear in tensorboard")
tf.app.flags.DEFINE_integer("epoch_count", 1, "Number of epochs to evaluate")

FLAGS = tf.app.flags.FLAGS


def extract_decoded_ids(sparse_tensor):
  ids = []
  last_batch_id = 0
  for i, index in enumerate(sparse_tensor.indices):
    batch_id, char_id = index
    if batch_id > last_batch_id:
      yield ids
      ids = []
      last_batch_id = batch_id
    ids.append(sparse_tensor.values[i])
  yield ids


def create_model(session, input_size, speech_input):
  """Create speechT model and initialize or load parameters in session."""
  model = Wav2LetterModel(speech_input,
                          input_size,
                          vocabulary.SIZE + 1,
                          learning_rate=0,
                          learning_rate_decay_factor=0,
                          max_gradient_norm=0,
                          log_dir=FLAGS.log_dir,
                          use_relu=FLAGS.relu,
                          run_name=FLAGS.run_name,
                          momentum=0,
                          run_type='dev')
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
    model.init_session(session, init_variables=False)
  else:
    raise FileNotFoundError('No checkpoint for evaluation found')
  return model


def evaluate():
  # Use training sub-directory if not specified otherwise
  if FLAGS.run_name and FLAGS.train_dir == 'train/':
    FLAGS.train_dir += FLAGS.run_name + '/'

  # Determine feature type
  feature_type = 'power' if FLAGS.power else 'mfcc'

  reader = SpeechCorpusReader(FLAGS.data_dir)

  def create_sample_generator(limit_count=FLAGS.epoch_count * FLAGS.batch_size):
    return reader.load_samples('dev',
                               loop_infinitely=False,
                               limit_count=limit_count,
                               feature_type=feature_type)

  print('Determine input size from first sample')
  input_size = next(create_sample_generator(limit_count=1))[0].shape[1]

  print('Initialize InputBatchLoader')
  speech_input = InputBatchLoader(input_size, FLAGS.batch_size, create_sample_generator, FLAGS.epoch_count)

  with tf.Session() as sess:

    model = create_model(sess, input_size, speech_input)

    coord = tf.train.Coordinator()
    print('Starting input pipeline')
    tf.train.start_queue_runners(sess=sess, coord=coord)
    speech_input.start_threads(sess=sess, coord=coord)

    try:
      print('Begin evaluation')
      for epoch in range(FLAGS.epoch_count):
        if coord.should_stop():
          break

        global_step = model.global_step.eval()

        # Validate on development set and write summary
        avg_loss, decoded, label, summary = model.step(sess, update=False, decode=True, return_label=True, summary=True)
        model.summary_writer.add_summary(summary, global_step)
        perplexity = np.exp(float(avg_loss)) if avg_loss < 300 else float("inf")
        print("validation average loss {:.2f} perplexity {:.2f}".format(avg_loss, perplexity))
        for decoded_ids, label_ids in zip(extract_decoded_ids(decoded), extract_decoded_ids(label)):
          decoded_str = vocabulary.ids_to_sentence(decoded_ids)
          expected_str = vocabulary.ids_to_sentence(label_ids)
          print('expected: {}'.format(expected_str))
          print('decoded: {}'.format(decoded_str))

    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    coord.join()
    sess.close()


def main(_):
  evaluate()


if __name__ == "__main__":
  tf.app.run()
