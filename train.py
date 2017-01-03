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
import time
import os

import vocabulary
from wav2letter_model import Wav2LetterModel
from preprocess import SpeechCorpusReader

tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9,
                          "Learning rate decays by this much (multiplication).")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_bool('relu', False, 'Use ReLU activation instead of tanh')
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "data/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "train/", "Training directory")
tf.app.flags.DEFINE_string("log_dir", "log/", "Logging directory for summaries")
tf.app.flags.DEFINE_integer("limit_training_set", 0,
                            "Train on a smaller training set, limited to the specified size")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")

FLAGS = tf.app.flags.FLAGS

N_COEFFICIENTS = 13 * 3


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


def create_model(session):
  """Create speechT model and initialize or load parameters in session."""
  model = Wav2LetterModel(N_COEFFICIENTS,
                          vocabulary.SIZE + 1,
                          FLAGS.learning_rate,
                          FLAGS.learning_rate_decay_factor,
                          FLAGS.max_gradient_norm,
                          FLAGS.log_dir,
                          FLAGS.relu)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
    model.init_session(session, init_variables=False)
  else:
    print("Created model with fresh parameters.")
    model.init_session(session, init_variables=True)
  return model


def train():
  # Create training directory if it does not exist
  if not os.path.exists(FLAGS.train_dir):
    os.makedirs(FLAGS.train_dir)

  with tf.Session() as sess:
    model = create_model(sess)

    reader = SpeechCorpusReader(FLAGS.data_dir)

    def batch(iterable, batch_size):
      args = [iter(iterable)] * batch_size
      return zip(*args)

    sample_generator = reader.load_samples('train', loop_infinitely=True, limit_count=FLAGS.limit_training_set)
    dev_sample_generator = reader.load_samples('dev', loop_infinitely=True)

    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []

    for sample_batch in batch(sample_generator, FLAGS.batch_size):
      input_list, label_list = zip(*sample_batch)

      start_time = time.time()
      avg_loss, _ = model.step(sess, input_list, label_list)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += avg_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        global_step = model.global_step.eval()

        # Print statistics for the previous epoch.
        perplexity = np.exp(float(avg_loss)) if avg_loss < 300 else float("inf")
        print("global step {:d} learning rate {:.4f} step-time {:.2f} average loss {:.2f} perplexity {:.2f}"
              .format(global_step, model.learning_rate.eval(), step_time, avg_loss, perplexity))

        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)

        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "speechT.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0

        # Generate and store summary
        avg_loss, summary = model.step(sess, input_list, label_list, update=False, decode=False, summary=True)
        model.train_writer.add_summary(summary, global_step)

        # Validate on development set and write summary
        audio, label = next(dev_sample_generator)
        avg_loss, decoded, summary = model.step(sess, [audio], [label], update=False, decode=True, summary=True)
        model.dev_writer.add_summary(summary, global_step)
        perplexity = np.exp(float(avg_loss)) if avg_loss < 300 else float("inf")
        print("Validation average loss {:.2f} perplexity {:.2f}".format(avg_loss, perplexity))
        decoded_ids = next(extract_decoded_ids(decoded))
        decoded_str = vocabulary.ids_to_sentence(decoded_ids)
        expected_str = vocabulary.ids_to_sentence(label_list[0])
        print('Expected: {}'.format(expected_str))
        print('Decoded: {}'.format(decoded_str))


def main(_):
  train()


if __name__ == "__main__":
  tf.app.run()
