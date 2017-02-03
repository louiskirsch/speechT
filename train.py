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
from speech_input import InputBatchLoader, BaseInputLoader
from speech_model import Wav2LetterModel, SpeechModel
from preprocess import SpeechCorpusReader

tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_bool("reset_learning_rate", False, "Reset the learning rate to the default or provided value")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0,
                          "Enable learning rate decay, decays by the given factor.")
tf.app.flags.DEFINE_float("momentum", 0.9, "Optimizer momentum")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_bool('relu', True, 'Use ReLU activation instead of tanh')
tf.app.flags.DEFINE_bool('power', False, 'Use a power spectrogram instead of mfccs as input')
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "data/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "train/", "Training directory")
tf.app.flags.DEFINE_string("log_dir", "log/", "Logging directory for summaries")
tf.app.flags.DEFINE_string("run_name", "", "Give this training a name to appear in tensorboard")
tf.app.flags.DEFINE_integer("limit_training_set", 0,
                            "Train on a smaller training set, limited to the specified size")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000,
                            "How many training steps to do per checkpoint.")

FLAGS = tf.app.flags.FLAGS


def create_model(session: tf.Session, input_size: int, speech_input: BaseInputLoader) -> SpeechModel:
  """Create speechT model and initialize or load parameters in session."""
  model = Wav2LetterModel(speech_input,
                          input_size,
                          vocabulary.SIZE + 1,
                          FLAGS.learning_rate,
                          FLAGS.learning_rate_decay_factor,
                          FLAGS.max_gradient_norm,
                          FLAGS.log_dir,
                          FLAGS.relu,
                          FLAGS.run_name,
                          FLAGS.momentum,
                          run_type='train')
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
    model.init_session(session, init_variables=False)
    if FLAGS.reset_learning_rate:
      session.run(model.learning_rate.assign(FLAGS.learning_rate))
  else:
    print("Created model with fresh parameters.")
    model.init_session(session, init_variables=True)
  return model


def train():
  # Create training sub-directory if not specified otherwise
  if FLAGS.run_name and FLAGS.train_dir == 'train/':
    FLAGS.train_dir += FLAGS.run_name + '/'

  # Determine feature type
  feature_type = 'power' if FLAGS.power else 'mfcc'

  # Create training directory if it does not exist
  if not os.path.exists(FLAGS.train_dir):
    os.makedirs(FLAGS.train_dir)

  reader = SpeechCorpusReader(FLAGS.data_dir)

  def create_sample_generator(limit_count=FLAGS.limit_training_set):
    return reader.load_samples('train',
                               loop_infinitely=True,
                               limit_count=limit_count,
                               feature_type=feature_type)

  print('Determine input size from first sample')
  input_size = next(create_sample_generator(limit_count=1))[0].shape[1]

  print('Initialize InputBatchLoader')
  speech_input = InputBatchLoader(input_size, FLAGS.batch_size, create_sample_generator)

  with tf.Session() as sess:

    model = create_model(sess, input_size, speech_input)

    coord = tf.train.Coordinator()
    print('Starting input pipeline')
    tf.train.start_queue_runners(sess=sess, coord=coord)
    speech_input.start_threads(sess=sess, coord=coord, n_threads=2)

    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []

    try:
      print('Begin training')
      while not coord.should_stop():

        current_step += 1
        is_checkpoint_step = current_step % FLAGS.steps_per_checkpoint == 0

        start_time = time.time()
        step_result = model.step(sess, summary=is_checkpoint_step)
        avg_loss = step_result[0]
        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += avg_loss / FLAGS.steps_per_checkpoint

        # Once in a while, we save checkpoint and print statistics
        if is_checkpoint_step:
          global_step = model.global_step.eval()

          # Print statistics for the previous epoch.
          perplexity = np.exp(float(avg_loss)) if avg_loss < 300 else float("inf")
          print("global step {:d} learning rate {:.4f} step-time {:.2f} average loss {:.2f} perplexity {:.2f}"
                .format(global_step, model.learning_rate.eval(), step_time, avg_loss, perplexity))

          # Retrieve and store summary
          summary = step_result[2]
          model.summary_writer.add_summary(summary, global_step)

          # Decrease learning rate if no improvement was seen over last 3 times.
          if FLAGS.learning_rate_decay_factor > 0 and len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
            sess.run(model.learning_rate_decay_op)
          previous_losses.append(loss)

          # Save checkpoint and zero timer and loss.
          checkpoint_path = os.path.join(FLAGS.train_dir, "speechT.ckpt")
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)
          print('Model saved')
          step_time, loss = 0.0, 0.0

    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    coord.join()
    sess.close()


def main(_):
  train()


if __name__ == "__main__":
  tf.app.run()
