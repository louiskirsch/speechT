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

import eval_utils
import preprocess
import vocabulary

from record_utils import AudioRecorder
from speech_input import SingleInputLoader
from speech_model import Wav2LetterModel

tf.app.flags.DEFINE_bool('relu', True, 'Use ReLU activation instead of tanh')
tf.app.flags.DEFINE_bool('power', False, 'Use a power spectrogram instead of mfccs as input')
tf.app.flags.DEFINE_bool('input_size', 39, 'The input size of each sample, depending on what preprocessing was used')
tf.app.flags.DEFINE_string('language_model', None, 'Use beam search with given language model. '
                                                   'Must be binary format with probing hash table.')
tf.app.flags.DEFINE_string("train_dir", "train/", "Training directory")
tf.app.flags.DEFINE_string("run_name", "", "The run name to append to the training directory")

FLAGS = tf.app.flags.FLAGS


def create_model(session, speech_input_loader):
  """Create speechT model and initialize or load parameters in session."""
  model = Wav2LetterModel(speech_input_loader,
                          FLAGS.input_size,
                          vocabulary.SIZE + 1,
                          learning_rate=0,
                          learning_rate_decay_factor=0,
                          max_gradient_norm=0,
                          log_dir='log/',
                          use_relu=FLAGS.relu,
                          run_name=FLAGS.run_name,
                          momentum=0,
                          run_type='record',
                          language_model=FLAGS.language_model)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
    model.init_session(session, init_variables=False)
  else:
    raise FileNotFoundError('No checkpoint for recording found')
  return model


def record_and_decode():
  # Use training sub-directory if not specified otherwise
  if FLAGS.run_name and FLAGS.train_dir == 'train/':
    FLAGS.train_dir += FLAGS.run_name + '/'

  print('Initialize SingleInputLoader')
  speech_input_loader = SingleInputLoader(FLAGS.input_size)

  sample_rate = 16000
  recorder = AudioRecorder(rate=sample_rate)

  with tf.Session() as sess:

    model = create_model(sess, speech_input_loader)

    while True:
      print('Recording audio')
      raw_audio, sample_width = recorder.record()
      raw_audio = np.array(raw_audio)

      print('Generate MFCCs or power spectrogram')
      if FLAGS.power:
        speech_input = preprocess.calc_power_spectrogram(raw_audio, sample_rate)
      else:
        speech_input = preprocess.calc_mfccs(raw_audio, sample_rate)

      speech_input_loader.set_input(speech_input)

      print('Running speech recognition')
      [decoded] = model.step(sess, loss=False, update=False, decode=True)

      # Print decoded string
      decoded_ids_paths = [eval_utils.extract_decoded_ids(path) for path in decoded]
      for decoded_path in decoded_ids_paths:
        decoded_ids = next(decoded_path)
        decoded_str = vocabulary.ids_to_sentence(decoded_ids)
        print('decoded: {}'.format(decoded_str))


def main(_):
  record_and_decode()


if __name__ == "__main__":
  tf.app.run()
