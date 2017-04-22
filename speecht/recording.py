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

import numpy as np
import tensorflow as tf
from speecht.evaluation import Evaluation
from speecht.speech_input import SingleInputLoader
from speecht.speech_model import create_default_model
import speecht.vocabulary

from speecht import preprocessing


class Recording:

  def __init__(self, flags):
    self.flags = flags

  def run(self):
    # Only import here to not always require 'pyaudio'
    from speecht.record_utils import AudioRecorder

    print('Initialize SingleInputLoader')
    speech_input_loader = SingleInputLoader(self.flags.input_size)

    sample_rate = 16000
    recorder = AudioRecorder(rate=sample_rate, chunk_size=4 * 1024)

    with tf.Session() as sess:

      model = create_default_model(self.flags, self.flags.input_size, speech_input_loader)
      model.restore(sess, self.flags.run_train_dir)

      while True:
        print('Recording audio')
        raw_audio, sample_width = recorder.record()
        raw_audio = np.array(raw_audio)

        print('Generate MFCCs or power spectrogram')
        if self.flags.feature_type == 'power':
          speech_input = preprocessing.calc_power_spectrogram(raw_audio, sample_rate)
        elif self.flags.feature_type == 'mfcc':
          speech_input = preprocessing.calc_mfccs(raw_audio, sample_rate)
        else:
          raise NotImplementedError('Only power and mfccs are supported for input types.')

        speech_input_loader.set_input(speech_input)

        print('Running speech recognition')
        [decoded] = model.step(sess, loss=False, update=False, decode=True)

        # Print decoded string
        decoded_ids_paths = [Evaluation.extract_decoded_ids(path) for path in decoded]
        for decoded_path in decoded_ids_paths:
          decoded_ids = next(decoded_path)
          decoded_str = speecht.vocabulary.ids_to_sentence(decoded_ids)
          print('decoded: {}'.format(decoded_str))

