# Copyright 2017 Louis Kirsch. All Rights Reserved.
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

from abc import abstractmethod, ABCMeta
from functools import partial

import tensorflow as tf
from speecht.speech_input import InputBatchLoader
from speecht.speech_model import create_default_model

from speecht.preprocessing import SpeechCorpusReader


class DatasetExecutor(metaclass=ABCMeta):

  def __init__(self, flags):
    self.flags = flags
    self.reader = SpeechCorpusReader(self.flags.data_dir)

    print('Determine input size from first sample')
    self.input_size = self.determine_input_size()

    print('Initialize InputBatchLoader')
    self.speech_input = InputBatchLoader(self.input_size, self.flags.batch_size,
                                         partial(self.create_sample_generator, self.get_loader_limit_count()),
                                         self.get_max_steps())

  def determine_input_size(self):
    return next(self.create_sample_generator(limit_count=1))[0].shape[1]

  def get_max_steps(self):
    return None

  @abstractmethod
  def get_loader_limit_count(self) -> int:
    raise NotImplementedError('Loader limit count needs to be implemented')

  @abstractmethod
  def create_sample_generator(self, limit_count: int):
    raise NotImplementedError('Sample generator creation needs to be implemented')

  def start_pipeline(self, sess, n_threads=1):
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)
    self.speech_input.start_threads(sess=sess, coord=coord, n_threads=n_threads)
    return coord

  def create_model(self, sess):
    model = create_default_model(self.flags, self.input_size, self.speech_input)
    model.restore(sess, self.flags.run_train_dir)
    return model
