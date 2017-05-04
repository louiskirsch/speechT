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
from typing import Dict

import editdistance
import numpy as np
import tensorflow as tf
import speecht.vocabulary
from speecht.execution import DatasetExecutor

from speecht.speech_model import SpeechModel
import itertools


class EvalStatistics:

  def __init__(self):
    self.decodings_counter = 0
    self.sum_letter_edit_distance = 0
    self.sum_letter_error_rate = 0
    self.sum_word_edit_distance = 0
    self.sum_word_error_rate = 0
    self.letter_edit_distance = 0
    self.letter_error_rate = 0
    self.word_edit_distance = 0
    self.word_error_rate = 0

  def track_decoding(self, decoded_str, expected_str):
    self.letter_edit_distance = editdistance.eval(expected_str, decoded_str)
    self.letter_error_rate = self.letter_edit_distance / len(expected_str)
    self.word_edit_distance = editdistance.eval(expected_str.split(), decoded_str.split())
    self.word_error_rate = self.word_edit_distance / len(expected_str.split())
    self.sum_letter_edit_distance += self.letter_edit_distance
    self.sum_letter_error_rate += self.letter_error_rate
    self.sum_word_edit_distance += self.word_edit_distance
    self.sum_word_error_rate += self.word_error_rate
    self.decodings_counter += 1

  @property
  def global_letter_edit_distance(self):
    return self.sum_letter_edit_distance / self.decodings_counter

  @property
  def global_letter_error_rate(self):
    return self.sum_letter_error_rate / self.decodings_counter

  @property
  def global_word_edit_distance(self):
    return self.sum_word_edit_distance / self.decodings_counter

  @property
  def global_word_error_rate(self):
    return self.sum_word_error_rate / self.decodings_counter


class Evaluation(DatasetExecutor):

  def create_sample_generator(self, limit_count: int):
    return self.reader.load_samples(self.flags.dataset,
                                    loop_infinitely=False,
                                    limit_count=limit_count,
                                    feature_type=self.flags.feature_type)

  def get_loader_limit_count(self):
    return self.flags.step_count * self.flags.batch_size

  def get_max_steps(self):
    if self.flags.step_count:
      return self.flags.step_count
    return None

  def run(self):

    stats = EvalStatistics()

    with tf.Session() as sess:

      model = self.create_model(sess)

      print('Starting input pipeline')
      coord = self.start_pipeline(sess)

      try:
        print('Begin evaluation')

        if self.flags.step_count:
          step_iter = range(self.flags.step_count)
        else:
          step_iter = itertools.count()
        for step in step_iter:
          if coord.should_stop():
            break

          should_save = self.flags.should_save and step == 0
          self.run_step(model, sess, stats, should_save)

      except tf.errors.OutOfRangeError:
        print('Done evaluating -- step limit reached')
      finally:
        coord.request_stop()

      self.print_global_statistics(stats)

      coord.join()

  @staticmethod
  def print_global_statistics(stats):
    print('Global statistics')
    print('LED: {} LER: {:.2f} WED: {} WER: {:.2f}'.format(stats.global_letter_edit_distance,
                                                           stats.global_letter_error_rate,
                                                           stats.global_word_edit_distance,
                                                           stats.global_word_error_rate))

  def run_step(self, model: SpeechModel, sess: tf.Session, stats: EvalStatistics,
               save: bool, verbose=True, feed_dict: Dict=None):
    global_step = model.global_step.eval()

    # Validate on data set and write summary
    if save:
      avg_loss, decoded, label, summary = model.step(sess, update=False, decode=True, return_label=True,
                                                     summary=True, feed_dict=feed_dict)
      model.summary_writer.add_summary(summary, global_step)
    else:
      avg_loss, decoded, label = model.step(sess, update=False, decode=True,
                                            return_label=True, feed_dict=feed_dict)

    if verbose:
      perplexity = np.exp(float(avg_loss)) if avg_loss < 300 else float("inf")
      print("validation average loss {:.2f} perplexity {:.2f}".format(avg_loss, perplexity))

    # Print decode
    decoded_ids_paths = [Evaluation.extract_decoded_ids(path) for path in decoded]
    for label_ids in Evaluation.extract_decoded_ids(label):
      expected_str = speecht.vocabulary.ids_to_sentence(label_ids)
      if verbose:
        print('expected: {}'.format(expected_str))
      for decoded_path in decoded_ids_paths:
        decoded_ids = next(decoded_path)
        decoded_str = speecht.vocabulary.ids_to_sentence(decoded_ids)
        stats.track_decoding(decoded_str, expected_str)
        if verbose:
          print('decoded: {}'.format(decoded_str))
          print('LED: {} LER: {:.2f} WED: {} WER: {:.2f}'.format(stats.letter_edit_distance,
                                                                 stats.letter_error_rate,
                                                                 stats.word_edit_distance,
                                                                 stats.word_error_rate))

  @staticmethod
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