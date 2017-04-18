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

import editdistance
import numpy as np
import tensorflow as tf

import vocabulary
from preprocess import SpeechCorpusReader
from speech_input import InputBatchLoader
from speech_model import create_default_model


class Evaluation:

  @staticmethod
  def run(flags):
    reader = SpeechCorpusReader(flags.data_dir)

    def create_sample_generator(limit_count=flags.epoch_count * flags.batch_size):
      return reader.load_samples(flags.dataset,
                                 loop_infinitely=False,
                                 limit_count=limit_count,
                                 feature_type=flags.feature_type)

    print('Determine input size from first sample')
    input_size = next(create_sample_generator(limit_count=1))[0].shape[1]

    print('Initialize InputBatchLoader')
    speech_input = InputBatchLoader(input_size, flags.batch_size, create_sample_generator, flags.epoch_count)

    with tf.Session() as sess:

      model = create_default_model(flags, input_size, speech_input)
      model.restore(sess, flags.run_train_dir)

      coord = tf.train.Coordinator()
      print('Starting input pipeline')
      tf.train.start_queue_runners(sess=sess, coord=coord)
      speech_input.start_threads(sess=sess, coord=coord)

      try:
        print('Begin evaluation')

        decodings_counter = 0
        global_letter_edit_distance = 0
        global_letter_error_rate = 0
        global_word_edit_distance = 0
        global_word_error_rate = 0

        for epoch in range(flags.epoch_count):
          if coord.should_stop():
            break

          global_step = model.global_step.eval()

          # Validate on development set and write summary
          if not flags.should_save or epoch > 0:
            avg_loss, decoded, label = model.step(sess, update=False, decode=True, return_label=True)
          else:
            avg_loss, decoded, label, summary = model.step(sess, update=False, decode=True,
                                                           return_label=True, summary=True)
            model.summary_writer.add_summary(summary, global_step)

          perplexity = np.exp(float(avg_loss)) if avg_loss < 300 else float("inf")
          print("validation average loss {:.2f} perplexity {:.2f}".format(avg_loss, perplexity))

          # Print decode
          decoded_ids_paths = [Evaluation.extract_decoded_ids(path) for path in decoded]
          for label_ids in Evaluation.extract_decoded_ids(label):
            expected_str = vocabulary.ids_to_sentence(label_ids)
            print('expected: {}'.format(expected_str))
            for decoded_path in decoded_ids_paths:
              decoded_ids = next(decoded_path)
              decoded_str = vocabulary.ids_to_sentence(decoded_ids)
              print('decoded: {}'.format(decoded_str))

              letter_edit_distance = editdistance.eval(expected_str, decoded_str)
              letter_error_rate = letter_edit_distance / len(expected_str)
              word_edit_distance = editdistance.eval(expected_str.split(), decoded_str.split())
              word_error_rate = word_edit_distance / len(expected_str.split())
              global_letter_edit_distance += letter_edit_distance
              global_letter_error_rate += letter_error_rate
              global_word_edit_distance += word_edit_distance
              global_word_error_rate += word_error_rate
              decodings_counter += 1

              print('LED: {} LER: {:.2f} WED: {} WER: {:.2f}'.format(letter_edit_distance,
                                                                     letter_error_rate,
                                                                     word_edit_distance,
                                                                     word_error_rate))

        print('Global statistics')
        global_letter_edit_distance /= decodings_counter
        global_letter_error_rate /= decodings_counter
        global_word_edit_distance /= decodings_counter
        global_word_error_rate /= decodings_counter
        print('LED: {} LER: {:.2f} WED: {} WER: {:.2f}'.format(global_letter_edit_distance,
                                                               global_letter_error_rate,
                                                               global_word_edit_distance,
                                                               global_word_error_rate))
      except tf.errors.OutOfRangeError:
        print('Done evaluating -- epoch limit reached')
      finally:
        # When done, ask the threads to stop.
        coord.request_stop()

      coord.join()
      sess.close()

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