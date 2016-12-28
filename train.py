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
from wav2letter_model import Wav2LetterModel
from preprocess import SpeechCorpusReader

tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "data/", "Data directory")

FLAGS = tf.app.flags.FLAGS

# TODO what's the right value? set it as parameter?
N_COEFFICIENTS = 13


def train():
  model = Wav2LetterModel(N_COEFFICIENTS, vocabulary.SIZE + 1,
                          FLAGS.learning_rate, FLAGS.max_gradient_norm)

  with tf.Session() as sess:
    model.init_session(sess)

    reader = SpeechCorpusReader(FLAGS.data_dir)

    def batch(iterable, batch_size):
      args = [iter(iterable)] * batch_size
      return zip(*args)

    sample_generator = reader.load_samples('train', loop_infinitely=True)

    for sample_batch in batch(sample_generator, FLAGS.batch_size):
      input_list, label_list = zip(*sample_batch)

      _, avg_loss, cost = model.step(sess, input_list, label_list)

      non_inf_count = np.count_nonzero(~np.isinf(cost))
      print('Number of not inf loss {}'.format(non_inf_count))
      perplexity = np.exp(float(avg_loss)) if avg_loss < 300 else float("inf")
      print('Average loss: {:.2f}; Perplexity: {:.2f}'.format(avg_loss, perplexity))


def main(_):
  train()


if __name__ == "__main__":
  tf.app.run()
