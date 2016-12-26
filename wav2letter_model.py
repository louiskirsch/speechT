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


class Wav2LetterModel:

  def __init__(self, max_time, input_size, num_classes,
               hidden_size, learning_rate, max_gradient_norm, num_layers):
    self.max_time = max_time
    self.input_size = input_size

    # Define input placeholders
    self.inputs = tf.placeholder(tf.float32, [None, max_time, input_size], name='inputs')
    self.sequence_lengths = tf.placeholder(tf.int32, [None], name='sequence_lengths')
    self.labels = tf.sparse_placeholder(tf.int32, name='labels')

    # Create RNN cells
    def create_cell():
      cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
      if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
      return cell

    cell_fw = create_cell()
    cell_bw = create_cell()

    # Define bidirectional RNN
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.inputs,
                                                 self.sequence_lengths, dtype=tf.float32)
    outputs = tf.concat(2, outputs)  # of size [batch_size, max_time, 2 * hidden_size]

    # Linear projection after LSTM output
    def create_fully_connected_variables(scope_name, output_size):
      with tf.variable_scope(scope_name) as scope:
        tf.get_variable('weights', initializer=tf.random_normal([2 * hidden_size, output_size]))
        tf.get_variable('biases', initializer=tf.random_normal([output_size]))

    def fully_connected(scope_name, prev_output, use_relu):
      with tf.variable_scope(scope_name, reuse=True):
        weights = tf.get_variable('weights')
        biases = tf.get_variable('biases')
        out = tf.matmul(prev_output, weights) + biases
        if use_relu:
          return tf.nn.relu(out)
        return out

    create_fully_connected_variables('hidden_layer', 2 * hidden_size)
    create_fully_connected_variables('logits_layer', num_classes)

    # iterate outputs and project linearly of size [batch_size, 2 * hidden_size]
    logits_per_output_list = []
    for output in tf.unpack(tf.transpose(outputs, (1, 0, 2))):
      hidden_layer = fully_connected('hidden_layer', output, True)
      logits_per_output_list.append(fully_connected('logits_layer', hidden_layer, True))
    # repack logits to size [max_time, batch_size, 2 * hidden_size]
    logits = tf.pack(logits_per_output_list, 0)

    # Define loss and optimizer
    self.cost = tf.nn.ctc_loss(logits, self.labels, self.sequence_lengths)
    self.avg_loss = tf.reduce_mean(self.cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(self.cost)
    gradients, trainables = zip(*gvs)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                     max_gradient_norm)
    self.update = optimizer.apply_gradients(zip(clipped_gradients, trainables))

    # Decoding
    # TODO use beam search here later
    self.decoded = tf.nn.ctc_greedy_decoder(logits, self.sequence_lengths)

    # TODO evaluate model

    # Initializing the variables
    self.init = tf.initialize_all_variables()

  def init_session(self, sess):
    sess.run(self.init)

  def step(self, sess, input_list, label_list):
    """

    Args:
      sess: tensorflow session
      input_list: spectrogram inputs, list of Tensors [time, input_size]
      label_list: identifiers from vocabulary, list of list of int32

    Returns: update, avg_loss

    """
    if len(input_list) != len(label_list):
      raise ValueError('Input list must have same length as label list')

    input_tensor = np.zeros((len(input_list), self.max_time, self.input_size))
    sequence_lengths = np.array([inp.shape[0] for inp in input_list])

    # Fill input tensor
    for idx, inp in enumerate(input_list):
      input_tensor[idx, :inp.shape[0], :] = inp

    # Fill label tensor
    label_shape = np.array([len(label_list), self.max_time], dtype=np.int)
    label_indices = []
    label_values = []
    for labelIdx, label in enumerate(label_list):
      for idIdx, identifier in enumerate(label):
        label_indices.append([labelIdx, idIdx])
        label_values.append(identifier)
    label_indices = np.array(label_indices, dtype=np.int)
    label_values = np.array(label_values, dtype=np.int)

    input_feed = {
      self.inputs: input_tensor,
      self.sequence_lengths: sequence_lengths,
      self.labels: tf.SparseTensorValue(label_indices, label_values, label_shape)
    }

    output_feed = [
      self.update,
      self.avg_loss
    ]

    return sess.run(output_feed, feed_dict=input_feed)
