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

  def __init__(self, input_size, num_classes,
               learning_rate, learning_rate_decay_factor, max_gradient_norm):
    self.input_size = input_size

    # TODO give all variables / ops proper names

    # Define input placeholders
    self.inputs = tf.placeholder(tf.float32, [None, None, input_size], name='inputs')
    self.sequence_lengths = tf.placeholder(tf.int32, [None], name='sequence_lengths')
    self.labels = tf.sparse_placeholder(tf.int32, name='labels')

    # Define non-trainables
    self.global_step = tf.Variable(0, trainable=False)
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32, name='learning_rate')
    self.learning_rate_decay_op = self.learning_rate.assign(tf.mul(self.learning_rate, learning_rate_decay_factor))

    def convolution(value, filter_width, stride, input_channels, out_channels, apply_non_linearity=True):
      # TODO Is stddev and constant a good choice?
      initial_filter = tf.truncated_normal([filter_width, input_channels, out_channels], stddev=0.1)
      filters = tf.Variable(initial_filter)
      bias = tf.Variable(tf.constant(0.1, shape=[out_channels]))
      convolution_out = tf.nn.conv1d(value, filters, stride, 'SAME', use_cudnn_on_gpu=True)
      convolution_out += bias
      if apply_non_linearity:
        convolution_out = tf.nn.tanh(convolution_out)
      return convolution_out, out_channels

    # TODO scale up input size of 13 to 250 channels?
    # One striding layer of output size [batch_size, max_time / 2, input_size]
    outputs, channels = convolution(self.inputs, 48, 2, input_size, 250)

    # 7 layers without striding of output size [batch_size, max_time / 2, input_size]
    for layer_idx in range(7):
      outputs, channels = convolution(outputs, 7, 1, channels, channels)

    # 1 layer with high kernel width and output size [batch_size, max_time / 2, input_size * 8]
    outputs, channels = convolution(outputs, 32, 1, channels, channels * 8)

    # 1 fully connected layer of output size [batch_size, max_time / 2, input_size * 8]
    outputs, channels = convolution(outputs, 1, 1, channels, channels)

    # 1 fully connected layer of output size [batch_size, max_time / 2, num_classes]
    # We must not apply a non linearity in this last layer
    outputs, channels = convolution(outputs, 1, 1, channels, num_classes, False)

    # transpose logits to size [max_time / 2, batch_size, num_classes]
    self.logits = tf.transpose(outputs, (1, 0, 2))

    # Define loss and optimizer
    self.cost = tf.nn.ctc_loss(self.logits, self.labels, self.sequence_lengths // 2)
    self.avg_loss = tf.reduce_mean(self.cost)
    optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
    gvs = optimizer.compute_gradients(self.avg_loss)
    gradients, trainables = zip(*gvs)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                     max_gradient_norm)
    self.update = optimizer.apply_gradients(zip(clipped_gradients, trainables), global_step=self.global_step)

    # Decoding
    # TODO use beam search here later
    self.decoded, self.log_probabilities = tf.nn.ctc_greedy_decoder(self.logits, self.sequence_lengths // 2)

    # TODO evaluate model

    # Initializing the variables
    self.init = tf.initialize_all_variables()

    # Create saver
    self.saver = tf.train.Saver(tf.all_variables())

  def init_session(self, sess):
    sess.run(self.init)

  def _get_inputs_feed_item(self, input_list):
    sequence_lengths = np.array([inp.shape[0] for inp in input_list])
    max_time = sequence_lengths.max()
    input_tensor = np.zeros((len(input_list), max_time, self.input_size))

    # Fill input tensor
    for idx, inp in enumerate(input_list):
      input_tensor[idx, :inp.shape[0], :] = inp

    return input_tensor, sequence_lengths, max_time

  @staticmethod
  def _get_labels_feed_item(label_list, max_time):
    # Fill label tensor
    label_shape = np.array([len(label_list), max_time], dtype=np.int)
    label_indices = []
    label_values = []
    for labelIdx, label in enumerate(label_list):
      for idIdx, identifier in enumerate(label):
        label_indices.append([labelIdx, idIdx])
        label_values.append(identifier)
    label_indices = np.array(label_indices, dtype=np.int)
    label_values = np.array(label_values, dtype=np.int)
    return tf.SparseTensorValue(label_indices, label_values, label_shape)

  def step(self, sess, input_list, label_list, update=True, decode=False):
    """

    Args:
      sess: tensorflow session
      input_list: spectrogram inputs, list of Tensors [time, input_size]
      label_list: identifiers from vocabulary, list of list of int32
      update: should the network be trained
      decode: should the decoding be performed and returned

    Returns: avg_loss, decoded (optional), update (optional)

    """
    if label_list is not None and len(input_list) != len(label_list):
      raise ValueError('Input list must have same length as label list')

    input_tensor, sequence_lengths, max_time = self._get_inputs_feed_item(input_list)

    input_feed = {
      self.inputs: input_tensor,
      self.sequence_lengths: sequence_lengths,
    }
    output_feed = []

    if label_list is not None:
      labels = self._get_labels_feed_item(label_list, max_time)
      input_feed[self.labels] = labels
      output_feed.append(self.avg_loss)

    if decode:
      output_feed.append(self.decoded[0])

    if update:
      output_feed.append(self.update)

    return sess.run(output_feed, feed_dict=input_feed)
