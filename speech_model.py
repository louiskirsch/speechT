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
import abc


class SpeechModel:
  def __init__(self, input_loader, input_size, num_classes, learning_rate, learning_rate_decay_factor,
               max_gradient_norm, log_dir, use_relu, run_name, momentum, run_type):
    """
    Create a new speech model

    Args:
      input_loader: the object that provides input tensors
      input_size: the number of values per time step
      num_classes: the number of output classes (vocabulary_size + 1 for blank label)
      learning_rate: the inital learning rate
      learning_rate_decay_factor: the factor to multiple the learning rate with when it should be decreased
      max_gradient_norm: the maximum gradient norm to apply, otherwise clipping is applied
      log_dir: the directory to log to for use of tensorboard
      use_relu: if True, use relu instead of tanh
      run_name: the name of this run
      momentum: the momentum parameter
      run_type: "train", "dev" or "test"
    """
    self.input_size = input_size
    self.convolution_count = 0

    self.activation_fnc = tf.nn.relu if use_relu else tf.nn.tanh

    # inputs is of dimension [batch_size, max_time, input_size]
    self.inputs, self.sequence_lengths, self.labels = input_loader.get_inputs()

    # Define non-trainables
    self.global_step = tf.Variable(0, trainable=False)
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32, name='learning_rate')
    self.learning_rate_decay_op = self.learning_rate.assign(learning_rate_decay_factor * self.learning_rate)

    # Variable summaries
    tf.summary.scalar('learning_rate', self.learning_rate)

    self.logits = self._create_network(num_classes)

    # Generate summary image for logits [batch_size=batch_size, height=num_classes, width=max_time / 2, channels=1]
    tf.summary.image('logits', tf.expand_dims(tf.transpose(self.logits, (1, 2, 0)), 3))
    tf.summary.histogram('logits', self.logits)

    # Define loss and optimizer
    with tf.name_scope('training'):
      self.cost = tf.nn.ctc_loss(self.logits, self.labels, self.sequence_lengths // 2)
      self.avg_loss = tf.reduce_mean(self.cost, name='average_loss')
      tf.summary.scalar('loss', self.avg_loss)
      optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum, name='optimizer')
      gvs = optimizer.compute_gradients(self.avg_loss)
      gradients, trainables = zip(*gvs)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm, name='clip_gradients')
      self.update = optimizer.apply_gradients(zip(clipped_gradients, trainables),
                                              global_step=self.global_step, name='apply_gradients')

    # Decoding
    with tf.name_scope('decoding'):
      self.decoded, self.log_probabilities = tf.nn.ctc_beam_search_decoder(self.logits,
                                                                           self.sequence_lengths // 2,
                                                                           kenlm_file_path='gigaword.binary',
                                                                           beam_width=1000,
                                                                           top_paths=1)

    # Initializing the variables
    self.init = tf.global_variables_initializer()

    # Create saver
    self.saver = tf.train.Saver(tf.global_variables())

    # Create summary writers
    self.merged_summaries = tf.summary.merge_all()
    if run_name:
      run_name += '_'
    self.summary_writer = tf.summary.FileWriter('{}/{}{}'.format(log_dir, run_name, run_type))

  def _convolution(self, value, filter_width, stride, input_channels, out_channels, apply_non_linearity=True):
    """
    Apply a convolutional layer

    Args:
      value: the input tensor to apply the convolution on
      filter_width: the width of the filter (kernel)
      stride: the striding of the filter (kernel)
      input_channels: the number if input channels
      out_channels: the number of output channels
      apply_non_linearity: whether to apply a non linearity

    Returns:
      the output after convolution, added biases and possible non linearity applied

    """

    layer_id = self.convolution_count
    self.convolution_count += 1

    with tf.name_scope('convolution_layer_{}'.format(layer_id)) as layer:
      # Filter and bias
      initial_filter = tf.truncated_normal([filter_width, input_channels, out_channels], stddev=0.01)
      filters = tf.Variable(initial_filter, name='filters')
      bias = tf.Variable(tf.constant(0.0, shape=[out_channels]), name='bias')

      # Apply convolution
      convolution_out = tf.nn.conv1d(value, filters, stride, 'SAME', use_cudnn_on_gpu=True, name='convolution')

      # Create summary
      with tf.name_scope('summaries'):
        # add depth of 1 (=grayscale) leading to shape [filter_width, input_channels, 1, out_channels]
        kernel_with_depth = tf.expand_dims(filters, 2)

        # to tf.image_summary format [batch_size=out_channels, height=filter_width, width=input_channels, channels=1]
        kernel_transposed = tf.transpose(kernel_with_depth, [3, 0, 1, 2])

        # this will display random 3 filters from all the output channels
        tf.summary.image(layer + 'filters', kernel_transposed, max_outputs=3)
        tf.summary.histogram(layer + 'filters', filters)

        tf.summary.image(layer + 'bias', tf.reshape(bias, [1, 1, out_channels, 1]))
        tf.summary.histogram(layer + 'bias', bias)

      # Add bias
      convolution_out = tf.nn.bias_add(convolution_out, bias)

      if apply_non_linearity:
        # Add non-linearity
        activations = self.activation_fnc(convolution_out, name='activation')
        tf.summary.histogram(layer + 'activation', activations)
        return activations, out_channels
      else:
        return convolution_out, out_channels

  def init_session(self, sess, init_variables=True):
    """
    Initialize a new session for the model.

    Args:
      sess: session to initalize
      init_variables: whether to initialize all variables

    """
    if init_variables:
      sess.run(self.init)

    self.summary_writer.add_graph(sess.graph)

  def step(self, sess, update=True, decode=False, return_label=False, summary=False):
    """
    Evaluate the graph, you may update weights, decode audio or generate a summary

    Args:
      sess: tensorflow session
      update: should the network be trained
      decode: should the decoding be performed and returned
      return_label: should the label be returned
      summary: should the summary be generated

    Returns: avg_loss, decoded (optional), label (optional), update (optional), summary (optional)

    """

    output_feed = [
      self.avg_loss
    ]

    if decode:
      output_feed.append(self.decoded)

    if return_label:
      output_feed.append(self.labels)

    if update:
      output_feed.append(self.update)

    if summary:
      output_feed.append(self.merged_summaries)

    return sess.run(output_feed)

  @abc.abstractclassmethod
  def _create_network(self, num_classes):
    """
    Should create the network producing the logits to then be used by the loss function

    Args:
      num_classes: specifies the number of classes to generate as logits

    Returns:
      the logits in the form [time, batch_size, num_classes]

    """
    raise NotImplementedError()


class Wav2LetterModel(SpeechModel):

  def __init__(self, input_loader, input_size, num_classes, learning_rate, learning_rate_decay_factor,
               max_gradient_norm, log_dir, use_relu, run_name, momentum, run_type):
    """
    Create a new Wav2Letter model

    Args:
      input_loader: the object that provides input tensors
      input_size: the number of values per time step
      num_classes: the number of output classes (vocabulary_size + 1 for blank label)
      learning_rate: the inital learning rate
      learning_rate_decay_factor: the factor to multiple the learning rate with when it should be decreased
      max_gradient_norm: the maximum gradient norm to apply, otherwise clipping is applied
      log_dir: the directory to log to for use of tensorboard
      use_relu: if True, use relu instead of tanh
      run_name: the name of this run
      momentum: the momentum parameter
      run_type: "train", "dev" or "test"

    """
    super().__init__(input_loader, input_size, num_classes, learning_rate, learning_rate_decay_factor,
                     max_gradient_norm, log_dir, use_relu, run_name, momentum, run_type)

  def _create_network(self, num_classes):
    # The first layer scales up from input_size channels to 250 channels
    # One striding layer of output size [batch_size, max_time / 2, 250]
    outputs, channels = self._convolution(self.inputs, 48, 2, self.input_size, 250)

    # 7 layers without striding of output size [batch_size, max_time / 2, 250]
    for layer_idx in range(7):
      outputs, channels = self._convolution(outputs, 7, 1, channels, channels)

    # 1 layer with high kernel width and output size [batch_size, max_time / 2, 2000]
    outputs, channels = self._convolution(outputs, 32, 1, channels, channels * 8)

    # 1 fully connected layer of output size [batch_size, max_time / 2, 2000]
    outputs, channels = self._convolution(outputs, 1, 1, channels, channels)

    # 1 fully connected layer of output size [batch_size, max_time / 2, num_classes]
    # We must not apply a non linearity in this last layer
    outputs, channels = self._convolution(outputs, 1, 1, channels, num_classes, False)

    # transpose logits to size [max_time / 2, batch_size, num_classes]
    return tf.transpose(outputs, (1, 0, 2))
