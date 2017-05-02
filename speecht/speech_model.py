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

import abc
import math

import tensorflow as tf
import speecht.vocabulary
from tensorflow.contrib.layers import xavier_initializer

from speecht.speech_input import BaseInputLoader


# noinspection PyAttributeOutsideInit
class SpeechModel:

  def __init__(self, input_loader: BaseInputLoader, input_size: int, num_classes: int):
    """
    Create a new speech model

    Args:
      input_loader: the object that provides input tensors
      input_size: the number of values per time step
      num_classes: the number of output classes (vocabulary_size + 1 for blank label)
    """
    self.input_loader = input_loader
    self.input_size = input_size
    self.convolution_count = 0

    self.global_step = tf.Variable(0, trainable=False)

    # inputs is of dimension [batch_size, max_time, input_size]
    self.inputs, self.sequence_lengths, self.labels = input_loader.get_inputs()

    self.logits = self._create_network(num_classes)

    # Generate summary image for logits [batch_size=batch_size, height=num_classes, width=max_time / 2, channels=1]
    tf.summary.image('logits', tf.expand_dims(tf.transpose(self.logits, (1, 2, 0)), 3))
    tf.summary.histogram('logits', self.logits)

  def add_training_ops(self, learning_rate: bool = 1e-3, learning_rate_decay_factor: float = 0,
                       max_gradient_norm: float = 5.0, momentum: float = 0.9):
    """
    Add the ops for training

    Args:
      learning_rate: the inital learning rate
      learning_rate_decay_factor: the factor to multiple the learning rate with when it should be decreased
      max_gradient_norm: the maximum gradient norm to apply, otherwise clipping is applied
      momentum: the momentum parameter
    """

    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32, name='learning_rate')
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

    # Variable summaries
    tf.summary.scalar('learning_rate', self.learning_rate)

    # Define loss and optimizer
    if self.labels is not None:
      with tf.name_scope('training'):
        self.cost = tf.nn.ctc_loss(self.labels, self.logits, self.sequence_lengths // 2)
        self.avg_loss = tf.reduce_mean(self.cost, name='average_loss')
        tf.summary.scalar('loss', self.avg_loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-3)
        gvs = optimizer.compute_gradients(self.avg_loss)
        gradients, trainables = zip(*gvs)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm, name='clip_gradients')
        self.update = optimizer.apply_gradients(zip(clipped_gradients, trainables),
                                                global_step=self.global_step, name='apply_gradients')

  def add_decoding_ops(self, language_model: str = None, lm_weight: float = 0.8, word_count_weight: float = 0.0,
                       valid_word_count_weight: float = 2.3):
    """
    Add the ops for decoding
j
    Args:
      language_model: the file path to the language model to use for beam search decoding or None
      word_count_weight: The weight added for each added word
      valid_word_count_weight: The weight added for each in vocabulary word
      lm_weight: The weight multiplied with the language model scoring
    """
    with tf.name_scope('decoding'):
      self.lm_weight = tf.placeholder_with_default(lm_weight, shape=(), name='language_model_weight')
      self.word_count_weight = tf.placeholder_with_default(word_count_weight, shape=(), name='word_count_weight')
      self.valid_word_count_weight = tf.placeholder_with_default(valid_word_count_weight, shape=(),
                                                                 name='valid_word_count_weight')

      if language_model:
        self.softmaxed = tf.log(tf.nn.softmax(self.logits, name='softmax') + 1e-8) / math.log(10)
        self.decoded, self.log_probabilities = tf.nn.ctc_beam_search_decoder(self.softmaxed,
                                                                             self.sequence_lengths // 2,
                                                                             kenlm_directory_path=language_model,
                                                                             kenlm_weight=self.lm_weight,
                                                                             word_count_weight=self.word_count_weight,
                                                                             valid_word_count_weight=self.valid_word_count_weight,
                                                                             beam_width=100,
                                                                             merge_repeated=False,
                                                                             top_paths=1)
      else:
        self.decoded, self.log_probabilities = tf.nn.ctc_greedy_decoder(self.logits,
                                                                        self.sequence_lengths // 2,
                                                                        merge_repeated=True)

  def finalize(self, log_dir: str, run_name: str, run_type: str):
    # Initializing the variables
    self.init = tf.global_variables_initializer()

    # Create saver
    self.saver = tf.train.Saver(tf.global_variables())

    # Create summary writers
    self.merged_summaries = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter('{}/{}_{}'.format(log_dir, run_name, run_type))

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

    with tf.variable_scope('convolution_layer_{}'.format(layer_id)) as layer:
      # Create variables filter and bias
      filters = tf.get_variable('filters', shape=[filter_width, input_channels, out_channels],
                                dtype=tf.float32, initializer=xavier_initializer())
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
        tf.summary.image(layer.name + 'filters', kernel_transposed, max_outputs=3)
        tf.summary.histogram(layer.name + 'filters', filters)

        tf.summary.image(layer.name + 'bias', tf.reshape(bias, [1, 1, out_channels, 1]))
        tf.summary.histogram(layer.name + 'bias', bias)

      # Add bias
      convolution_out = tf.nn.bias_add(convolution_out, bias)

      if apply_non_linearity:
        # Add non-linearity
        activations = tf.nn.relu(convolution_out, name='activation')
        tf.summary.histogram(layer.name + 'activation', activations)
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

  def step(self, sess, loss=True, update=True, decode=False, return_label=False, summary=False, feed_dict=None):
    """
    Evaluate the graph, you may update weights, decode audio or generate a summary

    Args:
      sess: tensorflow session
      update: should the network be trained
      loss: should output the avg_loss
      decode: should the decoding be performed and returned
      return_label: should the label be returned
      summary: should the summary be generated
      feed_dict: additional tensors that should be fed

    Returns: avg_loss (optional), decoded (optional), label (optional), update (optional), summary (optional)

    """

    output_feed = []

    if loss:
      output_feed.append(self.avg_loss)

    if decode:
      output_feed.append(self.decoded)

    if return_label:
      output_feed.append(self.labels)

    if update:
      output_feed.append(self.update)

    if summary:
      output_feed.append(self.merged_summaries)

    input_feed_dict = self.input_loader.get_feed_dict() or {}
    if feed_dict is not None:
      input_feed_dict.update(feed_dict)

    return sess.run(output_feed, feed_dict=input_feed_dict)

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

  def restore(self, session, checkpoint_directory: str, reset_learning_rate: float = None):
    ckpt = tf.train.get_checkpoint_state(checkpoint_directory)
    if ckpt and ckpt.model_checkpoint_path:
      print('Reading model parameters from {}'.format(ckpt.model_checkpoint_path))
      self.saver.restore(session, ckpt.model_checkpoint_path)
      self.init_session(session, init_variables=False)
      if reset_learning_rate:
        session.run(self.learning_rate.assign(reset_learning_rate))
    else:
      raise FileNotFoundError('No checkpoint for evaluation found')

  def restore_or_create(self, session, checkpoint_directory: str, reset_learning_rate: float = None):
    try:
      self.restore(session, checkpoint_directory, reset_learning_rate)
    except FileNotFoundError:
      print('Created model with fresh parameters.')
      self.init_session(session, init_variables=True)


class Wav2LetterModel(SpeechModel):

  def __init__(self, input_loader: BaseInputLoader, input_size: int, num_classes: int):
    super().__init__(input_loader, input_size, num_classes)

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


def create_default_model(flags, input_size: int, speech_input: BaseInputLoader) -> SpeechModel:
  model = Wav2LetterModel(input_loader=speech_input,
                          input_size=input_size,
                          num_classes=speecht.vocabulary.SIZE + 1)

  # TODO how can we restore only selected variables so we do not need to always create the full network?
  if flags.command == 'train':
    model.add_training_ops(learning_rate=flags.learning_rate,
                           learning_rate_decay_factor=flags.learning_rate_decay_factor,
                           max_gradient_norm=flags.max_gradient_norm,
                           momentum=flags.momentum)
    model.add_decoding_ops()
  elif flags.command == 'export':
    model.add_training_ops()
    model.add_decoding_ops()
  else:
    model.add_training_ops()
    model.add_decoding_ops(language_model=flags.language_model,
                           lm_weight=flags.lm_weight,
                           word_count_weight=flags.word_count_weight,
                           valid_word_count_weight=flags.valid_word_count_weight)

  model.finalize(log_dir=flags.log_dir,
                 run_name=flags.run_name,
                 run_type=flags.run_type)

  return model
