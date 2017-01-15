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
from abc import abstractmethod

import tensorflow as tf
import numpy as np


class BaseInputLoader:

  def __init__(self, input_size):
    self.input_size = input_size

  def _get_inputs_feed_item(self, input_list):
    """
    Generate the tensor from `input_list` to feed into the network

    Args:
      input_list: a list of numpy arrays of shape [time, input_size]

    Returns: tuple (input_tensor, sequence_lengths, max_time)

    """
    sequence_lengths = np.array([inp.shape[0] for inp in input_list])
    max_time = sequence_lengths.max()
    input_tensor = np.zeros((len(input_list), max_time, self.input_size))

    # Fill input tensor
    for idx, inp in enumerate(input_list):
      input_tensor[idx, :inp.shape[0], :] = inp

    return input_tensor, sequence_lengths, max_time

  @staticmethod
  def _get_labels_feed_item(label_list, max_time):
    """
    Generate the tensor from 'label_list' to feed as labels into the network

    Args:
      label_list: a list of encoded labels (ints)
      max_time: the maximum time length of `label_list`

    Returns: the SparseTensorValue to feed into the network

    """

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

  @abstractmethod
  def get_inputs(self):
    raise NotImplementedError()


class InputBatchLoader(BaseInputLoader):
  """
  This class manages the the background threads needed to fill a queue full of data.
  """

  def __init__(self, input_size, batch_size, data_generator):

    super().__init__(input_size)
    self.batch_size = batch_size
    self.data_generator = data_generator

    with tf.device("/cpu:0"):
      # Define input and label placeholders
      # inputs is of dimension [batch_size, max_time, input_size]
      self.inputs = tf.placeholder(tf.float32, [batch_size, None, input_size], name='inputs')
      self.sequence_lengths = tf.placeholder(tf.int32, [batch_size], name='sequence_lengths')
      self.labels = tf.sparse_placeholder(tf.int32, name='labels')

      # Queue for inputs and labels
      self.queue = tf.FIFOQueue(shapes=[[batch_size, None, input_size], [batch_size], []],
                                dtypes=[tf.float32, tf.int32, tf.int32],
                                capacity=50)

      self.enqueue_op = self.queue.enqueue([self.inputs,
                                            self.sequence_lengths,
                                            self.labels])

  def get_inputs(self):
    """
    Return's tensors for inputs, sequence_lengths and labels
    """
    with tf.device("/cpu:0"):
      inputs, sequence_lengths, labels = self.queue.dequeue()
    return inputs, sequence_lengths, labels

  def _batch(self, iterable):
    """
    Batch the values together
    Args:
      iterable: the data source

    Returns: list of  size `batch_size` with elements from `iterable`

    """
    args = [iter(iterable)] * self.batch_size
    return zip(*args)

  def _enqueue(self, sess, coord):
    """
    Function run on alternate thread. Basically, keep adding data to the queue.
    """
    for sample_batch in self._batch(self.data_generator):
      input_list, label_list = zip(*sample_batch)

      input_tensor, sequence_lengths, max_time = self._get_inputs_feed_item(input_list)
      labels = self._get_labels_feed_item(label_list, max_time)

      sess.run(self.enqueue_op, feed_dict={
        self.inputs: input_tensor,
        self.sequence_lengths: sequence_lengths,
        self.labels: labels
      })

      if coord.should_stop():
        break

  def start_threads(self, sess, coord, n_threads=1):
    """ Start background threads to feed queue """
    threads = []
    for n in range(n_threads):
      t = tf.threading.Thread(target=self._enqueue, args=(sess, coord))
      t.daemon = True  # thread will close when parent quits
      t.start()
      coord.register_thread(t)
      threads.append(t)
    return threads
