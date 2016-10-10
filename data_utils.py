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
import fnmatch
import glob
import os
import soundfile as sf
import random
import numpy as np
import warnings

# TODO limit to max vocabulary size
class Vocabulary:
  """
  Maintains a vocabulary, i.e. a mapping from words to ids and vice-versa.
  """

  PAD_ID = 0
  GO_ID = 1
  EOS_ID = 2
  UNK_ID = 3

  def __init__(self):
    self._word_to_id = dict()
    self._id_to_word = dict()
    self._counter = 4

  def create(self, word):
    new_id = self._counter
    self._counter += 1
    self._word_to_id[word] = new_id
    self._id_to_word[new_id] = word
    return new_id

  def retrieve_by_id(self, id):
    return self._id_to_word[id]

  def retrieve_by_word(self, word):
    return self._word_to_id[word]

  def retrieve_or_create(self, word):
    if word not in self._word_to_id:
      return self.create(word)
    return self.retrieve_by_word(word)

  def retrieve_from_ids(self, ids):
    return [self.retrieve_by_id(i) for i in ids]

  def size(self):
    return len(self._word_to_id)


def fragment_audio(audio_data, samplerate, fragment_length):
  # Pad the audio data to allow for splitting into multiple fragments
  fragment_size = (int)(samplerate * fragment_length)
  pad_size = fragment_size - (audio_data.shape[0] % fragment_size)
  audio_data = np.lib.pad(audio_data, (0, pad_size), 'constant')
  # Split into multiple fragments
  audio_fragments = np.reshape(audio_data, (-1, fragment_size))
  return audio_fragments


def iglob_recursive(directory, file_pattern):
  for root, dir_names, file_names in os.walk(directory):
    for filename in fnmatch.filter(file_names, file_pattern):
      yield os.path.join(root, filename)


class SpeechCorpusReader:
  """
  Reads and transforms the speech corpus to be used by the NN
  """
  def __init__(self, data_directory, vocabulary):
    self._data_directory = data_directory
    self._vocabulary = vocabulary
    self._transcript_dict = self._build_transcript()

  def _build_transcript(self):
    """
    Builds a transcript from transcript files, mapping from audio-id to a list of word-ids
    :return: the created transcript
    """
    transcript_dict = dict()
    transcript_files = iglob_recursive(self._data_directory, '*.trans.txt')
    for transcript_file in transcript_files:
      with open(transcript_file, 'r') as f:
        for line in f:
          # Strip included new line symbol
          line = line.rstrip('\n')

          # Each line is in the form
          # 00-000000-0000 WORD1 WORD2 ...
          splitted = line.split(' ')
          transcript_dict[splitted[0]] = \
            [self._vocabulary.retrieve_or_create(word) for word in splitted[1:]]

    return transcript_dict

  def generate_samples(self, directory, fragment_length=0.02, infinite=True):
    """
    Generates samples from the given directory in random order
    :param directory: the sub-directory of the initial data directory to sample from
    :param fragment_length: the length of a input fragment in seconds
    :return: generator with (audio_fragments: ndarray, transcript: list(int)) tuples
    """
    audio_files = list(iglob_recursive(self._data_directory + '/' + directory, '*.flac'))
    # Infinite stream
    while True:
      random.shuffle(audio_files)
      for audio_file in audio_files:
        audio_data, samplerate = sf.read(audio_file, dtype='float32')
        audio_fragments = fragment_audio(audio_data, samplerate, fragment_length)

        file_name = os.path.basename(audio_file)
        audio_id = os.path.splitext(file_name)[0]

        transcript = self._transcript_dict[audio_id]

        yield audio_fragments, transcript

      if not infinite:
        return


class BucketPicker:
  """
  Based on the read samples it fills and yields buckets
  """

  def __init__(self, sample_generator, bucket_sizes, threshold):
    self._sample_generator = sample_generator
    self._bucket_sizes = bucket_sizes
    self._buckets = [list() for i in range(len(bucket_sizes))]
    self._threshold = threshold

  def _find_matching_bucket(self, sample):
    sample_source_size = sample[0].shape[0]
    sample_target_size = len(sample[1])

    # Search for a bucket the sample fits in
    for bucket_id, (source_size, target_size) in enumerate(self._bucket_sizes):
      # FIXME why not <=, because of GO symbol?
      if sample_source_size < source_size and sample_target_size < target_size:
        bucket = self._buckets[bucket_id]
        return bucket, bucket_id
    warnings.warn('No bucket found for size source: {}, target: {}'.format(sample_source_size, sample_target_size))
    return None, -1

  def generate_buckets(self):
    """
    Generate buckets, always yielding the next bucket that exceeds threshold
    """
    for sample in self._sample_generator:
      bucket, bucket_id = self._find_matching_bucket(sample)
      if bucket is not None:
        bucket.append(sample)

        # Bucket is filled, let's yield it
        if len(bucket) >= self._threshold:
          yield bucket, self._bucket_sizes[bucket_id], bucket_id
          bucket.clear()

  def generate_all_buckets(self):
    """
    Yields full buckets of all types
    """
    for sample in self._sample_generator:
      bucket, bucket_id = self._find_matching_bucket(sample)
      if bucket is not None and len(bucket) < self._threshold:
        bucket.append(sample)

        # All buckets filled, let's yield them
        # FIXME what if a bucket is never filled?
        if all([len(b) >= self._threshold for b in self._buckets]):
          yield self._buckets
          for b in self._buckets:
            b.clear()