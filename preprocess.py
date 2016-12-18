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

import librosa
import os

import logging
import numpy as np
import fnmatch
import random
import vocabulary
import corpus
import argparse


def calc_spectrogram(audio_data, samplerate, number_mels):
  spectrogram = librosa.feature.melspectrogram(audio_data, sr=samplerate, n_mels=number_mels)

  # Convert to log scale (dB). We'll use the peak power as reference.
  # TODO why do we do this?
  log_spectrogram = librosa.logamplitude(spectrogram, ref_power=np.max)

  return log_spectrogram.T


def iglob_recursive(directory, file_pattern):
  for root, dir_names, file_names in os.walk(directory):
    for filename in fnmatch.filter(file_names, file_pattern):
      yield os.path.join(root, filename)


class SpeechCorpusReader:
  """
  Reads and transforms the speech corpus to be used by the NN
  """
  def __init__(self, data_directory):
    self._data_directory = data_directory
    self._transcript_dict_cache = None

  @property
  def _transcript_dict(self):
    if not self._transcript_dict_cache:
      self._transcript_dict_cache = self._build_transcript()
    return self._transcript_dict_cache

  @staticmethod
  def _get_transcript_entries(transcript_directory):
    """
    Iterate over all transcript lines and yield splitted entries
    Args:
      transcript_files: transcript files to iterate over

    Returns: Iterator for all splitted entries

    """
    transcript_files = iglob_recursive(transcript_directory, '*.trans.txt')
    for transcript_file in transcript_files:
      with open(transcript_file, 'r') as f:
        for line in f:
          # Strip included new line symbol
          line = line.rstrip('\n')

          # Each line is in the form
          # 00-000000-0000 WORD1 WORD2 ...
          splitted = line.split(' ', 1)
          yield splitted

  def _build_transcript(self):
    """
    Builds a transcript from transcript files, mapping from audio-id to a list of vocabulary ids
    :return: the created transcript
    """

    # Create the transcript dictionary
    transcript_dict = dict()
    for splitted in self._get_transcript_entries(self._data_directory):
      transcript_dict[splitted[0]] = vocabulary.sentence_to_ids(splitted[1])

    return transcript_dict

  def generate_samples(self, directory, number_mels):
    """
    Generates samples from the given directory
    :param directory: the sub-directory of the initial data directory to sample from
    :param number_mels: the parameter number_mels to use for the spectrogram
    :return: generator with (audio_fragments: ndarray, transcript: list(int)) tuples
    """
    audio_files = list(iglob_recursive(self._data_directory + '/' + directory, '*.flac'))

    transcript_dict = self._transcript_dict

    for audio_file in audio_files:
      audio_data, samplerate = librosa.load(audio_file)
      audio_fragments = calc_spectrogram(audio_data, samplerate, number_mels)

      file_name = os.path.basename(audio_file)
      audio_id = os.path.splitext(file_name)[0]

      transcript = transcript_dict[audio_id]

      yield audio_id, audio_fragments, transcript

  def store_samples(self, directory, number_mels):

    out_directory = self._data_directory + '/preprocessed/' + directory
    if not os.path.exists(out_directory):
      os.makedirs(out_directory)

    for audio_id, audio_fragments, transcript in self.generate_samples(directory, number_mels):
      np.savez(out_directory + '/' + audio_id, audio_fragments=audio_fragments, transcript=transcript)

  def load_samples(self, directory, max_size):

    load_directory = self._data_directory + '/preprocessed/' + directory

    files = list(iglob_recursive(load_directory, '*.npz'))
    random.shuffle(files)

    for file in files:
      with np.load(file) as data:
        audio_length = data['audio_fragments'].shape[0]
        if audio_length <= max_size:
          yield data['audio_fragments'], data['transcript']
        else:
          logging.warning('Audio snippet too long: {}'.format(audio_length))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Generate preprocessed file from audio files and transcripts')
  parser.add_argument('--data_directory', type=str, required=False, default='data',
                      help='the data directory to pull the files from and store the preprocessed file')
  parser.add_argument('--number_mels', type=int, required=False, default=128,
                      help='number of mels to be generated')
  args = parser.parse_args()

  corpus = corpus.SpeechCorpusProvider(args.data_directory)
  corpus.ensure_availability()
  corpus_reader = SpeechCorpusReader(args.data_directory)

  print('Preprocessing training data')
  corpus_reader.store_samples('train', args.number_mels)

  print('Preprocessing test data')
  corpus_reader.store_samples('test', args.number_mels)

  print('Preprocessing development data')
  corpus_reader.store_samples('dev', args.number_mels)
