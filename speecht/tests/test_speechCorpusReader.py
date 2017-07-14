import os
import shutil
from unittest import TestCase

import numpy as np

from speecht import preprocessing
from speecht.preprocessing import SpeechCorpusReader


class TestSpeechCorpusReader(TestCase):

  BASE_DIR = 'tests/data/'
  TEST_FILES_DIR = 'train'
  PREPROCESS_DIR = 'tests/data/preprocessed'
  SAMPLE_FILE = BASE_DIR + TEST_FILES_DIR + '/1089-134686-0037.flac'

  def setUp(self):
    self.reader = SpeechCorpusReader(self.BASE_DIR)

  def tearDown(self):
    if os.path.exists(self.PREPROCESS_DIR):
      shutil.rmtree(self.PREPROCESS_DIR)

  def test__get_transcript_entries(self):
    entries = list(SpeechCorpusReader._get_transcript_entries(self.BASE_DIR))

    firstEntry = ['1089-134686-0000',
                 'HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON '
                 'PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE']
    lastEntry = ['1089-134686-0037',
                 'IN THE SILENCE THEIR DARK FIRE KINDLED THE DUSK INTO A TAWNY GLOW']

    self.assertEquals(firstEntry, entries[0])
    self.assertEquals(lastEntry, entries[-1])

  def _transform_sample(self):
    return SpeechCorpusReader._transform_sample(self.SAMPLE_FILE, lambda x, y: x)

  def test__transform_sample(self):
    transformed = self._transform_sample()
    audio_id, audio_fragments = transformed

    self.assertEqual(audio_id, '1089-134686-0037')
    self.assertEqual(audio_fragments.shape, (114881,))

  def test_generate_samples(self):
    samples = list(self.reader.generate_samples(self.TEST_FILES_DIR, lambda x, y: x))

    self.assertEqual(len(samples), 1)

    audio_id, audio_fragments, transcript = samples[0]
    expected_audio_id, expected_audio_fragments = self._transform_sample()

    self.assertEqual(audio_id, expected_audio_id)
    self.assertTrue(np.array_equal(audio_fragments, expected_audio_fragments))

  def test_store_samples(self):
    self.reader.store_samples(self.TEST_FILES_DIR, preprocessing.calc_mfccs)
    self.assertTrue(os.path.exists(self.BASE_DIR + 'preprocessed/' + self.TEST_FILES_DIR + '/1089-134686-0037.npz'))

  def test_load_samples(self):
    self.reader.store_samples(self.TEST_FILES_DIR, preprocessing.calc_mfccs)
    samples_stored = list(self.reader.load_samples(self.TEST_FILES_DIR))
    samples_generated = [(audio_fragments, transcript) for audio_id, audio_fragments, transcript
                         in self.reader.generate_samples(self.TEST_FILES_DIR, preprocessing.calc_mfccs)]

    self.assertEqual(len(samples_stored), 1)
    self.assertEqual(len(samples_generated), 1)
    # assert audio_fragments
    self.assertTrue(np.array_equal(samples_generated[0][0], samples_stored[0][0]))
    # assert transcript
    self.assertTrue(np.array_equal(samples_generated[0][1], samples_stored[0][1]))
