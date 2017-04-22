# Copyright 2016 Louis Kirsch. All Rights Reserved.
#
# based on http://stackoverflow.com/questions/892199/detect-record-audio-in-python
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

from sys import byteorder
from array import array

import pyaudio


class AudioRecorder:

  def __init__(self, rate=16000, threshold=0.03, chunk_size=1024):
    self.rate = rate
    self.threshold = threshold
    self.chunk_size = chunk_size
    self.format = pyaudio.paFloat32
    self._pyaudio = pyaudio.PyAudio()

  def is_silent(self, snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < self.threshold

  def normalize(self, snd_data):
    "Average the volume out"
    MAXIMUM = 0.5
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)

    r = array('f')
    for i in snd_data:
      r.append(i * times)
    return r

  def trim(self, snd_data):
    "Trim the blank spots at the start and end"

    def _trim(snd_data):
      snd_started = False
      r = array('f')

      for i in snd_data:
        if not snd_started and abs(i) > self.threshold:
          snd_started = True
          r.append(i)

        elif snd_started:
          r.append(i)
      return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

  def add_silence(self, snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('f', [0 for i in range(int(seconds * self.rate))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds * self.rate))])
    return r

  def record(self):
    """
    Record a word or words from the microphone and
    return the data as an array of signed floats.

    Normalizes the audio, trims silence from the
    start and end, and pads with 0.5 seconds of
    blank sound to make sure VLC et al can play
    it without getting chopped off.
    """
    stream = self._pyaudio.open(format=self.format, channels=1, rate=self.rate,
                                input=True, output=True,
                                frames_per_buffer=self.chunk_size)

    num_silent = 0
    snd_started = False

    r = array('f')

    while 1:
      # little endian, signed short
      snd_data = array('f', stream.read(self.chunk_size))
      if byteorder == 'big':
        snd_data.byteswap()
      r.extend(snd_data)

      silent = self.is_silent(snd_data)

      if silent and snd_started:
        num_silent += 1
      elif not silent and not snd_started:
        snd_started = True

      if snd_started and num_silent > 30:
        break

    sample_width = self._pyaudio.get_sample_size(self.format)
    stream.stop_stream()
    stream.close()

    r = self.normalize(r)
    r = self.trim(r)
    r = self.add_silence(r, 0.1)
    return r, sample_width

  def terminate(self):
    self._pyaudio.terminate()
