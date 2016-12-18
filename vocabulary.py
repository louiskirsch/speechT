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

APOSTROPHE = 26
SPACE_ID = 27

A_ASCII_CODE = ord('a')

SIZE = 28


def letter_to_id(letter):
  if letter == ' ':
    return SPACE_ID
  if letter == '\'':
    return APOSTROPHE
  return ord(letter) - A_ASCII_CODE


def id_to_letter(identifier):
  if identifier == SPACE_ID:
    return ' '
  if identifier == APOSTROPHE:
    return '\''
  return chr(identifier + A_ASCII_CODE)


def sentence_to_ids(sentence):
  return [letter_to_id(letter) for letter in sentence.lower()]


def ids_to_sentence(identifiers):
  return ''.join(id_to_letter(identifier) for identifier in identifiers)
