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
  """
  Converts `letter` to vocabulary id

  Args:
    letter: letter to convert, allowed is a-z, apostrophe and space

  Returns: the vocabulary encoded letter

  """
  if letter == ' ':
    return SPACE_ID
  if letter == '\'':
    return APOSTROPHE
  return ord(letter) - A_ASCII_CODE


def id_to_letter(identifier):
  """
  Converts the vocabulary encoded letter `identifier` to its character representation

  Args:
    identifier: encoded letter to decode

  Returns: the character letter

  """
  if identifier == SPACE_ID:
    return ' '
  if identifier == APOSTROPHE:
    return '\''
  return chr(identifier + A_ASCII_CODE)


def sentence_to_ids(sentence):
  """
  Convert a string `sentence` to its encoded representation

  Args:
    sentence: sentence of type string

  Returns: list of ints (encoded characters)

  """
  return [letter_to_id(letter) for letter in sentence.lower()]


def ids_to_sentence(identifiers):
  """
  Convert an complete list of encoded characters `identifiers` to their character representation

  Args:
    identifiers:  list of ints (encoded characters)

  Returns: decoded sentence as string

  """
  return ''.join(id_to_letter(identifier) for identifier in identifiers)
