# Copyright 2016 Louis Kirsch. All Rights Reserved.
#
# based on
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import math
import os
import sys
import time
import pickle

import numpy as np
import tensorflow as tf

import data_utils
import record
import seq2seq_model

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_string("data_dir", "data/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "train/", "Training directory.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("decode_tests", False,
                            "Set to True for automatic test set decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

SAMPLERATE = 16000

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(375, 180), (570, 250), (650, 300), (770, 350)]

# Adapt buckets to FLAGS.size
for i, bucket in enumerate(_buckets):
  _buckets[i] = ((int)(bucket[0] * 1024 / FLAGS.size), bucket[1])

def create_model(session, forward_only):
  """Create speechT model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      data_utils.Vocabulary.SIZE,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train():
  """Train a speech->text model using OpenSLR data."""

  # Obtain data
  corpus_provider = data_utils.SpeechCorpusProvider(FLAGS.data_dir)
  corpus_provider.ensure_availability()

  with tf.Session() as sess:
    # Read data
    print ("Reading development and training data")
    reader = data_utils.SpeechCorpusReader(FLAGS.data_dir)
    # TODO fragment_length depends on FLAGS.size (embedding size), keep it that way?
    dev_set = reader.generate_samples(data_utils.SpeechCorpusProvider.DEV_DIR, FLAGS.size)
    train_set = reader.generate_samples(data_utils.SpeechCorpusProvider.TRAIN_DIR, FLAGS.size)
    bucket_picker_train = data_utils.BucketPicker(train_set, _buckets, FLAGS.batch_size)
    bucket_picker_dev = data_utils.BucketPicker(dev_set, _buckets, FLAGS.batch_size)

    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    # Create training directory if it does not exist
    if not os.path.exists(FLAGS.train_dir):
      os.makedirs(FLAGS.train_dir)

    # Create a bucket batch generator, yielding buckets of each type at a time
    bucket_batch_dev = bucket_picker_dev.generate_all_buckets()

    # This is the training loop, the bucket picker gives an endless stream of training data
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    print('Start training')
    sys.stdout.flush()
    for bucket, bucket_size, bucket_id in bucket_picker_train.generate_buckets():
      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(bucket, bucket_size)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "speechT.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id, bucket in enumerate(next(bucket_batch_dev)):
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            bucket, _buckets[bucket_id])
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              "inf")
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()


def init_forward_session(sess):
  # Create model and load parameters.
  model = create_model(sess, True)
  model.batch_size = 1  # We decode one sentence at a time.

  return model


def decode_audio_fragments(audio_fragments, model, sess, vocabulary):
  # Which bucket does it belong to?
  bucket_id = min([b for b in range(len(_buckets))
                   if _buckets[b][0] > audio_fragments.shape[0]], default=None)

  # Audio is too long if no bucket was found
  if bucket_id is None:
    print('Error: Audio too long')
    return

  # Get a 1-element batch to feed the sentence to the model.
  encoder_inputs, decoder_inputs, target_weights = model.get_batch(
    [(audio_fragments, [])], _buckets[bucket_id])
  # Get output logits for the sentence.
  _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, True)
  # This is a greedy decoder - outputs are just argmaxes of output_logits.
  outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
  # If there is an EOS symbol in outputs, cut them at that point.
  if data_utils.Vocabulary.EOS_ID in outputs:
    outputs = outputs[:outputs.index(data_utils.Vocabulary.EOS_ID)]
  # Return transcribed sentence corresponding to outputs.
  return vocabulary.ids_to_sentence(outputs)


def decode():
  with tf.Session() as sess:
    model = init_forward_session(sess)
    vocabulary = data_utils.Vocabulary()

    recorder = record.AudioRecorder(rate=SAMPLERATE)

    while True:
      # Record audio
      sys.stdout.write("\nRecording audio... ")
      audio_data, _ = recorder.record()
      audio_data = np.array(audio_data)
      audio_fragments = data_utils.fragment_audio(audio_data, SAMPLERATE, FLAGS.size)
      print('Audio recorded with bucket length {}\n'.format(audio_fragments.shape[0]))
      sys.stdout.flush()

      # Decode and print
      result = decode_audio_fragments(audio_fragments, model, sess, vocabulary)
      print(result)
      sys.stdout.flush()


def decode_tests():
  with tf.Session() as sess:
    model = init_forward_session(sess)
    vocabulary = data_utils.Vocabulary()

    # Obtain data
    corpus_provider = data_utils.SpeechCorpusProvider(FLAGS.data_dir)
    corpus_provider.ensure_availability(test_only=True)

    # Read data
    print("Reading test data")
    reader = data_utils.SpeechCorpusReader(FLAGS.data_dir)
    test_set = reader.generate_samples(data_utils.SpeechCorpusProvider.TEST_DIR, FLAGS.size, infinite=False)

    for audio_fragments, expected_output in test_set:
      expected_output = vocabulary.ids_to_sentence(expected_output)

      # TODO Play input

      # Decode and print
      result = decode_audio_fragments(audio_fragments, model, sess, vocabulary)
      if result is not None:
        print('Understood: ' + result)
        print('Expected: ' + expected_output)
        sys.stdout.flush()

def self_test():
  """Test the speechT model."""
  # TODO create simple model, test functionality
  pass


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode_tests:
    decode_tests()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
