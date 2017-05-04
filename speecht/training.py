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

import os
import time

import numpy as np
import tensorflow as tf
from speecht.execution import DatasetExecutor

from speecht.speech_model import create_default_model


class Training(DatasetExecutor):

  def create_sample_generator(self, limit_count: int):
    return self.reader.load_samples('train',
                                    loop_infinitely=True,
                                    limit_count=limit_count,
                                    feature_type=self.flags.feature_type)

  def get_loader_limit_count(self) -> int:
    return self.flags.limit_training_set

  def create_model(self, sess):
    model = create_default_model(self.flags, self.input_size, self.speech_input)
    model.restore_or_create(sess,
                            self.flags.run_train_dir,
                            self.flags.learning_rate if self.flags.reset_learning_rate else None)
    return model

  def run(self):

    with tf.Session() as sess:

      model = self.create_model(sess)
      coord = self.start_pipeline(sess, n_threads=2)

      step_time, loss = 0.0, 0.0
      current_step = 0
      previous_losses = []

      try:
        print('Begin training')
        while not coord.should_stop():

          current_step += 1
          is_checkpoint_step = current_step % self.flags.steps_per_checkpoint == 0

          start_time = time.time()
          step_result = model.step(sess, summary=is_checkpoint_step)
          avg_loss = step_result[0]
          step_time += (time.time() - start_time) / self.flags.steps_per_checkpoint
          loss += avg_loss / self.flags.steps_per_checkpoint

          # Once in a while, we save checkpoint and print statistics
          if is_checkpoint_step:
            global_step = model.global_step.eval()

            # Print statistics for the previous step.
            perplexity = np.exp(float(avg_loss)) if avg_loss < 300 else float("inf")
            print("global step {:d} learning rate {:.4f} step-time {:.2f} average loss {:.2f} perplexity {:.2f}"
                  .format(global_step, model.learning_rate.eval(), step_time, avg_loss, perplexity))

            # Retrieve and store summary
            summary = step_result[2]
            model.summary_writer.add_summary(summary, global_step)

            # Decrease learning rate if no improvement was seen over last 3 times.
            if self.flags.learning_rate_decay_factor > 0 and len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
              sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)

            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(self.flags.run_train_dir, "speechT.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            print('Model saved')
            step_time, loss = 0.0, 0.0

      except tf.errors.OutOfRangeError:
        print('Done training -- step limit reached')
      finally:
        # When done, ask the threads to stop.
        coord.request_stop()

      coord.join()

