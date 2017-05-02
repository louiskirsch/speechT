from pathlib import Path

from speecht.speech_input import SingleInputLoader
from speecht.speech_model import create_default_model

import tensorflow as tf
import numpy as np


class Exporting:

  def __init__(self, flags):
    self.flags = flags

  def create_model(self, sess: tf.Session):
    input_loader = SingleInputLoader(self.flags.input_size)
    model = create_default_model(self.flags, self.flags.input_size, input_loader)
    model.restore(sess, self.flags.run_train_dir)
    return model

  def run(self):
    with tf.Session() as sess:
      self.create_model(sess)

      if self.flags.export_weights_dir:
        path = Path(self.flags.export_weights_dir)
        if not path.exists():
          path.mkdir()

        variables = tf.trainable_variables()
        values = sess.run(variables)

        for variable, value in zip(variables, values):
          file_path = path / variable.name
          parent_dir = Path(file_path.parent)
          if not parent_dir.exists():
            parent_dir.mkdir()

          # noinspection PyTypeChecker
          np.save(file_path, value)

        return

      print('Nothing to do.')
