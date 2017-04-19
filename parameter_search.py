import random
import bisect
from curses import wrapper
from typing import List, Iterable

from evaluation import Evaluation, EvalStatistics
import tensorflow as tf
import numpy as np

from speech_model import SpeechModel


class Candidate:

  def __init__(self, lm_weight: float, word_count_weight: float):
    self.score = None
    self.stats = None
    self.lm_weight = lm_weight
    self.word_count_weight = word_count_weight

  def __gt__(self, other):
    return self.score > other.score

  def __lt__(self, other):
    return self.score < other.score

  def __str__(self):
    return ('{:.2f} Candidate (lm_weight={:.2f}, wc_weight={:.2f}) '
            'has LER: {:.2f} WER: {:.2f}').format(self.score,
                                                  self.lm_weight,
                                                  self.word_count_weight,
                                                  self.stats.global_letter_error_rate,
                                                  self.stats.global_word_error_rate)

  def update_score(self, score: float, stats: EvalStatistics):
    self.score = score
    self.stats = stats

  def mutate(self, std: float):
    return Candidate(lm_weight=self.lm_weight + np.random.normal(loc=0, scale=std),
                     word_count_weight=self.word_count_weight + np.random.normal(loc=0, scale=std))


class LanguageModelParameterSearch(Evaluation):

  def __init__(self, flags):
    super().__init__(flags)
    self.candidates = []
    self.num_iterations = 0

  def create_sample_generator(self, limit_count: int):
    return self.reader.load_samples('dev',
                                    loop_infinitely=True,
                                    limit_count=limit_count,
                                    feature_type=self.flags.feature_type)

  def _update_score_for_candidate(self, model: SpeechModel, sess: tf.Session, candidate: Candidate):
    stats = EvalStatistics()
    feed_dict = {
      model.lm_weight: candidate.lm_weight,
      model.word_count_weight: candidate.word_count_weight
    }
    self.run_epoch(model, sess, stats, save=False, verbose=False, feed_dict=feed_dict)
    score = -(stats.global_letter_error_rate + stats.global_word_error_rate)
    candidate.update_score(score, stats)

  def get_loader_limit_count(self):
    return 0

  def get_max_epochs(self):
    return None

  def run(self):

    with tf.Session() as sess:

      model = self.create_model(sess)
      coord = self.start_pipeline(sess)

      def run_search(stdscr=None):
        if stdscr:
          stdscr.clear()
          stdscr.addstr(0, 0, 'Loading...')
          stdscr.refresh()

        new_candidate = Candidate(1.0, 1.0)
        self._update_score_for_candidate(model, sess, new_candidate)
        self.candidates.append(new_candidate)

        if stdscr:
          self.print_population(stdscr)
        else:
          print(new_candidate)

        while True:
          if coord.should_stop():
            break

          random_candidate = random.choice(self.candidates)
          new_candidate = random_candidate.mutate(self.flags.noise_std)
          self._update_score_for_candidate(model, sess, new_candidate)

          # Note: We're dealing with tiny populations, so O(n) is not an issue
          bisect.insort(self.candidates, new_candidate)

          if len(self.candidates) > self.flags.population_size:
            del self.candidates[0]

          self.num_iterations += 1

          if stdscr:
            self.print_population(stdscr)
          else:
            print(new_candidate)

        coord.request_stop()
        coord.join()

      if self.flags.use_ui:
        wrapper(run_search)
      else:
        run_search()

  def print_population(self, stdscr):
    stdscr.clear()
    stdscr.addstr(0, 0, 'Current population after {} iterations'.format(self.num_iterations))
    for idx, candidate in enumerate(reversed(self.candidates)):
      stdscr.addstr(idx + 2, 0, str(candidate))
    stdscr.refresh()
