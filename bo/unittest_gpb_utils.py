"""
  Unit tests for gpb_utils.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=relative-import

import numpy as np
# Local imports
import gpb_utils
from utils.base_test_class import BaseTestClass, execute_tests



class MFGPBUtilsTestCase(BaseTestClass):
  """ Unit tests for generic functions gpb_utils.py """

  def setUp(self):
    """ Sets up unit tests. """
    self.lhs_data = [(1, 10), (2, 5), (4, 10), (10, 100)]

  @classmethod
  def _check_sample_sizes(cls, data, samples):
    """ Data is a tuple of the form (dim, num_samples) ans samples is an ndarray."""
    assert (data[1], data[0]) == samples.shape

  def test_latin_hc_indices(self):
    """ Tests latin hyper-cube index generation. """
    self.report('Test Latin hyper-cube indexing. Only a sufficient condition check.')
    for data in self.lhs_data:
      lhs_true_sum = data[1] * (data[1] - 1) / 2
      lhs_idxs = gpb_utils.latin_hc_indices(data[0], data[1])
      lhs_idx_sums = np.array(lhs_idxs).sum(axis=0)
      assert np.all(lhs_true_sum == lhs_idx_sums)

  def test_latin_hc_sampling(self):
    """ Tests latin hyper-cube sampling. """
    self.report('Test Latin hyper-cube sampling. Only a sufficient condition check.')
    for data in self.lhs_data:
      lhs_max_sum = float(data[1] + 1)/2
      lhs_min_sum = float(data[1] - 1)/2
      lhs_samples = gpb_utils.latin_hc_sampling(data[0], data[1])
      lhs_sample_sums = lhs_samples.sum(axis=0)
      self._check_sample_sizes(data, lhs_samples)
      assert lhs_sample_sums.max() <= lhs_max_sum
      assert lhs_sample_sums.min() >= lhs_min_sum

  def test_random_sampling(self):
    """ Tests random sampling. """
    self.report('Test random sampling.')
    for data in self.lhs_data:
      self._check_sample_sizes(data, gpb_utils.random_sampling(data[0], data[1]))

  def test_random_sampling_kmeans(self):
    """ Tests random sampling with k-means. """
    self.report('Test random sampling with k-means.')
    for data in self.lhs_data:
      self._check_sample_sizes(data, gpb_utils.random_sampling_kmeans(data[0], data[1]))


if __name__ == '__main__':
  execute_tests()

