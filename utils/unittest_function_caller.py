"""
  Unit tests for function caller.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=maybe-no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=abstract-class-not-used

import numpy as np
# Local
from base_test_class import BaseTestClass, execute_tests
from syn_functions import get_syn_function_caller_from_name


class FunctionCallerTestCase(BaseTestClass):
  """ Unit tests for FunctionCaller class. """

  def __init__(self, *args, **kwargs):
    super(FunctionCallerTestCase, self).__init__(*args, **kwargs)

  def setUp(self):
    """ Set up attributes. """
    self.synthetic_functions = ['hartmann3', 'hartmann6', 'hartmann-23', 'shekel',
                                'branin-20', 'branin-31', 'shekel-40']
#                                 'branin']
    self.num_samples = 10000

  def test_max_vals(self):
    """ Tests for the maximum of the function. """
    self.report('Testing for Maximum value. ')
    for test_fn_name in self.synthetic_functions:
      caller = get_syn_function_caller_from_name(test_fn_name)
      self.report('Testing %s with max value %0.4f.'%(test_fn_name, caller.opt_val),
                  'test_result')
      X = np.random.random((self.num_samples, caller.domain_dim))
      eval_vals, _ = caller.eval_multiple(X)
      assert eval_vals.max() <= caller.opt_val
      if caller.opt_pt_normalised is not None:
        max_val_norm = caller.eval_single(caller.opt_pt_normalised, normalised=True)
        max_val_unnorm = caller.eval_single(caller.opt_pt_unnormalised, normalised=False)
        assert np.abs(max_val_norm[0] - caller.opt_val) < 1e-5
        assert np.abs(max_val_unnorm[0] - caller.opt_val) < 1e-5


class FunctionCallerWithRandomTimeTestCase(FunctionCallerTestCase):
  """ Unit tests for FunctionCallerWithRandomTimeTestCase. """

  def __init__(self, *args, **kwargs):
    super(FunctionCallerWithRandomTimeTestCase, self).__init__(*args, **kwargs)

  def setUp(self):
    """ Set up attributes. """
    super(FunctionCallerWithRandomTimeTestCase, self).setUp()
    self.time_distros = ['const', 'uniform', 'halfnormal', 'exponential', 'pareto']
    self.time_means = [1.0] * 5
    self.time_vars = [0.0, 1/3.0, np.pi/2 - 1, 1, 1/15.0]

  def testTimeDistros(self):
    """ Tests for the time. """
    self.report('Testing for time. Probabilistic test, might fail.')
    for idx in range(len(self.time_distros)):
      time_distro = self.time_distros[idx]
      caller = get_syn_function_caller_from_name('hartmann3', time_distro=time_distro)
      X = np.random.random((self.num_samples, caller.domain_dim))
      _, anc = caller.eval_multiple(X)
#       import pdb; pdb.set_trace()
      eval_times_mean = anc.eval_times.mean()
      eval_times_var = anc.eval_times.std() ** 2
      msg = '%s:: mean (true, emp) = (%0.2f, %0.2f), var (true, emp) = (%0.2f, %0.2f).'%(
            time_distro, self.time_means[idx], eval_times_mean, self.time_vars[idx],
            eval_times_var)
      self.report(msg, 'test_result')
      assert np.abs(eval_times_mean - self.time_means[idx]) < 2e-1
      assert np.abs(eval_times_var - self.time_vars[idx]) < 2e-1


if __name__ == '__main__':
  execute_tests()

