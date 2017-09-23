"""
  Unit tests for Black box optimiser
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=maybe-no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class

# Local
import blackbox_optimiser
import utils.reporters as reporters
from utils.syn_functions import get_syn_function_caller_from_name
from utils.base_test_class import BaseTestClass, execute_tests
from worker_manager import SyntheticWorkerManager


class RandomOptimiserTestCase(BaseTestClass):
  """ Unit test for Random Optimisation. """

  def setUp(self):
    """ Set up. """
    self.func_caller = get_syn_function_caller_from_name('Hartmann3',
                         noise_type='gauss', noise_scale=0.1, time_distro='exponential')
    self.worker_manager_1 = SyntheticWorkerManager(self.func_caller, 1)
    self.worker_manager_4 = SyntheticWorkerManager(self.func_caller, 4)

  def test_instantiation(self):
    """ Test creation of object. """
    self.report('Testing Random Optimiser instantiation.')
    optimiser = blackbox_optimiser.RandomOptimiser(self.func_caller,
      self.worker_manager_1, reporter=reporters.SilentReporter())
    assert optimiser.domain_dim == self.func_caller.domain_dim
    self.report('Instantiated RandomOptimiser object.')
    for attr in dir(optimiser):
      if not attr.startswith('_'):
        self.report('optimiser.%s = %s'%(attr, str(getattr(optimiser, attr))))

  def _test_optimiser_results(self, opt_val, opt_pt, history):
    """ Tests optimiser results. """
    assert opt_val == history.curr_opt_vals[-1]
    assert opt_pt.shape[0] == self.func_caller.domain_dim

  def test_optimisation_single(self):
    """ Test optimisation. """
    self.report('Testing Random Optimiser with just one worker.')
    opt_val, opt_pt, history = blackbox_optimiser.random_optimiser_from_func_caller(
      self.func_caller, self.worker_manager_1, 100, 'asy')
    self._test_optimiser_results(opt_val, opt_pt, history)
    self.report('')

  def test_optimisation_asynchronous(self):
    """ Test optimisation. """
    self.report('Testing Random Optimiser with just four workers asynchronously.')
    opt_val, opt_pt, history = blackbox_optimiser.random_optimiser_from_func_caller(
      self.func_caller, self.worker_manager_4, 100, 'asy')
    self._test_optimiser_results(opt_val, opt_pt, history)
    self.report('')

  def test_optimisation_synchronous(self):
    """ Test optimisation. """
    self.report('Testing Random Optimiser with just four workers synchronously.')
    opt_val, opt_pt, history = blackbox_optimiser.random_optimiser_from_func_caller(
      self.func_caller, self.worker_manager_4, 100, 'syn')
    self._test_optimiser_results(opt_val, opt_pt, history)
    self.report('')


if __name__ == '__main__':
  execute_tests()

