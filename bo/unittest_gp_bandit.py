"""
  Unit tests for GP-Bandits
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=maybe-no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class

import numpy as np
# Local
from utils.syn_functions import get_syn_function_caller_from_name
from worker_manager import SyntheticWorkerManager
import gp_bandit
from utils.ancillary_utils import is_non_decreasing_sequence
from utils.base_test_class import BaseTestClass, execute_tests
import utils.reporters as reporters
from utils import option_handler


class GPBanditTestCase(BaseTestClass):
  """ Unit tests for gp_bandit.GPBandit. """

  def setUp(self):
    """ Set up. """
    self.func_caller = get_syn_function_caller_from_name('Hartmann3')
    self.worker_manager_1 = SyntheticWorkerManager(self.func_caller, 1)

  def test_instantiation(self):
    """ Tests creation of object. """
    self.report('Testing GP Bandit instantiation.')
    gpb = gp_bandit.GPBandit(self.func_caller, self.worker_manager_1,
                             reporter=reporters.SilentReporter())
    assert gpb.domain_dim == self.func_caller.domain_dim
    self.report('Instantiated GPBandit object.')
    for attr in dir(gpb):
      if not attr.startswith('_'):
        self.report('gpb.%s = %s'%(attr, str(getattr(gpb, attr))))


# class GPBanditAPITestCase(BaseTestClass):
#   """ Unit tests for the two APIs in gp_bandit.py """
# 
#   def test_gpb_from_func_caller(self):
#     """ Unig test for gpb_from_func_caller """

if __name__ == '__main__':
  execute_tests()

