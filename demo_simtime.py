"""
  A simple demo for parallelised BO via Thompson sampling on synthetic functions and
  simulated time units.
  The time for each evaluation is sampled from the specified random distribution.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=relative-import

# Local imports
from bo.blackbox_optimiser import random_optimiser_from_func_caller
from bo.gp_bandit import gpb_from_func_caller
from bo.worker_manager import SyntheticWorkerManager
from utils.syn_functions import get_syn_function_caller_from_name


## Set the parameters in all caps ------------------------------------------------
NUM_WORKERS = 2
MAX_CAPITAL = 40


## Parameters for the Synthetic Experiment ---------------------------------------
# 1. Choose Synthetic Function
# EXP_NAME = 'Hartmann3'
# EXP_NAME = 'Hartmann6'
# EXP_NAME = 'CurrinExp'
EXP_NAME = 'Branin'
# EXP_NAME = 'Shekel'
# EXP_NAME = 'Hartmann12'
# EXP_NAME = 'Park1'
# EXP_NAME = 'Park2'
# EXP_NAME = 'Branin-4'
# EXP_NAME = 'CurrinExp-20'

# 2. Choose distribution for time
# TIME_DISTRO = 'uniform'
TIME_DISTRO = 'halfnormal'
# TIME_DISTRO = 'exponential'
# TIME_DISTRO = 'pareto';


## Other parameters that are not critical ----------------------------------------
NOISE_SCALE = 0.1 # standard deviation of gaussian noise


def main():
  """ Main function. """

  func_caller = get_syn_function_caller_from_name(EXP_NAME, noise_type='gauss',
                  noise_scale=NOISE_SCALE, time_distro=TIME_DISTRO)
  worker_manager = SyntheticWorkerManager(func_caller, NUM_WORKERS)

  # 1. random synchronous
  worker_manager.reset()
  syn_rand_opt_val, _, syn_rand_history = random_optimiser_from_func_caller(
    func_caller, worker_manager, MAX_CAPITAL, 'syn')
  syn_rand_true_opt_val = syn_rand_history.curr_true_opt_vals[-1]

  # 2. random asynchronous
  worker_manager.reset()
  asy_rand_opt_val, _, asy_rand_history = random_optimiser_from_func_caller(
    func_caller, worker_manager, MAX_CAPITAL, 'asy')
  asy_rand_true_opt_val = asy_rand_history.curr_true_opt_vals[-1]

  # 3. ts synchronous
  worker_manager.reset()
  syn_ts_opt_val, _, syn_ts_history = gpb_from_func_caller(
    func_caller, worker_manager, MAX_CAPITAL, 'syn', 'TS')
  syn_ts_true_opt_val = syn_ts_history.curr_true_opt_vals[-1]

  # 4. ts asynchronous
  worker_manager.reset()
  asy_ts_opt_val, _, asy_ts_history = gpb_from_func_caller(
    func_caller, worker_manager, MAX_CAPITAL, 'asy', 'TS')
  asy_ts_true_opt_val = asy_ts_history.curr_true_opt_vals[-1]

  # Print out results
  print '\nExperiment: %s, true-max-value: %0.4f'%(EXP_NAME, func_caller.opt_val)
  print 'Synchronous  random: noisy-max-value: %0.4f, true-max-value: %0.4f'%(
         syn_rand_opt_val, syn_rand_true_opt_val)
  print 'Asynchronous random: noisy-max-value: %0.4f, true-max-value: %0.4f'%(
         asy_rand_opt_val, asy_rand_true_opt_val)
  print 'Synchronous  ts: noisy-max-value: %0.4f, true-max-value: %0.4f'%(
         syn_ts_opt_val, syn_ts_true_opt_val)
  print 'Asynchronous ts: noisy-max-value: %0.4f, true-max-value: %0.4f'%(
         asy_ts_opt_val, asy_ts_true_opt_val)


if __name__ == '__main__':
  main()

