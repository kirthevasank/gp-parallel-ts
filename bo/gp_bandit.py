"""
  Harness for GP Bandit Optimisation.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class

from argparse import Namespace
import numpy as np

# Local imports
from bo import acquisitions
from blackbox_optimiser import blackbox_opt_args, BlackboxOptimiser
from gp.kernel import SEKernel
from gp.gp_core import GP, mandatory_gp_args
from gp.gp_instances import SimpleGPFitter, all_simple_gp_args
from utils.optimisers import random_maximise
from utils.option_handler import get_option_specs, load_options
from utils.function_caller import get_function_caller_from_function
from utils.reporters import get_reporter

gp_bandit_args = [
  # Acquisition
  get_option_specs('acq', False, None,
    'Which acquisition to use: TS, UCB, BUCB, UCBPE.'),
  get_option_specs('acq_opt_criterion', False, 'rand',
    'Which optimiser to use when maximising the acquisition function.'),
  get_option_specs('acq_opt_max_evals', False, -1,
    'Number of evaluations when maximising acquisition. If negative uses default value.'),
  # The following are perhaps not so important.
  get_option_specs('shrink_kernel_with_time', False, 0,
    'If True, shrinks the kernel with time so that we don\'t get stuck.'),
  get_option_specs('perturb_thresh', False, 1e-4,
    ('If the next point chosen is too close to an exisiting point by this times the '
     'diameter, then we will perturb the point a little bit before querying. This is '
     'mainly to avoid numerical stability issues.')),
  get_option_specs('track_every_time_step', False, 0,
    ('If 1, it tracks every time step.')),
  # TODO: implement code for next_pt_std_thresh
  get_option_specs('next_pt_std_thresh', False, 0.005,
    ('If the std of the queried point queries below this times the kernel scale ',
     'frequently we will reduce the bandwidth range')),
  ]

all_gp_bandit_args = all_simple_gp_args + blackbox_opt_args + gp_bandit_args


# The GPBandit Class
# ========================================================================================
class GPBandit(BlackboxOptimiser):
  """ GPBandit Class. """
  # pylint: disable=attribute-defined-outside-init

  # Constructor.
  def __init__(self, func_caller, worker_manager, options=None, reporter=None):
    """ Constructor. """
    self.reporter = get_reporter(reporter)
    if options is None:
      options = load_options(all_gp_bandit_args, reporter=reporter)
    super(GPBandit, self).__init__(func_caller, worker_manager, options, self.reporter)

  def _child_set_up(self):
    """ Some set up for the GPBandit class. """
    # Set up acquisition optimisation
    self._set_up_acq_opt()
    self.method_name = 'GP-' + str(self.options.acq)

  def _set_up_acq_opt(self):
    """ Sets up optimisation for acquisition. """
    # First set up function to get maximum evaluations.
    if isinstance(self.options.acq_opt_max_evals, int):
      if self.options.acq_opt_max_evals > 0:
        self.get_acq_opt_max_evals = lambda t: self.options.acq_opt_max_evals
      else:
        self.get_acq_opt_max_evals = None
    else: # In this case, the user likely passed a function here.
      self.get_acq_opt_max_evals = self.options.acq_opt_max_evals
    # Additional set up based on the specific optimisation procedure
    if self.options.acq_opt_criterion == 'direct':
      self._set_up_acq_opt_direct()
    elif self.options.acq_opt_criterion == 'rand':
      self._set_up_acq_opt_rand()
    else:
      raise NotImplementedError('Not implemented acquisition optimisation for %s yet.'%(
                                self.options.acq_opt_criterion))

  def _set_up_acq_opt_direct(self):
    """ Sets up optimisation for acquisition using direct. """
    raise NotImplementedError('Not implemented DiRect yet.')

  def _set_up_acq_opt_rand(self):
    """ Sets up optimisation for acquisition using random search. """
    def _random_max_wrap(*args):
      """ A wrapper so as to only return optimal point."""
      _, opt_pt = random_maximise(*args)
      return opt_pt
    # Set this up in acq_optimise
    self.acq_optimise = lambda obj, max_evals: _random_max_wrap(obj, self.domain_bounds,
                                                                max_evals)
    if self.get_acq_opt_max_evals is None:
      lead_const = 10 * min(5, self.domain_dim)**2
      self.get_acq_opt_max_evals = lambda t: np.clip(
        lead_const * np.sqrt(min(t, 1000)), 2000, 3e4)
    # Acquisition function should be evaluated via multiple evaluations
    self.acq_query_type = 'multiple'

  # Managing the GP ---------------------------------------------------------
  def _child_build_new_model(self):
    """ Builds a new model. """
    self._build_new_gp()

  def _build_new_gp(self):
    """ Builds a GP with the data in history and stores in self.gp. """
    if hasattr(self.func_caller, 'init_gp') and self.func_caller.init_gp is not None:
      # If you know the true GP.
      raise NotImplementedError('Not implemented passing given GP yet.')
    else:
      if self.options.shrink_kernel_with_time:
        raise NotImplementedError('Not implemented kernel shrinking for the GP yet.')
      else:
        self.options.bandwidth_log_bounds = np.array([[0.0, 4.1]] * self.domain_dim)
      # Invoke the GP fitter.
      reg_X = np.concatenate((self.pre_eval_points, self.history.query_points), axis=0)
      reg_Y = np.concatenate((self.pre_eval_vals, self.history.query_vals), axis=0)
      gp_fitter = SimpleGPFitter(reg_X, reg_Y,
                                 options=self.options, reporter=self.reporter)
      self.gp, _ = gp_fitter.fit_gp()
      gp_fit_report_str = '    -- Fitting GP (j=%d): %s'%(self.step_idx, str(self.gp))
      self.reporter.writeln(gp_fit_report_str)

  def _add_data_to_model(self, qinfos):
    """ Add data to self.gp """
    if len(qinfos) == 0:
      return
    new_points = np.empty((0, self.domain_dim))
    new_vals = np.empty(0)
    for i in range(len(qinfos)):
      new_points = np.concatenate((new_points,
                                  qinfos[i].point.reshape(-1, self.domain_dim)), axis=0)
      new_vals = np.append(new_vals, [qinfos[i].val], axis=0)
    if self.gp is not None:
      self.gp.add_data(new_points, new_vals)

  # Methods needed for initialisation ----------------------------------------
  def _child_init(self):
    """ Any initialisation for a child class. """
    self._create_init_gp()

  def _create_init_gp(self):
    """ Creates an initial GP. """
    reg_X = np.concatenate((self.pre_eval_points, self.history.query_points), axis=0)
    reg_Y = np.concatenate((self.pre_eval_vals, self.history.query_vals), axis=0)
    range_Y = reg_Y.max() - reg_Y.min()
    mean_func = lambda x: np.array([np.median(reg_X)] * len(x))
    kernel = SEKernel(self.domain_dim, range_Y/4.0,
                      dim_bandwidths=0.05*np.sqrt(self.domain_dim))
    noise_var = (reg_Y.std()**2)/10
    self.gp = GP(reg_X, reg_Y, kernel, mean_func, noise_var)

  # Methods needed for optimisation ----------------------------------------
  def _get_ancillary_data_for_acquisition(self):
    """ Returns ancillary data for the acquisitions. """
    max_num_acq_opt_evals = self.get_acq_opt_max_evals(self.step_idx)
    return Namespace(max_evals=max_num_acq_opt_evals,
                     t=self.step_idx,
                     curr_max_val=self.curr_opt_val,
                     evals_in_progress=self.eval_points_in_progress)

  def _determine_next_eval_point(self):
    """ Determine the next point for evaluation. """
    anc_data = self._get_ancillary_data_for_acquisition()
    acq_to_use = getattr(acquisitions.asy, self.options.acq.lower())
    next_eval_point = acq_to_use(self.gp, self.acq_optimise, anc_data)
    return next_eval_point

  def _determine_next_batch_of_eval_points(self):
    """ Determine the next batch of eavluation points. """
    anc_data = self._get_ancillary_data_for_acquisition()
    acq_to_use = getattr(acquisitions.syn, self.options.acq.lower())
    next_batch_of_eval_points = acq_to_use(self.num_workers, self.gp, self.acq_optimise,
                                           anc_data)
    return next_batch_of_eval_points

  def update_model(self):
    """ Update the model. """
    raise NotImplementedError('Implement in a child class.!')

# GP Bandit class ends here
# =====================================================================================

# APIs for GP Bandit optimisation. ----------------------------------------------------

# 1. Optimisation from a FunctionCaller object.
def gpb_from_func_caller(func_caller, worker_manager, max_capital, mode=None, acq=None,
                         options=None, reporter='default'):
  """ GP Bandit optimisation from a utils.function_caller.FunctionCaller instance. """
  if options is None:
    reporter = get_reporter(reporter)
    options = load_options(all_gp_bandit_args, reporter=reporter)
    options.acq = acq
    options.mode = mode
  return (GPBandit(func_caller, worker_manager, options, reporter)).optimise(max_capital)

# 2. Optimisation from all args.
def gpb_from_args(func, domain_bounds, max_capital, acq=None, options=None, reporter=None,
                  vectorised=False, **kwargs):
  """ This function executes GP Bandit (Bayesian) Optimisation.
    Input Arguments:
      - func: The function to be optimised.
      - domain_bounds: The bounds for the domain.
      - max_capital: The maximum capital for optimisation.
      - options: A namespace which gives other options.
      - reporter: A reporter object to write outputs.
      - vectorised: If true, it means func take matrix inputs. If
          false, they take only single point inputs.
      - true_opt_pt, true_opt_val: The true optimum point and value (if known). Mostly for
          experimenting with synthetic problems.
      - time_distro: The time distribution to be used when sampling.
      - time_distro_params: parameters for the time distribution.
    Returns: (gpb_opt_pt, gpb_opt_val, history)
      - gpb_opt_pt, gpb_opt_val: The optimum point and value.
      - history: A namespace which contains a history of all the previous queries.
  """
  func_caller = get_function_caller_from_function(func, domain_bounds=domain_bounds,
                                                  vectorised=vectorised, **kwargs)
  return gpb_from_func_caller(func_caller, max_capital, acq, options, reporter)

