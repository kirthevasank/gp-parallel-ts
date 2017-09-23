"""
  A module for fitting a GP and tuning its kernel.
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import


import sys
import numpy as np

from utils.general_utils import stable_cholesky, draw_gaussian_samples
from utils.optimisers import direct_ft_maximise
from utils.option_handler import get_option_specs, load_options
from utils.reporters import get_reporter


# These are mandatory requirements. Every GP implementation should probably use them.
mandatory_gp_args = [
  get_option_specs('hp_tune_criterion', False, 'ml',
                   'Which criterion to use when tuning hyper-parameters.'),
  get_option_specs('hp_tune_opt', False, 'direct',
                   'Which optimiser to use when maximising the tuning criterion.'),
  get_option_specs('hp_tune_max_evals', False, -1,
                   'How many evaluations to use when maximising the tuning criterion.'),
  ]


def _check_feature_label_lengths_and_format(X, Y):
  """ Checks if the length of X and Y are the same. """
  if X.shape[0] != len(Y):
    raise ValueError('Size of X (' + str(X.shape) + ') and Y (' +
      str(Y.shape) + ') do not match.')
  if len(X.shape) != 2 or len(Y.shape) != 1:
    raise ValueError('X should be an nxd matrix and Y should be an n-vector.' +
      'Given shapes of X, Y are: ', str(X.shape) + ', ' + str(Y.shape))


class GP(object):
  '''
  Base class for Gaussian processes.
  '''
  def __init__(self, X, Y, kernel, mean_func, noise_var, build_posterior=True,
               reporter=None):
    """ Constructor. """
    super(GP, self).__init__()
    _check_feature_label_lengths_and_format(X, Y)
    self.X = X
    self.Y = Y
    self.kernel = kernel
    self.mean_func = mean_func
    self.noise_var = noise_var
    self.reporter = reporter
    # Some derived attribues.
    self.num_tr_data = len(self.Y)
    self.input_dim = self.X.shape[1]
    # Initialise other attributes we will need.
    self.L = None
    self.alpha = None
    # Build posterior if necessary
    if build_posterior:
      self.build_posterior()

  def _write_message(self, msg):
    """ Writes a message via the reporter or the std out. """
    if self.reporter:
      self.reporter.write(msg)
    else:
      sys.stdout.write(msg)

  def add_data(self, X_new, Y_new, rebuild=True):
    """ Adds new data to the GP. If rebuild is true it rebuilds the posterior. """
    _check_feature_label_lengths_and_format(X_new, Y_new)
    self.X = np.vstack((self.X, X_new))
    self.Y = np.append(self.Y, Y_new)
    self.num_tr_data = len(self.Y)
    if rebuild:
      self.build_posterior()

  def build_posterior(self):
    """ Builds the posterior GP by computing the mean and covariance. """
    prior_covar = self.kernel(self.X, self.X) + self.noise_var * np.eye(self.num_tr_data)
    Y_centred = self.Y - self.mean_func(self.X)
    self.L = stable_cholesky(prior_covar)
    self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y_centred))

  def eval(self, X_test, uncert_form='none'):
    """ Evaluates the GP on X_test. If uncert_form is
          covar: returns the entire covariance on X_test (nxn matrix)
          std: returns the standard deviations on the test set (n vector)
          none: returns nothing (default).
    """
    # First check for uncert_form
    if not uncert_form in ['none', 'covar', 'std']:
      raise ValueError('uncert_form should be one of none, std or covar.')
    # Compute the posterior mean.
    test_mean = self.mean_func(X_test)
    K_tetr = self.kernel(X_test, self.X)
    pred_mean = test_mean + K_tetr.dot(self.alpha)
    # Compute the posterior variance or standard deviation as required.
    if uncert_form == 'none':
      uncert = None
    else:
      K_tete = self.kernel(X_test, X_test)
      V = np.linalg.solve(self.L, K_tetr.T)
      post_covar = K_tete - V.T.dot(V)
      if uncert_form == 'covar':
        uncert = post_covar
      elif uncert_form == 'std':
        uncert = np.sqrt(np.diag(post_covar))
      else:
        raise ValueError('uncert_form should be none, covar or std.')
    return (pred_mean, uncert)

  def eval_with_hallucinated_observations(self, X_test, X_halluc, uncert_form='none'):
    """ Evaluates the GP with additional hallucinated observations in the
        kernel matrix. """
    pred_mean, _ = self.eval(X_test, uncert_form='none') # Just compute the means.
    if uncert_form == 'none':
      uncert = None
    else:
      # Computed the augmented kernel matrix and its cholesky decomposition.
      X_aug = np.concatenate((self.X, X_halluc), axis=0)
      num_aug_data = len(X_aug)
      aug_prior_covar = (self.kernel(X_aug, X_aug) +
                         self.noise_var * np.eye(num_aug_data))
      aug_L = stable_cholesky(aug_prior_covar)
      # Augmented kernel matrices for the test data
      aug_K_tete = self.kernel(X_test, X_test)
      aug_K_tetr = self.kernel(X_test, X_aug)
      aug_V = np.linalg.solve(aug_L, aug_K_tetr.T)
      aug_post_covar = aug_K_tete - aug_V.T.dot(aug_V)
      if uncert_form == 'covar':
        uncert = aug_post_covar
      elif uncert_form == 'std':
        uncert = np.sqrt(np.diag(aug_post_covar))
      else:
        raise ValueError('uncert_form should be none, covar or std.')
    return (pred_mean, uncert)

  def compute_log_marginal_likelihood(self):
    """ Computes the log marginal likelihood. """
    Y_centred = self.Y - self.mean_func(self.X)
    ret = -0.5 * Y_centred.T.dot(self.alpha) - (np.log(np.diag(self.L))).sum() \
          - 0.5 * self.num_tr_data * np.log(2*np.pi)
    return ret

  def __str__(self):
    """ Returns a string representation of the GP. """
    return '%s, eta2: %0.4f (n=%d)'%(self._child_str(), self.noise_var, len(self.Y))

  def _child_str(self):
    """ String representation for child GP. """
    raise NotImplementedError('Implement in child class. !')

  def draw_samples(self, num_samples, X_test=None, mean_vals=None, covar=None):
    """ Draws num_samples samples at returns their values at X_test. """
    if X_test is not None:
      mean_vals, covar = self.eval(X_test, 'covar')
    return draw_gaussian_samples(num_samples, mean_vals, covar)

  def draw_samples_with_hallucinated_observations(self, num_samples, X_test,
                                                  X_halluc):
    """ Draws samples with hallucinated observations. """
    mean_vals, aug_covar = self.eval_with_hallucinated_observations(X_test,
                         X_halluc, uncert_form='covar')
    return draw_gaussian_samples(num_samples, mean_vals, aug_covar)

  def visualise(self, file_name=None, boundary=None, true_func=None,
                num_samples=20, conf_width=3):
    """ Visualises the GP. """
    # pylint: disable=unused-variable
    # pylint: disable=too-many-locals
    if self.input_dim != 1:
      self._write_message('Cannot visualise in greater than 1 dimension.\n')
    else:
      import matplotlib.pyplot as plt
      fig = plt.figure()
      N = 400
      leg_handles = []
      leg_labels = []
      if not boundary:
        boundary = [self.X.min(), self.X.max()]
      grid = np.linspace(boundary[0], boundary[1], N).reshape((N, 1))
      (pred_vals, pred_stds) = self.eval(grid, 'std')
      # Shade a high confidence region
      conf_band_up = pred_vals + conf_width * pred_stds
      conf_band_down = pred_vals - conf_width * pred_stds
      leg_conf = plt.fill_between(grid.ravel(), conf_band_up, conf_band_down,
                                  color=[0.9, 0.9, 0.9])
      # Plot the samples
      gp_samples = self.draw_samples(num_samples, grid)
      plt.plot(grid, gp_samples.T, '--', linewidth=0.5)
      # plot the true function if available.
      if true_func:
        leg_true = plt.plot(grid, true_func(grid), 'b--', linewidth=3,
                            label='true function')
        leg_handles.append(leg_true)
      # Plot the posterior mean
      leg_post_mean = plt.plot(grid, pred_vals, 'k-', linewidth=4,
                               label='post mean')
      # Finally plot the training data.
      leg_data = plt.plot(self.X, self.Y, 'kx', mew=4, markersize=10,
                               label='data')
      # TODO: create a legend.
      # Finally either plot or show the figure
      if file_name is None:
        plt.show()
      else:
        fig.savefig(file_name)


class GPFitter(object):
  """
    Class for fitting Gaussian processes.
  """
  # pylint: disable=attribute-defined-outside-init
  # pylint: disable=abstract-class-not-used

  def __init__(self, options, reporter='default'):
    """ Constructor. """
    super(GPFitter, self).__init__()
    self.reporter = get_reporter(reporter)
    if isinstance(options, list):
      options = load_options(options, 'GP', reporter=self.reporter)
    self.options = options
    self._set_up()

  def _set_up(self):
    """ Sets up a bunch of ancillary parameters. """
    # The following hyper-parameters need to be set mandatorily in _child_setup.
    self.hp_bounds = None # The bounds for each hyper parameter should be a num_hps x 2
                          # array where the 1st/2nd columns are the lowe/upper bounds.
    # Set up hyper-parameters for the child.
    self._child_set_up()
    self.hp_bounds = np.array(self.hp_bounds)
    # Some post child set up
    self.num_hps = len(self.hp_bounds) # The number of hyper parameters
    # The optimiser for the hyper parameters
    if self.options.hp_tune_opt == 'direct':
      self._direct_set_up()
    else:
      raise ValueError('hp_tune_opt should be direct.')

  def _child_set_up(self):
    """ Here you should set up parameters for the child, such as the bounds for the
        optimiser etc. """
    raise NotImplementedError('Implement _child_set_up in a child method.')

  def _direct_set_up(self):
    """ Sets up optimiser for direct. """
    # define the following internal function to abstract things out more.
    def _direct_wrap(*args):
      """ A wrapper so as to only return the optimal value. """
      _, opt_pt, _ = direct_ft_maximise(*args)
      return opt_pt
    # Set some parameters
    lower_bounds = self.hp_bounds[:, 0]
    upper_bounds = self.hp_bounds[:, 1]
    if (hasattr(self.options, 'hp_tune_max_evals') and
        self.options.hp_tune_max_evals is not None and
        self.options.hp_tune_max_evals > 0):
      hp_tune_max_evals = self.options.hp_tune_max_evals
    else:
      hp_tune_max_evals = min(1e5, max(300, self.num_hps * 30))
    # Set hp_optimise
    self.hp_optimise = lambda obj: _direct_wrap(obj,
      lower_bounds, upper_bounds, hp_tune_max_evals)

  def _build_gp(self, gp_hyperparams):
    """ A method which builds a GP from the given gp_hyperparameters. It calls
        _child_build_gp after running some checks. """
    # Check the length of the hyper-parameters
    if self.num_hps != len(gp_hyperparams):
      raise ValueError('gp_hyperparams should be of length %d. Given length: %d.'%(
        self.num_hps, len(gp_hyperparams)))
    return self._child_build_gp(gp_hyperparams)

  def _child_build_gp(self, gp_hyperparams):
    """ A method which builds the child GP from the given gp_hyperparameters. Should be
        implemented in a child method. """
    raise NotImplementedError('Implement _build_gp in a child method.')

  def _tuning_objective(self, gp_hyperparams):
    """ This function computes the tuning objective (such as the marginal likelihood)
        which is to be maximised in fit_gp. """
    built_gp = self._build_gp(gp_hyperparams)
    if self.options.hp_tune_criterion in ['ml', 'marginal_likelihood']:
      ret = built_gp.compute_log_marginal_likelihood()
    elif self.options.hp_tune_criterion in ['cv', 'cross_validation']:
      raise NotImplementedError('Yet to implement cross validation based hp-tuning.')
    else:
      raise ValueError('hp_tune_criterion should be either ml or cv')
    return ret

  def fit_gp(self):
    """ Fits a GP according to the tuning criterion. Returns the best GP along with the
        hyper-parameters. """
    opt_hps = self.hp_optimise(self._tuning_objective)
    opt_gp = self._build_gp(opt_hps)
    return opt_gp, opt_hps

