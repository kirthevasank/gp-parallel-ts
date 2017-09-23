"""
  A module which implements different instances of GPs.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=abstract-class-not-used
# pylint: disable=super-on-old-class


import numpy as np
# Local imports
import gp_core
import kernel
from utils.option_handler import get_option_specs


# Some basic parameters for simple GPs.
basic_gp_args = [
  get_option_specs('kernel_type', False, 'se',
                   'Specify type of kernel. Should be se or poly'),
  get_option_specs('mean_func_type', False, 'median',
                   ('Specify the type of mean function. Should be mean, median, const ',
                    'or zero. If const, specifcy value in mean-func-const.')),
  get_option_specs('mean_func_const', False, 0.0,
                   'The constant value to use if mean_func_type is const.'),
  get_option_specs('noise_var_type', False, 'tune',
                   ('Specify how to obtain the noise variance. Should be tune, label ',
                    'or value. Specify appropriate value in noise_var_label or',
                     'noise_var_value')),
  get_option_specs('noise_var_label', False, 0.05,
                   'The fraction of label variance to use as noise variance.'),
  get_option_specs('noise_var_value', False, 0.1,
                   'The (absolute) value to use as noise variance.'),
  ]
# Parameters for the SE kernel.
se_gp_args = [
  get_option_specs('use_same_bandwidth', False, False,
                   'If true uses same bandwidth on all dimensions. Default is False.'),
]
# Parameters for the Polynomial kernel.
poly_gp_args = [
  get_option_specs('use_same_scalings', False, False,
                   'If true uses same scalings on all dimensions. Default is False.'),
  get_option_specs('poly_order', False, 1,
                   'Order of the polynomial to be used. Default is 1 (linear kernel).')
]
# All parameters
all_simple_gp_args = gp_core.mandatory_gp_args + basic_gp_args + se_gp_args + poly_gp_args


class SEGP(gp_core.GP):
  """ An implementation of a GP using a SE kernel. """
  def __init__(self, X, Y, ke_scale, ke_dim_bandwidths, mean_func, noise_var,
               *args, **kwargs):
    """ Constructor. ke_scale and ke_dim_bandwidths are the kernel hyper-parameters.
        ke_dim_bandwidths can be a vector of length dim or a scalar (in which case we
        will use the same bandwidth for all dimensions).
    """
    se_kernel = kernel.SEKernel(dim=X.shape[1], scale=ke_scale,
                                dim_bandwidths=ke_dim_bandwidths)
    super(SEGP, self).__init__(X, Y, se_kernel, mean_func, noise_var, *args, **kwargs)

  def _child_str(self):
    """ Description of the child GP. """
    if self.kernel.dim > 6:
      bw_str = 'avg-bw: %0.4f'%(self.kernel.hyperparams['dim_bandwidths'].mean())
    else:
      bw_str = 'bws:[' + ' '.join(['%0.2f'%(dbw) for dbw in
                                   self.kernel.hyperparams['dim_bandwidths']]) + ']'
    scale_str = 'sc:%0.4f'%(self.kernel.hyperparams['scale'])
    return 'SE: ' + scale_str + ' ' + bw_str

class PolyGP(gp_core.GP):
  """ An implementation of a GP using a polynomial kernel. """
  def __init__(self, X, Y, ke_order, ke_dim_scalings, mean_func, noise_var,
               *args, **kwargs):
    """ Constructor. ke_order and ke_dim_scalings are the kernel hyper-parameters.
        see kernel.PolyKernel for more info.
    """
    poly_kernel = kernel.PolyKernel(dim=X.shape[1], order=ke_order,
                                    dim_scalings=ke_dim_scalings)
    super(PolyGP, self).__init__(X, Y, poly_kernel, mean_func, noise_var, *args, **kwargs)

  def _child_str(self):
    """ Description of the child GP. """
    raise NotImplementedError('Implement this!')


class SimpleGPFitter(gp_core.GPFitter):
  """ A concrete implementation to fit a simple GP. Use this as an example."""
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, X, Y, options=None, *args, **kwargs):
    """ Constructor.
        options should either be a Namespace, a list or None"""
    # Just call the super constructor.
    self.X = X
    self.Y = Y
    if options is None:
      options = all_simple_gp_args
    super(SimpleGPFitter, self).__init__(options, *args, **kwargs)

  def _child_set_up(self):
    """ Sets parameters for GPFitter. """
    # Check args - so that we don't have to keep doing this all the time
    if not self.options.kernel_type in ['se', 'poly']:
      raise ValueError('Unknown kernel_type. Should be either se or poly.')
    if not self.options.noise_var_type in ['tune', 'label', 'value']:
      raise ValueError('Unknown noise_var_type. Should be either tune, label or value.')
    if not self.options.mean_func_type in ['mean', 'median', 'const', 'zero']:
      raise ValueError('Unknown mean_func_type. Should be one of mean/median/const/zero.')
    # Set some parameters we will be using often.
    self.Y_var = self.Y.std()**2
    self.input_dim = self.X.shape[1]
    # Bounds for the hyper-parameters
    self.hp_bounds = []
    # Noise variance
    if self.options.noise_var_type == 'tune':
      self.noise_var_log_bounds = [np.log(0.005 * self.Y_var), np.log(0.2 * self.Y_var)]
      self.hp_bounds.append(self.noise_var_log_bounds)
    # Kernel parameters
    if self.options.kernel_type == 'se':
      self._se_kernel_set_up()
    elif self.options.kernel_type == 'poly':
      self._poly_kernel_set_up()

  def _se_kernel_set_up(self):
    """ Set up for the SE kernel. """
    # Scale
    self.scale_log_bounds = [np.log(0.1 * self.Y_var), np.log(10 * self.Y_var)]
    # Bandwidths
    X_std_norm = np.linalg.norm(self.X.std(axis=0))
    single_bandwidth_log_bounds = [np.log(0.01 * X_std_norm), np.log(10 * X_std_norm)]
    if self.options.use_same_bandwidth:
      self.bandwidth_log_bounds = [single_bandwidth_log_bounds]
    else:
      self.bandwidth_log_bounds = [single_bandwidth_log_bounds] * self.input_dim
    self.hp_bounds += [self.scale_log_bounds] + self.bandwidth_log_bounds

  def _poly_kernel_set_up(self):
    """ Set up for the Poly kernel. """
    # TODO: Implement poly kernel set up.
    raise NotImplementedError('Not implemented Poly kernel yet.')

  def _child_build_gp(self, gp_hyperparams):
    """ Builds the GP. """
    # Noise variance ####################################
    if self.options.noise_var_type == 'tune':
      noise_var = np.exp(gp_hyperparams[0])
      gp_hyperparams = gp_hyperparams[1:]
    elif self.options.noise_var_type == 'label':
      noise_var = self.options.noise_var_label * (self.Y.std()**2)
    else:
      noise_var = self.options.noise_var_value
    # Mean function #####################################
    if hasattr(self.options, 'mean_func') and self.options.mean_func is not None:
      mean_func = self.options.mean_func
    else:
      if self.options.mean_func_type == 'mean':
        mean_func_const_value = self.Y.mean()
      elif self.options.mean_func_type == 'median':
        mean_func_const_value = np.median(self.Y)
      elif self.options.mean_func_type == 'const':
        mean_func_const_value = self.options.mean_func_const
      else:
        mean_func_const_value = 0
      mean_func = lambda x: np.array([mean_func_const_value] * len(x))
    # Build kernels and return ##########################
    if self.options.kernel_type == 'se':
      return self._build_se_gp(noise_var, mean_func, gp_hyperparams)
    elif self.options.kernel_type == 'poly':
      return self._build_poly_gp(noise_var, mean_func, gp_hyperparams)

  def _build_se_gp(self, noise_var, mean_func, gp_hyperparams):
    """ Builds the GP if using an SE kernel. """
    # Kernel parameters
    ke_scale = np.exp(gp_hyperparams[0])
    ke_dim_bandwidths = (
      [np.exp(gp_hyperparams[1])] * self.input_dim if self.options.use_same_bandwidth
      else np.exp(gp_hyperparams[1:]))
    # return an squared exponential GP.
    return SEGP(self.X, self.Y, ke_scale, ke_dim_bandwidths, mean_func, noise_var,
                build_posterior=True, reporter=self.reporter)

  def _build_poly_gp(self, noise_var, mean_func, gp_hyperparams):
    """ Builds the GP if using a Poly kernel. """
    # TODO: Implement poly kernel build.
    raise NotImplementedError('Not implemented Poly kernel yet.')

