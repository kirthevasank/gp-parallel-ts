"""
  Harness for calling function.
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
from utils.general_utils import map_to_cube, map_to_bounds

class FunctionCaller(object):
  """ Function caller instance. """

  def __init__(self, func, dom_bounds, vectorised, opt_pt=None, opt_val=None,
               noise_type='none', noise_scale=None, descr=''):
    # NB: dom_bounds refers to the true bounds of the function being optimised. It is
    # stored under the private attribute _true_domain_bounds.
    # In contrast, the attribute domain_bounds is what is made visible outside. This might
    # be necessary in case you want the optimisation to occur only in the unit cube for
    # example.
    """
      Constructor.
        - func: takes argument x and returns function value.
        - dom_bounds: the bounds for optimisng.
        - vectorised: If true, means func can take multiple inputs and produce multiple
            outputs. If False, the functions can only take single inputs in 'column' form.
        - opt_pt, opt_val are the optimum point and value.
        - noise_type: What kind of noise to add to the function values. If 'none', will
            not add any noise. If 'gauss' will add gaussian noise.
        - noise_scale: The scale of the noise - usually the standard deviation of the
            noise.
    """
    self.func = func
    self._true_domain_bounds = np.array(dom_bounds)
    self.domain_dim = len(dom_bounds)
    self.domain_bounds = np.array([[0, 1]]*self.domain_dim)
    self.vectorised = vectorised
    self.noise_type = noise_type
    self.noise_scale = noise_scale
    self.descr = descr
    # Set up optimal point and optimal value.
    self.opt_pt_unnormalised = opt_pt
    if opt_pt is None:
      self.opt_pt_normalised = None
    else:
      self.opt_pt_normalised = self.get_normalised_coords(opt_pt)
    self.opt_val = opt_val

  # Wrappers for evaluating the function -------------------------------------------------
  def eval_single(self, x, data=None, normalised=True, noisy=True):
    """ Evaluates func at a single point x. If noisy is True and noise_type is not None
        then will add noise. If normalised is true, evaluates on normalised coords."""
    x_sent = x
    if normalised:
      x = self.get_unnormalised_coords(x)
    if not self.vectorised:
      func_val = float(self.func(x))
    else:
      X = np.array(x).reshape(1, self.domain_dim)
      func_val = float(self.func(X))
    if noisy and self.noise_type != 'none':
      ret = func_val + self.noise_single()
    else:
      ret = func_val
    if data is None:
      data = Namespace()
    data.point = x_sent # Include the query point in data.
    data.true_val = func_val
    return (ret, data) # The none is for ancillary data a child class might want to send.

  def eval_multiple(self, X, data=None, normalised=True, noisy=True):
    """ Evaluates func at multiple points. """
    # Check the dat namespace
    if data is None:
      data = Namespace()
    if normalised:
      X = self.get_unnormalised_coords(X)
    if self.vectorised:
      func_vals = self.func(X).ravel()
    else:
      ret = []
      for i in range(len(X)):
        ret_val, _ = self.eval_single(X[i, :])
        ret.append(ret_val)
      func_vals = np.array(ret)
    if noisy and self.noise_type != 'none':
      ret = func_vals + self.noise_multiple(len(X))
    else:
      ret = func_vals
    data.points = X # Include the query point in data.
    data.true_vals = func_vals
    return (ret, data) # The none is for ancillary data a child class might want to send.

  # Map to normalised coordinates and vice versa -----------------------------------------
  def get_normalised_coords(self, X):
    """ Maps points in the original space to the unit cube. """
    return map_to_cube(X, self._true_domain_bounds)

  def get_unnormalised_coords(self, X):
    """ Maps points in the unit cube to the orignal space. """
    return map_to_bounds(X, self._true_domain_bounds)

  # Wrappers for adding noise ------------------------------------------------------------
  def noise_single(self):
    """ Returns single noise value. """
    return float(self.noise_multiple(1))

  def noise_multiple(self, num_samples):
    """ Returns multiple noise values. """
    if self.noise_type == 'none':
      return np.zeros(num_samples)
    elif self.noise_type == 'gauss':
      return np.random.normal(scale=self.noise_scale, size=(num_samples))
    else:
      raise NotImplementedError('Only implemented Gaussian noise so far.')


# A function caller which adds a time variable in the data.
class FunctionCallerWithRandomTime(FunctionCaller):
  """ Function caller instance. """

  def __init__(self, func, domain_bounds, vectorised, opt_pt=None, opt_val=None,
               noise_type='none', noise_scale=None, descr='',
               time_distro='const', time_distro_params=None):
    """ Constructor. Read FunctionCaller for other arguments.
      time_distro: Can be const, uniform, halfnormal, exponential, pareto.
      time_distro_params: The parameters for the distribution. If None, will create RVs in
        such a way that the mean is always 1.
    """
    super(FunctionCallerWithRandomTime, self).__init__(func, domain_bounds,
      vectorised=vectorised, opt_pt=opt_pt, opt_val=opt_val, noise_type=noise_type,
      noise_scale=noise_scale)
    self.time_distro = time_distro
    self.time_distro_params = time_distro_params
    self._set_up_sampler()

  def _set_up_sampler(self):
    """ Sets up atrributes for the sampler. """
    if self.time_distro_params is not None:
      return
    self.time_distro_params = Namespace
    if self.time_distro == 'const':
      self.time_distro_params.const_val = 1
    elif self.time_distro == 'uniform':
      self.time_distro_params.lb = 0
      self.time_distro_params.ub = 2
    elif self.time_distro == 'halfnormal':
      self.time_distro_params.sigma = np.sqrt(np.pi/2)
    elif self.time_distro == 'exponential':
      self.time_distro_params.exp_mean = 1
    elif self.time_distro == 'pareto':
      self.time_distro_params.power = 5
    else:
      err_msg = 'Not Implemented time_distro=%s yet.'%(self.time_distro)
      raise NotImplementedError(err_msg)

  def sample_time_single(self):
    """ Samples a single noise point and returns. """
    return float(self.sample_time_multiple(1))

  def sample_time_multiple(self, num_samples):
    """ Sample with multiple. """
    if self.time_distro == 'const':
      return np.ones(num_samples)
    if self.time_distro == 'uniform':
      return self.time_distro_params.lb + np.random.rand(num_samples) * (
        self.time_distro_params.ub - self.time_distro_params.lb)
    elif self.time_distro == 'halfnormal':
      return np.abs(np.random.normal(scale=self.time_distro_params.sigma,
                                     size=num_samples))
    elif self.time_distro == 'exponential':
      return np.random.exponential(scale=1/float(self.time_distro_params.exp_mean),
                                   size=num_samples)
    elif self.time_distro == 'pareto':
      if not hasattr(self.time_distro_params, 'offset'):
        self.time_distro_params.offset = (
          (self.time_distro_params.power - 1) / float(self.time_distro_params.power))
      return self.time_distro_params.offset*(1 + np.random.pareto(
               a=self.time_distro_params.power, size=num_samples))

  # Finally override eval functions. -----------------------------------------------
  def eval_single(self, x, qinfo=None, normalised=True, noisy=True):
    """ Calls super eval_single and then adds time data. """
    ret, qinfo = super(FunctionCallerWithRandomTime, self).eval_single(x, qinfo,
                                                                      normalised, noisy)
    qinfo.eval_time = self.sample_time_single()
    return ret, qinfo

  def eval_multiple(self, X, qinfo=None, normalised=True, noisy=True):
    """ Calls super eval_multiple and then adds time qinfo. """
    ret, qinfo = super(FunctionCallerWithRandomTime, self).eval_multiple(X, qinfo,
                                                                        normalised, noisy)
    qinfo.eval_times = self.sample_time_multiple(len(X))
    return ret, qinfo

  def get_time_distro_as_str(self):
    """ Returns a strign representation of the time distribution. """
    if self.time_distro == 'pareto':
      ret = 'pareto(k=%d)'%(self.time_distro_params.power)
    else:
      ret = self.time_distro
    return ret


# A wrapper for function caller
# Get function caller from functions
def get_function_caller_from_function(func, domain_bounds, vectorised, time_distro=None,
  **kwargs):
  """ A wrapper which constructs and returns a desired function caller object given the
      options. """
  if time_distro is None:
    return FunctionCaller(func, domain_bounds, vectorised, **kwargs)
  else:
    return FunctionCallerWithRandomTime(func, domain_bounds, vectorised,
                                        time_distro=time_distro, **kwargs)

