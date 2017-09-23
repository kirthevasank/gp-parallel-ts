"""
  Implements various kernels.
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import

import numpy as np

# Local imports
from utils.general_utils import dist_squared


class Kernel(object):
  """ A kernel class. """

  def __init__(self):
    """ Constructor. """
    super(Kernel, self).__init__()
    self.hyperparams = {}

  def __call__(self, X1, X2=None):
    """ Evaluates the kernel by calling evaluate. """
    return self.evaluate(X1, X2)

  def evaluate(self, X1, X2=None):
    """ Evaluates kernel values between X1 and X2 and returns an n1xn2 kernel matrix.
        This is a wrapper for _child_evaluate.
    """
    X2 = X1 if X2 is None else X2
    return self._child_evaluate(X1, X2)

  def _child_evaluate(self, X1, X2):
    """ Evaluates kernel values between X1 and X2 and returns an n1xn2 kernel matrix.
        This is to be implemented in a child kernel.
    """
    raise NotImplementedError('Implement in a child class.')

  def set_hyperparams(self, **kwargs):
    """ Set hyperparameters here. """
    self.hyperparams = kwargs

  def add_hyperparams(self, **kwargs):
    """ Set additional hyperparameters here. """
    for key, value in kwargs.iteritems():
      self.hyperparams[key] = value

  def get_effective_norm(self, X, order=None, *args, **kwargs):
    """ Gets the effective norm scaled by bandwidths. """
    raise NotImplementedError('Implement in a child class.')

  def compute_std_slack(self, X1, X2):
    """ Computes a bound on the maximum standard deviation between X1 and X2. """
    raise NotImplementedError('Implement in a child class.')

  def change_smoothness(self, factor):
    """ Decreases smoothness by the factor given. """
    raise NotImplementedError('Implement in a child class.')


class SEKernel(Kernel):
  """ Squared exponential kernel. """

  def __init__(self, dim, scale=None, dim_bandwidths=None):
    """ Constructor. dim is the dimension. """
    super(SEKernel, self).__init__()
    self.dim = dim
    self.set_se_hyperparams(scale, dim_bandwidths)

  def set_dim_bandwidths(self, dim_bandwidths):
    """ Sets the bandwidth for each dimension. """
    if dim_bandwidths is not None:
      if len(dim_bandwidths) != self.dim:
        raise ValueError('Dimension of dim_bandwidths should be the same as dimension.')
      dim_bandwidths = np.array(dim_bandwidths)
    self.add_hyperparams(dim_bandwidths=dim_bandwidths)

  def set_single_bandwidth(self, bandwidth):
    """ Sets the bandwidht of all dimensions to be the same value. """
    dim_bandwidths = None if bandwidth is None else [bandwidth] * self.dim
    self.set_dim_bandwidths(dim_bandwidths)

  def set_scale(self, scale):
    """ Sets the scale parameter for the kernel. """
    self.add_hyperparams(scale=scale)

  def set_se_hyperparams(self, scale, dim_bandwidths):
    """ Sets both the scale and the dimension bandwidths for the SE kernel. """
    self.set_scale(scale)
    if hasattr(dim_bandwidths, '__len__'):
      self.set_dim_bandwidths(dim_bandwidths)
    else:
      self.set_single_bandwidth(dim_bandwidths)

  def _child_evaluate(self, X1, X2):
    """ Evaluates the SE kernel between X1 and X2 and returns the gram matrix. """
    scaled_X1 = self.get_scaled_repr(X1)
    scaled_X2 = self.get_scaled_repr(X2)
    dist_sq = dist_squared(scaled_X1, scaled_X2)
    K = self.hyperparams['scale'] * np.exp(-dist_sq/2)
    return K

  def get_scaled_repr(self, X):
    """ Returns the scaled version of an input by the bandwidths. """
    return X/self.hyperparams['dim_bandwidths']

  def get_effective_norm(self, X, order=None, is_single=True):
    """ Gets the effective norm. That is the norm of X scaled by bandwidths. """
    # pylint: disable=arguments-differ
    scaled_X = self.get_scaled_repr(X)
    if is_single:
      return np.linalg.norm(scaled_X, ord=order)
    else:
      return np.array([np.linalg.norm(sx, ord=order) for sx in scaled_X])

  def compute_std_slack(self, X1, X2):
    """ Computes a bound on the maximum standard deviation diff between X1 and X2. """
    k_12 = np.array([float(self.evaluate(X1[i].reshape(1, -1), X2[i].reshape(1, -1)))
                     for i in range(len(X1))])
    return np.sqrt(self.hyperparams['scale'] - k_12)

  def change_smoothness(self, factor):
    """ Decreases smoothness by the given factor. """
    self.hyperparams['dim_bandwidths'] *= factor


class PolyKernel(Kernel):
  """ The polynomial kernel. """
  # pylint: disable=abstract-method

  def __init__(self, dim, order, scale, dim_scalings=None):
    """ Constructor. """
    super(PolyKernel, self).__init__()
    self.dim = dim
    self.set_poly_hyperparams(order, scale, dim_scalings)

  def set_order(self, order):
    """ Sets the order of the polynomial. """
    self.add_hyperparams(order=order)

  def set_scale(self, scale):
    """ Sets the scale of the kernel. """
    self.add_hyperparams(scale=scale)

  def set_dim_scalings(self, dim_scalings):
    """ Sets the scaling for each dimension in the polynomial kernel. This will be a
        dim+1 dimensional vector.
    """
    if dim_scalings is not None:
      if len(dim_scalings) != self.dim:
        raise ValueError('Dimension of dim_scalings should be dim + 1.')
      dim_scalings = np.array(dim_scalings)
    self.add_hyperparams(dim_scalings=dim_scalings)

  def set_single_scaling(self, scaling):
    """ Sets the same scaling for all dimensions. """
    if scaling is None:
      self.set_dim_scalings(None)
    else:
      self.set_dim_scalings([scaling] * self.dim)

  def set_poly_hyperparams(self, order, scale, dim_scalings):
    """Sets the hyper parameters. """
    self.set_order(order)
    self.set_scale(scale)
    if hasattr(dim_scalings, '__len__'):
      self.set_dim_scalings(dim_scalings)
    else:
      self.set_single_scaling(dim_scalings)

  def _child_evaluate(self, X1, X2):
    """ Evaluates the polynomial kernel and returns and the gram matrix. """
    X1 = X1 * self.hyperparams['dim_scalings']
    X2 = X2 * self.hyperparams['dim_scalings']
    K = self.hyperparams['scale'] * ((X1.dot(X2.T) + 1)**self.hyperparams['order'])
    return K


class UnscaledPolyKernel(Kernel):
  """ The polynomial kernel. """
  # pylint: disable=abstract-method

  def __init__(self, dim, order, dim_scalings=None):
    """ Constructor. """
    super(UnscaledPolyKernel, self).__init__()
    self.dim = dim
    self.set_unscaled_poly_hyperparams(order, dim_scalings)

  def set_order(self, order):
    """ Sets the order of the polynomial. """
    self.add_hyperparams(order=order)

  def set_dim_scalings(self, dim_scalings):
    """ Sets the scaling for each dimension in the polynomial kernel. This will be a
        dim+1 dimensional vector.
    """
    if dim_scalings is not None:
      if len(dim_scalings) != self.dim + 1:
        raise ValueError('Dimension of dim_scalings should be dim + 1.')
      dim_scalings = np.array(dim_scalings)
    self.add_hyperparams(dim_scalings=dim_scalings)

  def set_single_scaling(self, scaling):
    """ Sets the same scaling for all dimensions. """
    if scaling is None:
      self.set_dim_scalings(None)
    else:
      self.set_dim_scalings([scaling] * (self.dim + 1))

  def set_unscaled_poly_hyperparams(self, order, dim_scalings):
    """Sets the hyper parameters. """
    self.set_order(order)
    if hasattr(dim_scalings, '__len__'):
      self.set_dim_scalings(dim_scalings)
    else:
      self.set_single_scaling(dim_scalings)

  def _child_evaluate(self, X1, X2):
    """ Evaluates the polynomial kernel and returns and the gram matrix. """
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    X1 = np.concatenate((np.ones((n1, 1)), X1), axis=1) * self.hyperparams['dim_scalings']
    X2 = np.concatenate((np.ones((n2, 1)), X2), axis=1) * self.hyperparams['dim_scalings']
    K = (X1.dot(X2.T))**self.hyperparams['order']
    return K


class CoordinateProductKernel(Kernel):
  """ Implements a coordinatewise product kernel. """
  # pylint: disable=abstract-method

  def __init__(self, dim, scale, kernel_list=None, coordinate_list=None):
    """ Constructor.
        kernel_list is a list of n Kernel objects. coordinate_list is a list of n lists
        each indicating the coordinates each kernel in kernel_list should be applied to.
    """
    super(CoordinateProductKernel, self).__init__()
    self.dim = dim
    self.scale = scale
    self.kernel_list = kernel_list
    self.coordinate_list = coordinate_list

  def set_kernel_list(self, kernel_list):
    """ Sets a new list of kernels. """
    self.kernel_list = kernel_list

  def set_new_kernel(self, kernel_idx, new_kernel):
    """ Sets new_kernel to kernel_list[kernel_idx]. """
    self.kernel_list[kernel_idx] = new_kernel

  def set_kernel_hyperparams(self, kernel_idx, **kwargs):
    """ Sets the hyper-parameters for kernel_list[kernel_idx]. """
    self.kernel_list[kernel_idx].set_hyperparams(**kwargs)

  def _child_evaluate(self, X1, X2):
    """ Evaluates the combined kernel. """
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = self.scale * np.ones((n1, n2))
    for idx, kernel in enumerate(self.kernel_list):
      X1_sel = X1[:, self.coordinate_list[idx]]
      X2_sel = X2[:, self.coordinate_list[idx]]
      K *= kernel(X1_sel, X2_sel)
    return K

