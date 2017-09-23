"""
  A collection of utilities for ancillary purposes.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import

import numpy as np


# Print lists as strings
def get_rounded_list(float_list, round_to_decimals=3):
  """ Rounds the list and returns. """
  ret = np.array(float_list).round(round_to_decimals)
  if isinstance(float_list, list):
    ret = list(ret)
  return ret

def get_list_as_str(list_of_objs):
  """ Returns the list as a string. """
  return '[' + ' '.join([str(x) for x in list_of_objs]) + ']'

def get_list_of_floats_as_str(float_list, round_to_decimals=3):
  """ Rounds the list and returns a string representation. """
  float_list = get_rounded_list(float_list, round_to_decimals)
  return get_list_as_str(float_list)


# Some other utilities
def is_non_decreasing_sequence(vals):
  """ Returns true if vals is nondecreasing. """
  for i in range(len(vals)-1):
    if vals[i] > vals[i+1]:
      return False
  return True


# Some plotting utilities.
def plot_2d_function(func, bounds, x_label='x', y_label='y', title=None):
  """ Plots a 2D function in bounds. """
  # pylint: disable=unused-variable
  dim_grid_size = 20
  x_grid = np.linspace(bounds[0][0], bounds[0][1], dim_grid_size)
  y_grid = np.linspace(bounds[1][0], bounds[1][1], dim_grid_size)
  XX, YY = np.meshgrid(x_grid, y_grid)
  f_vals = func(XX.ravel(), YY.ravel())
  FF = f_vals.reshape(dim_grid_size, dim_grid_size)
  # Create plot
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot_surface(XX, YY, FF)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  if title is not None:
    plt.title(title)
  return fig, ax, plt

