"""
  A collection of wrappers for optimisng a function.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=superfluous-parens

from argparse import Namespace
from datetime import datetime
import os
import numpy as np
# Local imports
try:
  import direct_fortran.direct as direct_ft_wrap
except ImportError:
  print('Could not import fortran direct library')
  direct_ft_wrap = None
from general_utils import map_to_bounds

def random_maximise(obj, bounds, max_evals, vectorised=True):
  """ Optimises a function by randomly sampling and choosing its maximum. """
  dim = len(bounds)
  rand_pts = map_to_bounds(np.random.random((int(max_evals), dim)), bounds)
  if vectorised:
    obj_vals = obj(rand_pts)
  else:
    obj_vals = np.array([obj(x) for x in rand_pts])
  max_idx = obj_vals.argmax()
  max_val = obj_vals[max_idx]
  max_pt = rand_pts[max_idx]
#   import pdb; pdb.set_trace()
  return max_val, max_pt


# DIRECT #########################################################################
# Some constants
_MAX_DIRECT_FN_EVALS = 2.6e6 # otherwise the fortran software complains

def direct_ft_minimise(obj, lower_bounds, upper_bounds, max_evals,
                       eps=1e-5,
                       return_history=False,
                       max_iterations=None,
                       alg_method=0,
                       fglobal=-1e100,
                       fglper=0.01,
                       volper=-1.0,
                       sigmaper=-1.0,
                       log_file_name='',
                      ):
  """
    A wrapper for the fortran implementation. The four mandatory arguments are self
    explanatory. If return_history is True it also returns the history of evaluations.
    max_iterations is the maximum number of iterations of the direct algorithm.
    I am not sure what the remaining arguments are for.
  """
  # pylint: disable=too-many-locals
  # pylint: disable=too-many-arguments

  # Preliminaries.
  max_evals = min(_MAX_DIRECT_FN_EVALS, max_evals) # otherwise the fortran sw complains.
  max_iterations = max_evals if max_iterations is None else max_iterations
  lower_bounds = np.array(lower_bounds, dtype=np.float64)
  upper_bounds = np.array(upper_bounds, dtype=np.float64)
  if len(lower_bounds) != len(upper_bounds):
    raise ValueError('The dimensionality of the lower and upper bounds should match.')

  # Create a wrapper to comply with the fortran requirements.
  def _objective_wrap(x, *_):
    """ A wrapper to comply with the fortran requirements. """
    return (obj(x), 0)

  # Some dummy data to comply with the fortran requirements.
  iidata = np.ones(0, dtype=np.int32)
  ddata = np.ones(0, dtype=np.float64)
  cdata = np.ones([0, 40], dtype=np.uint8)
  # Call the function.
  min_pt, min_val, _ = direct_ft_wrap.direct(_objective_wrap,
                                             eps,
                                             max_evals,
                                             max_iterations,
                                             lower_bounds,
                                             upper_bounds,
                                             alg_method,
                                             log_file_name,
                                             fglobal,
                                             fglper,
                                             volper,
                                             sigmaper,
                                             iidata,
                                             ddata,
                                             cdata
                                            )
  if return_history:
    # TODO: implement this. Read it off the log file.
    pass
  else:
    history = None
  # return
  return min_val, min_pt, history


def direct_ft_maximise(obj, lower_bounds, upper_bounds, max_evals, **kwargs):
  """
    A wrapper for maximising a function which calls direct_ft_minimise. See arguments
    under direct_ft_minimise for more details.
  """
  min_obj = lambda x: -obj(x)
  min_val, max_pt, history = direct_ft_minimise(min_obj, lower_bounds, upper_bounds,
                                                max_evals, **kwargs)
  max_val = - min_val
  # TODO: Fix history here.
  return max_val, max_pt, history


def direct_ft_maximise_with_history(obj, lower_bounds, upper_bounds, max_evals, **kwargs):
  """
    A wrapper for maximising a function which calls direct_ft_minimise. But also
    returns the history.
  """
  log_file_name = 'direct_log_%s'%(datetime.now().strftime('%m%d-%H%M%S'))
  max_val, max_pt, _ = direct_ft_maximise(obj, lower_bounds, upper_bounds, max_evals,
                                          log_file_name=log_file_name, **kwargs)
  history = get_history_from_direct_log(log_file_name)
  # delete file
  os.remove(log_file_name)
  return max_val, max_pt, history

def get_history_from_direct_log(log_file_name):
  """ Returns the history from the direct log file. """
  saved_iterations = [0]
  saved_max_vals = [-np.inf]
  phase = 'boiler'
  log_file_handle = open(log_file_name, 'r')
  for line in log_file_handle.readlines():
    words = line.strip().split()
    if phase == 'boiler':
      if words[0] == 'Iteration':
        phase = 'results'
    elif phase == 'results':
      if len(words) == 3 and words[0].isdigit():
        saved_iterations.append(int(words[1]))
        saved_max_vals.append(-float(words[2]))
      else:
        phase = 'final'
    elif phase == 'final':
      if words[0] == 'Final':
        saved_max_vals.append(max(-float(words[-1]), saved_max_vals[-1]))
        # doing max as the fortran library rounds off the last result for some reason.
      if words[0] == 'Number':
        saved_iterations.append(int(words[-1]))
  # Now fill in the rest of the history.
  curr_opt_vals = np.zeros((saved_iterations[-1]), dtype=np.float64)
  for i in range(len(saved_iterations)-1):
    curr_opt_vals[saved_iterations[i]:saved_iterations[i+1]] = saved_max_vals[i]
  curr_opt_vals[-1] = saved_max_vals[-1]
  return Namespace(curr_opt_vals=curr_opt_vals)


def direct_maximise_from_mfof(mfof, max_evals, **kwargs):
  """ Direct maximise from an mfof object. """
  obj = lambda x: mfof.eval_single(mfof.opt_fidel, x)
  lower_bounds = mfof.domain_bounds[:, 0]
  upper_bounds = mfof.domain_bounds[:, 1]
  return direct_ft_maximise_with_history(obj, lower_bounds, upper_bounds, max_evals,
                                         **kwargs)

# DIRECT end ######################################################################

