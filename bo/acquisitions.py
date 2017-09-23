"""
  Acquisition functions for Bayesian Optimisation.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class
# pylint: disable=no-name-in-module
# pylint: disable=star-args

from argparse import Namespace
import numpy as np
from scipy.stats import norm as normal_distro

def _optimise_acquisition(acq_fn, acq_optimiser, anc_data):
  """ All methods will just call this. """
  return acq_optimiser(acq_fn, anc_data.max_evals)

# Thompson sampling ---------------------------------------------------------------
def asy_ts(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via TS in the asyuential setting. """
  gp_sample = lambda x: gp.draw_samples(1, X_test=x, mean_vals=None, covar=None).ravel()
  return _optimise_acquisition(gp_sample, acq_optimiser, anc_data)

def asy_hts(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via TS using hallucinated observaitons in the asynchronus
      setting. """
  halluc_pts = (np.empty((0, gp.input_dim)) if len(anc_data.evals_in_progress) == 0
                     else np.array(anc_data.evals_in_progress))
  gp_sample = lambda x: gp.draw_samples_with_hallucinated_observations(1, x,
                                                                       halluc_pts).ravel()
  return _optimise_acquisition(gp_sample, acq_optimiser, anc_data)

def syn_ts(num_workers, gp, acq_optimiser, anc_data, **kwargs):
  """ Returns a batch of recommendations via TS in the synchronous setting. """
  recommendations = []
  for _ in range(num_workers):
    rec_j = asy_ts(gp, acq_optimiser, anc_data, **kwargs)
    recommendations.append(rec_j)
  return recommendations

# UCB ------------------------------------------------------------------------------
def _get_ucb_beta_th(dim, time_step):
  """ Computes the beta t for UCB based methods. """
  return np.sqrt(5 * dim * np.log(2 * dim * time_step + 1))

def asy_ucb(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via UCB in the asyuential setting. """
  beta_th = _get_ucb_beta_th(gp.input_dim, anc_data.t)
  def _ucb_acq(x):
    """ Computes the GP-UCB acquisition. """
    mu, sigma = gp.eval(x, uncert_form='std')
    return mu + beta_th * sigma
  return _optimise_acquisition(_ucb_acq, acq_optimiser, anc_data)

def _halluc_ucb(gp, acq_optimiser, halluc_pts, anc_data):
  """ Returns a recommendation via UCB using hallucinated inputs in the asynchronous
      setting. """
  beta_th = _get_ucb_beta_th(gp.input_dim, anc_data.t)
  halluc_pts = (np.empty((0, gp.input_dim)) if len(halluc_pts) == 0
                     else np.array(halluc_pts))
  def _ucb_halluc_acq(x):
    """ Computes GP-UCB acquisition with hallucinated observations. """
    mu, sigma = gp.eval_with_hallucinated_observations(x, halluc_pts, uncert_form='std')
    return mu + beta_th * sigma
  return _optimise_acquisition(_ucb_halluc_acq, acq_optimiser, anc_data)

def asy_bucb(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via UCB using hallucinated inputs in the asynchronous
      setting. """
  return _halluc_ucb(gp, acq_optimiser, anc_data.evals_in_progress, anc_data)

def syn_bucb(num_workers, gp, acq_optimiser, anc_data):
  """ Returns a recommendation via Batch UCB in the synchronous setting. """
  recommendations = [asy_ucb(gp, acq_optimiser, anc_data)]
  for _ in range(1, num_workers):
    recommendations.append(_halluc_ucb(gp, acq_optimiser, recommendations, anc_data))
  return recommendations

def syn_ucbpe(num_workers, gp, acq_optimiser, anc_data):
  """ Returns a recommendation via UCB-PE in the synchronous setting. """
  # Define some internal functions.
  beta_th = _get_ucb_beta_th(gp.input_dim, anc_data.t)
  # 1. An LCB for the function
  def _ucbpe_lcb(x):
    """ An LCB for GP-UCB-PE. """
    mu, sigma = gp.eval(x, uncert_form='std')
    return mu - beta_th * sigma
  # 2. A modified UCB for the function using hallucinated observations
  def _ucbpe_2ucb(x):
    """ An LCB for GP-UCB-PE. """
    mu, sigma = gp.eval(x, uncert_form='std')
    return mu + 2 * beta_th * sigma
  # 3. UCB-PE acquisition for the 2nd point in the batch and so on.
  def _ucbpe_acq(x, yt_dot, halluc_pts):
    """ Acquisition for GP-UCB-PE. """
    _, halluc_stds = gp.eval_with_hallucinated_observations(x, halluc_pts,
                                                            uncert_form='std')
    return (_ucbpe_2ucb(x) > yt_dot).astype(np.double) * halluc_stds

  # Now the algorithm
  yt_dot_arg = _optimise_acquisition(_ucbpe_lcb, acq_optimiser, anc_data)
  yt_dot = _ucbpe_lcb(yt_dot_arg.reshape((-1, gp.input_dim)))
  recommendations = [asy_ucb(gp, acq_optimiser, anc_data)]
  for _ in range(1, num_workers):
    curr_acq = lambda x: _ucbpe_acq(x, yt_dot, np.array(recommendations))
    new_rec = _optimise_acquisition(curr_acq, acq_optimiser, anc_data)
    recommendations.append(new_rec)
  return recommendations

# EI stuff ----------------------------------------------------------------------------
def asy_ei(gp, acq_optimiser, anc_data):
  """ Returns a recommendation based on GP-EI. """
  curr_best = anc_data.curr_max_val
  def _ei_acq(x):
    mu, sigma = gp.eval(x, uncert_form='std')
    Z = (mu - curr_best) / sigma
    return (mu - curr_best)*normal_distro.cdf(Z) + sigma*normal_distro.pdf(Z)
  return _optimise_acquisition(_ei_acq, acq_optimiser, anc_data)


# Put all of them into the following namespaces.
syn = Namespace(
  # UCB
  bucb=syn_bucb,
  ucbpe=syn_ucbpe,
  # TS
  ts=syn_ts,
  )

asy = Namespace(
  # UCB
  ucb=asy_ucb,
  bucb=asy_bucb,
  # EI
  ei=asy_ei,
  # TS
  ts=asy_ts,
  hts=asy_hts,
  )

seq = Namespace(
  ucb=asy_ucb,
  ts=asy_ts,
  )
