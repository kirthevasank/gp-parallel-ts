"""
  Some utilities for GP Bandit Optimisation.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class

import numpy as np
# Local imports

# For initialisation
# ========================================================================================
def latin_hc_indices(dim, num_samples):
  """ Obtains indices for Latin Hyper-cube sampling. """
  index_set = [list(range(num_samples))] * dim
  lhs_indices = []
  for i in range(num_samples):
    curr_idx_idx = np.random.randint(num_samples-i, size=dim)
    curr_idx = [index_set[j][curr_idx_idx[j]] for j in range(dim)]
    index_set = [index_set[j][:curr_idx_idx[j]] + index_set[j][curr_idx_idx[j]+1:]
                 for j in range(dim)]
    lhs_indices.append(curr_idx)
  return lhs_indices

def latin_hc_sampling(dim, num_samples):
  """ Latin Hyper-cube sampling in the unit hyper-cube. """
  if num_samples == 0:
    return np.zeros((0, dim))
  elif num_samples == 1:
    return 0.5 * np.ones((1, dim))
  lhs_lower_boundaries = (np.linspace(0, 1, num_samples+1)[:num_samples]).reshape(1, -1)
  width = lhs_lower_boundaries[0][1] - lhs_lower_boundaries[0][0]
  lhs_lower_boundaries = np.repeat(lhs_lower_boundaries, dim, axis=0).T
  lhs_indices = latin_hc_indices(dim, num_samples)
  lhs_sample_boundaries = []
  for i in range(num_samples):
    curr_idx = lhs_indices[i]
    curr_sample_boundaries = [lhs_lower_boundaries[curr_idx[j]][j] for j in range(dim)]
    lhs_sample_boundaries.append(curr_sample_boundaries)
  lhs_sample_boundaries = np.array(lhs_sample_boundaries)
  uni_random_width = width * np.random.random((num_samples, dim))
  lhs_samples = lhs_sample_boundaries + uni_random_width
  return lhs_samples

def random_sampling(dim, num_samples):
  """ Just picks uniformly random samples from a  dim-dimensional space. """
  return np.random.random((num_samples, dim))

def random_sampling_kmeans(dim, num_samples):
  """ Picks a large number of points uniformly at random and then runs k-means to
      select num_samples points. """
  try:
    from sklearn.cluster import KMeans
    num_candidates = np.clip(100*(dim**2), 4*num_samples, 20*num_samples)
    candidates = random_sampling(dim, num_candidates)
    centres = KMeans(n_clusters=num_samples).fit(candidates)
    return centres.cluster_centers_
  except ImportError:
    return random_sampling(dim, num_samples)

