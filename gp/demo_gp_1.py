"""
  A simple demo for gps.
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=superfluous-parens

from argparse import Namespace
import numpy as np
# Local
import gp_core
import gp_instances
import kernel


def get_data():
  """ Generates data. """
  func = lambda t: (-70 * (t-0) * (t-0.35) * (t+0.55) * (t-0.65) * (t-0.97)).sum(axis=1)
  N = 5
  X_tr = np.array(range(N)).astype(float).reshape((N, 1))/N + 1/(float(2*N))
  Y_tr = func(X_tr)
  kern = kernel.SEKernel(1, 1, 0.5)
  data = Namespace(func=func, X_tr=X_tr, Y_tr=Y_tr, kern=kern)
  return data

def _demo_common(gp, data, desc):
  """ Common processes for the demo. """
  lml = gp.compute_log_marginal_likelihood()
  print(desc + ': Log-Marg-Like: ' + str(lml) + ', kernel: ' + str(gp.kernel.hyperparams))
  gp.visualise(true_func=data.func, boundary=[0, 1])

def demo_gp_given_hps(data, kern, desc):
  """ A demo given the kernel hyper-parameters. """
  mean_func = lambda x: np.array([data.Y_tr.mean()] * len(x))
  noise_var = data.Y_tr.std()/10
  est_gp = gp_core.GP(data.X_tr, data.Y_tr, kern, mean_func, noise_var)
  _demo_common(est_gp, data, desc)

def demo_gp_fit_hps(data, desc):
  """ A demo where the kernel hyper-parameters are fitted. """
  fitted_gp, _ = (gp_instances.SimpleGPFitter(data.X_tr, data.Y_tr)).fit_gp()
  _demo_common(fitted_gp, data, desc)

def main():
  """ Main function. """
  data = get_data()
  print('First fitting a GP with the given kernel. Close window to continue.')
  demo_gp_given_hps(data, data.kern, 'Given Kernel')
  print('\nNow estimating kernel via marginal likelihood. Close window to continue.')
  demo_gp_fit_hps(data, 'Fitted Kernel')


if __name__ == '__main__':
  main()

