
### Installation
- Run `source set_up_thompson` to set up all environment variables.
- To test the installation, run `bash run_all_tests.sh`. Some of the tests are
  probabilistic and could fail at times. If this happens, run the same test several times
  and make sure it is not consistently failing.

### Demo
- Check demo_simtime.py for a demo on how to run experiments with simulated time values
  (see paper below for experiments).
- We will also include a demo for real time experiments soon, but for now you can check
  out resnet_experiments/resnet_function_caller.py to see how to set it up. You need to
  use the RealWorkerManager class under bo/worker_manager.py


### Citation
If you use any part of this code in your work, please cite the following paper:
Asynchronous Parallel Bayesian Optimisation via Thompson Sampling
Kandasamy, Kirthevasan and Krishnamurthy, Akshay and Schneider, Jeff and Poczos, Barnabas
arXiv preprint arXiv:1705.09236

For questions and bug reports, please contact kandasamy@cs.cmu.edu
