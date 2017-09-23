## gp-parallel-ts
This is a python implementation of parallelised Bayesian optimisation via Thompson
sampling. For more details, see our paper below.

### Download
You can download the code from github
```bash
$ git clone https://github.com/kirthevasank/gp-parallel-ts
```

### Installation & Getting Started
- Run `source set_up_thompson` to set up all environment variables.
- You also need to build the direct fortran library. For this `cd` into
  `direct_fortran` and run `bash make_direct.sh`. You will need a fortran compiler
  such as gnu95. Once this is done, you can run `simple_direct_test.py` to make sure that
  it was installed correctly.
- To test the installation, run `bash run_all_tests.sh`. Some of the tests are
  probabilistic and could fail at times. If this happens, run the same test several times
  and make sure it is not consistently failing.

### Demo
- Check demo_simtime.py for a demo on how to run experiments with simulated time values
  (see paper below for experiments).
- We will also include a demo for real time experiments soon, but for now you can check
  out resnet_experiments/resnet_function_caller.py to see how to set it up. You need to
  use the
[RealWorkerManager](https://github.com/kirthevasank/gp-parallel-ts/blob/master/bo/worker_manager.py) class under bo/worker_manager.py

### Some notes
- We choose the GP hyper-parameters every 25 iterations via marginal likelihood
  maximisation for each GP. The chosen values are printed out.
- We report progress on the optimisation every 20 iterations.

### Citation
If you use any part of this code in your work, please cite our
[Arxiv paper](https://arxiv.org/pdf/1705.09236.pdf):

```bibtex
@article{kandasamy2017asynchronous,
  title={Asynchronous Parallel Bayesian Optimisation via Thompson Sampling},
  author={Kandasamy, Kirthevasan and Krishnamurthy, Akshay and Schneider, Jeff and Poczos, Barnabas},
  journal={arXiv preprint arXiv:1705.09236},
  year={2017}
}
```


### License
This software is released under the MIT license. For more details, please refer
[LICENSE.txt](https://github.com/kirthevasank/gp-parallel-ts/blob/master/LICENSE.txt).

"Copyright 2017 Kirthevasan Kandasamy"

- For questions and bug reports please email kandasamy@cs.cmu.edu
