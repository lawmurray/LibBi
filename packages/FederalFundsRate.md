---
layout: package
name: FederalFundsRate
version: 1.0.0
author: Lawrence Murray
email: lawrence.murray@csiro.au
website-url: http://www.github.com/lawmurray/FederalFundsRate
download-url: http://www.github.com/lawmurray/FederalFundsRate/archive/master.tar.gz
github-url: http://www.github.com/lawmurray/FederalFundsRate
description: Ornstein--Uhlenbeck model for diffusion bridge sampling and parameter estimation with U.S. Federal Funds Rate data.
---

Synopsis
--------

    ./init.sh

This simulates a number of parameter sets from the prior for testing. GNU
Octave is required. Running it is optional, as a number of parameter sets are
already included.

    ./run.sh
    
This runs a particle filter as well as samples from the posterior distribution
using a Federal Funds Rate data set.

    ./test.sh

This runs tests on the bootstrap and bridge particle filters on the simulated
parameter sets. Alternatively, these tests may be run as an array job on a
cluster:

    qsub -t 0-15 qsub_test_bridge.sh
    qsub -t 0-15 qsub_test_bootstrap.sh

Finally, results may be plot with:

    octave --path oct/ --eval "plot_and_print"

GNU Octave and OctBi are required.

Note that, as of version 1.1.0 of LibBi, running any of these gives the
warnings:

    Warning (line 29): 'obs' variables should not appear on the right side of actions in the 'transition' block.
    Warning (line 42): 'obs' variables should not appear on the right side of actions in the 'lookahead_transition' block.

This is normal.

   
Description
-----------

This package includes an Ornstein--Uhlenbeck model that is observed
directly. The task is to simulate diffusion bridges between the observed
values. The form of the model is: $$dx=(\theta_{1}-\theta_{2}x)\,
dt+\theta_{3}\, dW,$$ with parameters $\theta_1$, $\theta_2$ and $\theta_3$ to
be estimated. It was used as a test case in Del Moral & Murray (2014). The
package may be used to reproduce the results in that paper.

A U.S. Federal Funds Rate data set is included. The data set contains 25 years
of data at a monthly time interval, beginning January 1989 and ending December
2013. The observation for January 1989 defines the initial value of the system
and is hardcoded in the model file. The remaining observations, beginning
February 1989, are in the file `data/obs.nc`. The data set was obtained from
http://www.federalreserve.gov/releases/h15/data.htm on 7 April 2014.

References
----------

Del Moral, P. & Murray, L. M. Sequential Monte Carlo with Highly Informative
Observations, 2014. [\[arXiv\]](http://arxiv.org/abs/1405.4081)
