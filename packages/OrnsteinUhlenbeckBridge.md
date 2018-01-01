---
layout: package
title: OrnsteinUhlenbeckBridge
version: 1.0.0
author: Lawrence Murray
email: lawrence.murray@it.uu.se
website-url: http://www.github.com/lawmurray/OrnsteinUhlenbeckBridge
download-url: http://www.github.com/lawmurray/OrnsteinUhlenbeckBridge/archive/master.tar.gz
github-url: http://www.github.com/lawmurray/OrnsteinUhlenbeckBridge
description: Ornstein--Uhlenbeck model for diffusion bridge sampling.
---

Synopsis
--------

    ./init.sh

This simulates a number of data sets for testing. GNU Octave is
required. Running it is optional, as a number of simulated data sets are
already included.

    ./run.sh
    
This runs a particle filter as well as samples from the posterior
distribution for a data set of a single observation at time 1.

    ./test.sh

This runs tests on the bootstrap and bridge particle filters on the simulated
data sets. Alternatively, these tests may be run as an array job on a cluster:

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
values. The form of the model is as studied in Aït-Sahalia (1999):

$$dx=(\theta_{1}-\theta_{2}x)\, dt+\theta_{3}\, dW,$$

with fixed parameters $\theta_1 = 0.0187$, $\theta_2 = 0.2610$ and $\theta_3 =
0.0224$.

It was used as a test case in Del Moral & Murray (2014). The package may be
used to reproduce the results in that paper.


References
----------

Aït-Sahalia, Y. Transition Densities for Interest Rate and Other Nonlinear
Diffusions. *The Journal of Finance*, 1999, 54, 1361--1395.

Del Moral, P. & Murray, L. M. Sequential Monte Carlo with Highly Informative
Observations, 2014. [\[arXiv\]](http://arxiv.org/abs/1405.4081)
