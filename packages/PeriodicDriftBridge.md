---
layout: package
name: PeriodicDriftBridge
version: 1.0.0
author: Lawrence Murray
email: lawrence.murray@csiro.au
website-url: http://www.github.com/lawmurray/PeriodicDriftBridge
download-url: http://www.github.com/lawmurray/PeriodicDriftBridge/archive/master.tar.gz
github-url: http://www.github.com/lawmurray/PeriodicDriftBridge
description: Periodic drift process for diffusion bridge sampling.
---

Synopsis
--------

    ./init.sh

This simulates a number of data sets for testing and fits the bridge weight
function. GNU Octave is required. Running it is optional, as a number of
simulated data sets are already included.

    ./run.sh
    
This runs a particle filter as well as samples from the posterior distribution
for a fixed data set of four observations given in Lin, Chen & Mykland (2010).

    ./test.sh

This runs tests on the bootstrap and bridge particle filters on the simulated
data sets. Alternatively, these tests may be run as an array job on a cluster:

    qsub -t 0-15 qsub_test_bridge.sh
    qsub -t 0-15 qsub_test_bootstrap.sh
    qsub -t 0-15 qsub_test_exact.sh

The last line computes normalising constants to be used as "exact" values
when computing the MSE metric for comparison plots.

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

This package implements the periodic drift diffusion process introduced in
Beskos et al. (2006) and further studied in Lin, Chen & Mykland (2010). The
form of the process model is the Ito stochastic differential
equation$$dx=\sin(x-\pi)\, dt+dW.$$ The task is to simulate diffusion bridges
between the observed values. It was used as a test case in Del Moral & Murray
(2014). The package may be used to reproduce the results in that paper.


References
----------

Beskos, A.; Papaspiliopoulos, O.; Roberts, G. & Fearnhead, P. Exact and
computationally efficient likelihood-based estimation for discretely observed
diffusion processes. *Journal of the Royal Statistical Society Series B*,
2006, 68, 333-382.

Del Moral, P. & Murray, L. M. Sequential Monte Carlo with Highly Informative
Observations, 2014. [\[arXiv\]](http://arxiv.org/abs/1405.4081)

Lin, M.; Chen, R. & Mykland, P. On Generating Monte Carlo Samples of
Continuous Diffusion Bridges. *Journal of the American Statistical
Association*, 2010, 105, 820-838.
