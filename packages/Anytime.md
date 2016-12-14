---
layout: package
title: Anytime
version: 1.0.0
author: Lawrence Murray
email: lawrence.murray@it.uu.se
website-url: http://www.github.com/lawmurray/Anytime
download-url: http://www.github.com/lawmurray/Anytime/archive/stable.tar.gz
github-url: http://www.github.com/lawmurray/Anytime
description: "Experiments for the paper Anytime Monte Carlo."
---

Synopsis
--------

    ./run.sh
    
Runs the Lorenz '96 inference experiments using SMC$^2$ on a cluster.

    ./run_toy.sh

Runs the toy examples in GNU Octave. Alternatively, these can be run from MATLAB, inspect the script to see how this works.

    ./run_tau.sh
    
Runs the Lorenz '96 likelihood and timing experiments.

    ./run_bifurc.sh
    
Runs the Lorenz '96 experiments necessary to construct the bifurcation plots in the paper.

    ./plot.sh
    
Produces all plots after these runs are complete.


Description
-----------

This package can be used to reproduce the results of the experiments in Murray *et al.* 2016.

Some configuration of the `*.sh` scripts and associated `*.conf` configuration files may be necessary to get them running on a particular system. Also see the Lorenz96 package, on which this package is based.

The package requires new features in LibBi 1.3.0.


References
----------

L. M. Murray, S. Singh, P. E. Jacob and A. Lee. *Anytime Monte Carlo*. 2016. Online at [https://arxiv.org/abs/1612.03319](https://arxiv.org/abs/1612.03319).
