---
layout: package
title: StochasticVolatility
version: 1.0.0
author: Lawrence Murray
email: lawrence.murray@csiro.au
website-url: http://www.github.com/lawmurray/StochasticVolatility
download-url: http://www.github.com/lawmurray/StochasticVolatility/archive/master.tar.gz
github-url: http://www.github.com/lawmurray/StochasticVolatility
description: Simple stochastic volatility model.
---

Synopsis
--------

    ./run.sh

This samples from the posterior distribution using both PMCMC (Andrieu, Doucet
& Holenstein 2010) and SMC$^2$ (Chopin, Papaspiliopoulos & Jacob 2013).

The `oct/` directory contains a few functions for plotting these results (GNU
Octave and OctBi required), in particular the `plot_and_print()` function,
which produces PDF files in the `figs/` directory.


Description
-----------

This package implements a simple stochastic volatility model. The model is fit
to 3 years of daily Standard & Poors 500 log returns, from 12 January 2002 to
30 December 2005, obtained from Yahoo Finance. The same range of data was used
in Andrieu, Doucet & Holenstein (2010).

The transition model is given by:

$$v_t \sim \mathcal{N}(\phi_v v_{t-1}, \sigma_v)$$

with $\sigma_v$ the standard deviation. The observation model is given by:

$$y_t \sim \mathcal{N}(\mu_y, \sigma_y \exp v/2).$$


References
----------

Andrieu, C.; Doucet, A. & Holenstein, R. Particle Markov Chain Monte Carlo
Methods. *Journal of the Royal Statistical Society B*, 2010, 72, 269-302.

Chopin, N.; Jacob, P. & Papaspiliopoulos, O. SMC$^2$: An Efficient Algorithm
for Sequential Analysis of State Space Models. *Journal of the Royal
Statistical Society B*, 2013, 75, 397-426.
