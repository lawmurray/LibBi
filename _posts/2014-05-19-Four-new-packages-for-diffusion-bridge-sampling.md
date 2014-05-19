---
layout: post
title: Four new packages for diffusion bridge sampling
date: 2014-05-19 12:00:00 +08:00
---

Four new packages are now available to demonstrate the bridge particle filter
included in LibBi 1.1.0. The bridge particle filter is described in this new
paper:

Del Moral, P. & Murray, L. M. Sequential Monte Carlo with highly informative
Observations, 2014. [\[arXiv\]](http://arxiv.org/abs/1405.4081)

It is meant for state-space models where each observation is highly
informative on the state process. A special case is that of diffusion bridge
sampling, where there is no observation noise---the observation is of the
state directly. The filter works by introducing a schedule of intermediate
times at which additional weighting and resampling steps are performed.

The bridge particle filter is enabled in LibBi by adding a `--filter bridge`
command-line option, and a `--nbridges n` option to set the frequency of
intermediate times. A `bridge` top-level block is then added to the model,
describing the additional weighting function.

The four packages are:

* [OrnsteinUhlenbeckBridge](/packages/OrnsteinUhlenbeckBridge.html) a
  linear--Gaussian Ornstein--Uhlenbeck process with fixed parameters.

* [FederalFundsRate](/packages/FederalFundsRate.html) a linear--Gaussian
  Ornstein--Uhlenbeck process with parameter estimation for a Federal Funds
  Rate data set.

* [PeriodicDriftBridge](/packages/PeriodicDriftBridge.html) a nonlinear
  periodic drift process with fixed parameters.

* [SIR](/packages/SIR.html) a multivariate and nonlinear
  susceptible/infected/recovered compartmental model used in epidemiology,
  with parameter estimation for an influenza data set.

The existing [NPZD](/packages/NPZD.html) package has also been updated to
support (and use by default) the new bridge particle filter.

These four new packages, plus the NPZD package, are used in the paper to
demonstrate the new method, so further details can be found there.
