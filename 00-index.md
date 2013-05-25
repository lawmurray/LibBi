---
layout: index
title: LibBi (library for Bayesian inference)
menu: Home
permalink: index.html
---

LibBi is used for state-space modelling and Bayesian inference on modern
computer hardware, including multi-core CPUs, many-core GPUs (graphics
processing units) and distributed-memory clusters.

The staple methods used in LibBi are those based on sequential Monte Carlo
(SMC). This includes particle Markov chain Monte Carlo (PMCMC) and
SMC<sup>2</sup> methods. Extra methods include the extended Kalman filter and
some parameter optimisation routines.

LibBi consists of a C++ template library, as well as a parser and compiler,
written in Perl, for its own domain-specific language that is used to specify
models.
