---
layout: package
name: Lorenz96
version: 1.0.0
author: Lawrence Murray
email: lawrence.murray@csiro.au
website-url: http://www.github.com/lawmurray/Lorenz96
download-url: http://www.github.com/lawmurray/Lorenz96/archive/master.tar.gz
github-url: http://www.github.com/lawmurray/Lorenz96
description: Lorenz '96 differential equation model.
---

Synopsis
--------

    ./run.sh

This samples from the prior and posterior distributions, and performs a
posterior prediction. The `oct/` directory contains a few functions for
plotting these results (GNU Octave and OctBi required).

A synthetic data set is provided, but a new one one may be generated with
`init.sh` (GNU Octave and OctBi required).

The `time.sh` script can be used to reproduce the timing results in Murray
(2013).

Description
-----------

The original, deterministic Lorenz '96 model (Lorenz 2006) is given by
$$\frac{dx_{n}}{dt}=x_{n-1}(x_{n+1}-x_{n-2})-x_{n}+F,$$ where $\mathbf{x}$ is
the state vector, of length 8 in this package, with subscripts indexing its
components in a circular fashion. $F$ is a forcing parameter. This form of the
model is given in the `Lorenz96Deterministic.bi` file.

A stochastic extension of the model adds an additional $\sigma$ parameter and
rewrites the above ordinary differential equation as a stochastic differential
equation:
$$dx_{n}=\left(x_{n-1}(x_{n+1}-x_{n-2})-x_{n}+F\right)\, dt+\sigma\, dW_{n}.$$
This form is specified in `Lorenz96.bi` and used for inference in LibBi.

The interest in the Lorenz '96 model is that its dimensionality can be scaled
arbitrarily, and that, according to this number of dimensions and $F$, the
deterministic model exhibits varying behaviours from convergence, to
periodicity, to chaos. The stochastic model exhibits similar behaviours.

The model is one of the examples given in the LibBi introductory paper (Murray
2013). The package may be used to reproduce the results in that paper.

References
----------

E. N. Lorenz. *Predictability - a problem partly solved*, chapter 3, page
118. Cambridge University Press, 2006.

L. M. Murray. *Bayesian state-space modelling on high-performance hardware
using LibBi*. 2013.
