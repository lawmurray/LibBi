---
layout: package
name: PZ
version: 1.0.0
author: Lawrence Murray
email: lawrence.murray@csiro.au
website-url: http://www.github.com/lawmurray/PZ
download-url: http://www.github.com/lawmurray/PZ/archive/master.tar.gz
github-url: http://www.github.com/lawmurray/PZ
description: Lotka-Volterra-like model of the interaction of phytoplankton (prey) and zooplankton (predator).
---

Synopsis
--------

    ./run.sh

This samples from the prior and posterior distributions. The `oct/` directory
contains a few functions for plotting these results (GNU Octave and OctBi
required).

A synthetic data set is provided, but a new one one may be generated with
`init.sh` (GNU Octave and OctBi required).


Description
-----------

This package is based on a Lotka-Volterra model of the interaction between
phytoplankton $P$ (prey) and zooplankton $Z$ (predator). It differs from the
classic Lotka-Volterra by having a stochastic growth term for phytoplankton,
and quadratic mortality term for zooplankton.

The process model is given by the equations:
\begin{eqnarray}
\frac{dP}{dt} &=& \alpha_t P - cPZ \\\\
\frac{dZ}{dt} &=& ecPZ - m_lZ - m_q Z^2,
\end{eqnarray}
where $t$ is time in days, and the stochastic growth term $a_t$ is drawn daily
as $\alpha_t \sim \mathcal{N}(\mu,\sigma)$, with $\mu$ and $\sigma$ being the
two parameters of the model.

This version of the model was originally used in Jones, Parslow & Murray
(2010). Its behaviour under sampling with the particle marginal
Metropolis-Hastings (PMMH) sampler is also studied in Murray, Jones & Parslow
(2013).

References
----------

Jones, E.; Parslow, J. & Murray, L. M. *A Bayesian approach to state and
parameter estimation in a Phytoplankton-Zooplankton model*. Australian
Meteorological and Oceanographic Journal, 2010, 59, 7-16.

Murray, L. M.; Jones, E. M. & Parslow, J. On collapsed state-space models and
the particle marginal Metropolis-Hastings sampler, 2013.
