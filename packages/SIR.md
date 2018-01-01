---
layout: package
title: SIR
version: 1.0.0
author: Lawrence Murray
email: lawrence.murray@it.uu.se
website-url: http://www.github.com/lawmurray/SIR
download-url: http://www.github.com/lawmurray/SIR/archive/master.tar.gz
github-url: http://www.github.com/lawmurray/SIR
description: Susceptible/infectious/recovered epidemiological compartmental model for Russian influenza data set.
---

Synopsis
--------

    ./init.sh

This fits the bridge weight function. GNU Octave and GPML are
required. Running it is optional, as the included files already have this set
up.

    ./run.sh
    
This samples from the posterior distribution using a Russian influenza data
set.

    octave --path oct/ --eval "plot_and_print"

This plots the results.


Description
-----------

This package includes a stochastic SIR (susceptible/infectious/recovered)
epidemiological compartmental model of the form

$$\begin{eqnarray}
dS(t) &=& -\beta(t)S(t)I(t)\, dt \\\\
dI(t) &=& \left(\beta(t)S(t)I(t)-\nu(t)I(t)\right)\, dt \\\\
dR(t) &=& \nu(t)I(t)\, dt \\\\
d\log\beta(t) &=& \left(\theta_{\beta,1}-\theta_{\beta,2}\log\beta(t)\right)\, dt+\theta_{\beta,3}\, dW_{\beta}(t) \\\\
d\log\nu(t) &=& \left(\theta_{\nu,1}-\theta_{\nu,2}\log\nu(t)\right)\, dt+\theta_{\nu,3}\, dW_{\nu}(t).
\end{eqnarray}$$

It also includes an observational data set of an epidemic of Russian influenza
at a boys boarding school (Anonymous 1978). As this is a closed system the
observations are considered exact, and the task is to simulate diffusion
bridges between the observed values, and to estimate parameters.

The model and data set were used as a test case in Del Moral & Murray
(2014). The package may be used to reproduce the results in that paper.


References
----------

Anonymous. Influenza in a boarding school. *British Medical Journal*, 1978, 1,
587.

Del Moral, P. & Murray, L. M. Sequential Monte Carlo with Highly Informative
Observations, 2014. [\[arXiv\]](http://arxiv.org/abs/1405.4081)
