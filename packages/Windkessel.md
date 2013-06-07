---
layout: package
name: Windkessel
version: 1.0.0
author: Lawrence Murray
email: lawrence.murray@csiro.au
website-url: http://www.github.com/lawmurray/Windkessel
download-url: http://www.github.com/lawmurray/Windkessel/archive/master.tar.gz
github-url: http://www.github.com/lawmurray/Windkessel
description: Three-element windkessel model of the arterial system.
---

Synopsis
--------

    ./run.sh

This samples from the prior and posterior distributions. The `oct/` directory
contains a few functions for plotting these results (GNU Octave and OctBi
required).

Synthetic inputs and observations are provided in the `data/` directory, but
new files may be generated with the `init.sh` script (GNU Octave and OctBi
required).

Description
-----------

The three-element windkessel model (Frank 1899, Westerhof et al. 2009, Kind et
al. 2010) can be given by the discrete-time transition model: $$P_{p}(t+\Delta
t)=\exp\left(-\frac{\Delta
t}{RC}\right)P_{p}(t)+R\left(1-\exp\left(-\frac{\Delta
t}{RC}\right)\right)F(t)$$ and observation model: $$P_{a}(t)=P_{p}(t)+F(t)Z.$$
where $P_{a}$ is aortal blood pressure, $P_{p}$ peripheral blood pressure, and
$F$ blood flow. $R$, $C$ and $Z$ are parameters.

The windkessel model has linear-Gaussian transition and observation models,
and so is suitable for inference with the Kalman filter. The package has been
developed primarily for demonstrating and testing inference methods.

The model is one of the examples given in the LibBi introductory paper
(Murray 2013). The package may be used to reproduce the results in that paper.

References
----------

N. Westerhof, J.-W. Lankhaar, and B. E. Westerhof. *The arterial
windkessel*. Medical and Biological Engineering and Computing, 47(2):131–141,
2009. doi: 10.1007/s11517-008-0359-2.

O. Frank. *Die grundform des arterielen pulses erste abhandlung: mathematische
analyse*. Zeitschrift fuer Biologie, 37:483–526, 1899.

T. Kind, T. J. C. Faes, J.-W. Lankhaar, A. Vonk-Noordegraaf, and
M. Verhaegen. *Estimation of three- and four-element windkessel parameters
using subspace model identification*. IEEE Transactions on Biomedical
Engineering, 57:1531–1538, 2010. doi: 10.1109/TBME.2010.2041351.

L. M. Murray. *Bayesian state-space modelling on high-performance hardware
using LibBi*. 2013.
