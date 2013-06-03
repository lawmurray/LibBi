---
layout: page
title: "LibBi: Frequently asked questions"
menu: FAQ
permalink: faq.html
---

Frequently asked questions
==========================

### What's in a name?

LibBi might nominally stand for *Library for Bayesian inference*, although is
not meant to be an abbreviation as such. Development of the software began in
2009, with a working title of just *Bi* (for *Bayesian inference*), which, at
the time, was sufficiently generic for anything we might want to put in
it. *Lib* was added closer to public release, when something more unique was
required.


### Is it pronounced &quot;Lib Bee-Eye&quot; or &quot;Lib Bye&quot;?

Whichever you prefer, and for as long as you like, until you settle on
&quot;Libby&quot;.


### Who's behind LibBi?

Development of LibBi began in 2009 under a [CSIRO](http://www.csiro.au)
project that forms part of the [Computational and Simulation
Sciences](http://www.csiro.au/en/Organisation-Structure/Divisions/Mathematics-Informatics-and-Statistics/Computational-simulation-sciences.aspx)
platform. The aim of the project was to develop appropriate models and
methodology for quantifying uncertainty in marine biogeochemical models. Work
continues there and the project remains a major driver of the software's
development. Recognising potential interest in the broader scientific
community, the software was released under an open source licence in June
2013.

The main developer is Lawrence Murray. Other suspects in the abovementioned
project are John Parslow, Noel Cressie, Eddy Campbell and Emlyn Jones (see
[this paper](http://www.esajournals.org/doi/abs/10.1890/12-0312.1), also on
[arXiv](http://arxiv.org/abs/1211.1717)). Emlyn Jones in particular suffered
through the earliest versions of the software. Pierre Jacob and Anthony Lee
have both made significant contributions. Dan Pagendam has suffered through
later versions of the software.

### How is LibBi licensed?

LibBi is licensed under the [CSIRO Open Source Software License
(GPL)](https://github.com/libbi/LibBi/blob/master/LICENSE). This is the full
text of the GPL version 2 with some additional provisions.

### How can I cite LibBi?

Please cite the following paper:

Murray, L.M. (2013) Bayesian state-space modelling on high-performance hardware using LibBi.

### Why can't I just use BUGS, or JAGS, or Stan, or something else?

You can! But that may not be your best choice, depending on the problem you
have at hand. LibBi differs from these packages in two ways:

* it is specialised for state-space models (SSMs) rather than more general
  Bayesian hierarchical models, and
* it is designed from the outset for parallel computing.

The first point is reflected in the methods for inference that LibBi has
available. Its staple methods are from the family of sequential Monte Carlo
(SMC), not Gibbs (as in BUGS and JAGS) or Hamiltonian Monte Carlo (as in
Stan). LibBi *can* be used for non-SSMs by omitting the `transition` and
`initial` blocks in a model specification, but its machinery for such models
is rudimentary at this stage. The potential is there to develop in such a
direction in future, however.

We are not aware of other software packages in this space that have the same
high performance computing orientation as LibBi.
