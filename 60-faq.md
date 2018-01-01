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

Development of LibBi began in 2009 at [CSIRO](http://www.csiro.au). The aim of the initial project was to develop appropriate models and
methodology for quantifying uncertainty in marine biogeochemical models. Recognising potential interest in the broader scientific
community, the software was released under an open source licence in June
2013 and has since been used in several other problem domains.

The main developer is [Lawrence Murray](http://www.indii.org/research/). [Sebastian Funk](http://sbfnk.github.io/) has made significant contributions, especially to the [RBi](/related.html) interface for R, and Homebrew packaging.

### How is LibBi licensed?

LibBi is licensed under the [CSIRO Open Source Software License
(GPL)](https://github.com/lawmurray/LibBi/blob/master/LICENSE). This is the full
text of the GPL version 2 with some additional provisions.

### How can I cite LibBi?

Please cite the following paper:

L. M. Murray, Bayesian state-space modelling on high-performance hardware
using LibBi, 2013. [\[arXiv\]](http://arxiv.org/abs/1306.3277)

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
