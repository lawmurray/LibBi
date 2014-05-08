---
layout: post
title: Updated paper and package on resampling algorithms in LibBi
---

The following paper has been updated on arXiv, detailing the particle filter
resampling schemes implemented in LibBi, including the Metropolis and
rejection algorithms, which can provide an extra performance boost:

L. M. Murray, A. Lee, and P. E. Jacob. [Parallel resampling in the particle
filter](http://arxiv.org/abs/1301.4019), 2014.

A new [Resampling](/packages/Resampling.html) package has also been added to
the website. It can be used to reproduce the results of that paper using the
`test_resampler` command that comes built into LibBi. At time of writing, the
package requires the latest development version of LibBi from the [GitHub
repository](https://github.com/libbi/LibBi) (to be released soon as stable
version 1.1.0).
