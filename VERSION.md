LibBi VERSION.md
================

v1.4.5
------

Fixes:

* Fixed OpenMP check
* Fixed border case for binomial distribution

v1.4.4
------

Fixes:

* Fixed a bug in the calculation of binomial densities

v1.4.3
------

New features:

* Metropolis-Hastings sampling with Gaussian proposals now work if the standard deviation is zero; this makes it possible to repeatedly propose the same theta, for example to adjust the number of particles, without the need to change the model and recompile.
* `build-dir`, `version` and `with-lldb` command line options.
* Parameter or initial condition assignments are now overwritten by a given init file.

Fixes:

* The input-interval has been changed to run from [t, t+1) so that timings are preserved if output from a previous run is used as input.
* Fixed compile error in Metropolis resampler.
* Fixed some CUDA problems.

v1.4.2
------

Performance:

* Improved beta and negative binomial random generators.

v1.4.1
------

Fixes:

* Fixed compatibility with automake 1.16.
* Fixes in truncated Gaussian and negative binomial probability distributions.

v1.4.0
------

New features:

* New distributions: Poisson, binomial, negative binomial, beta-binomial, exponential.
* Added `--cuda-arch` option for specifying the CUDA architecture.
* Added `--enable-cudafastmath` option for using fast CUDA maths.

Changes:

* State and noise variables now can have inputs.

Performance:

* Improved truncated Gaussian sampler.
* Models are now only simulated if the parameter prior probability density is >0.
* Parsing and translating models is now faster.

Fixes:

* No more error is thrown if all particle weights are 0.
* `optimise` now works again.
* No more warning is thrown if the model name does not match the file name.

v1.3.0
------

New features:

* Added SMC$^2$ with anytime moves, enabled with `--tmoves` command-line option.
* Added `--mpi-hostfile` option.

v1.2.0
------

New features:

* Streamlined installation instructions: Perl module dependencies are now
  installed automatically, `make test` checks for these dependencies, and
  MAGMA is optional.
* CPU builds now work on Windows under Cygwin.
* CUDA builds now work on Mac OS X.
* AVX support added (use `--enable-avx`).
* SMC^2 now works with a Kalman filter over the state, not just a particle
  filter, and has been renamed marginal SIR to reflect this.
* Output at observation times can now be disabled (use
  `--without-output-at-obs`).
* Non-const scalar expressions are now allowed as indexes on the left side of
  actions.

Changes:

* Distribution now contains a compiled version of the manual.
* Major refactoring of methods code to reduce replication.

Performance:

* Improved performance under `--enable-sse`.

Fixes:

* Fixed the first observation in a time series being zero when `--start-time`
  did not match the earliest time in `--obs-file`.
* Fixed CUDA kernel execution configurations for large numbers of variables.
* Fixed matrix range being ignored on the right side of actions.


v1.1.0
------

New features:

* Added bridge particle filter (from Del Moral & Murray, 2014).
* Added built-in variables `t_now`, `t_last_input` and `t_next_obs`.
* Added `transpose` and `cholesky` actions.
* Added `log` argument to `pdf` action.
* Added matrix-matrix multiply.
* Added range syntax, e.g. `x[0:4]`.
* Added checks for `*.sh`, `data/*.nc` and `oct/*.m` files to
  `libbi package --validate` checks.

Changes:

* Removed ill-defined `uninformative` action.
* Action syntax made stricter: an action that returns a scalar cannot be
  applied to a vector on the left.

Performance:

* Minor performance improvements in GPU random number generation and
  resampling.
* Minor I/O performance improvements.

Fixes:

* Restored NetCDF 4.1 support.
* Fixed initialisation of parameters from init file when
  `--with-transform-initial-to-param` used.
* Fixed `C_` and `U2_` variables in Kalman filter output files.
* Fixed reporting of log-likelihood in PMCMC output when particle filter
  degenerates.
* Fixed build error when model name does not begin with an uppercase letter.
* Fixed runtime error when empty `--output-file` given.
* Fixed race condition in locking the build directory under some
  circumstances.
* Fixed unnecessary recompiles triggered by new hash implementation in newer
  versions of Perl.

v1.0.2
------

* Removed dependency on NetCDF C++ interface, the C interface is now used
  directly.
* Added `'extended'` boundary condition for dimensions.
* Added `--enable-openmp`/`--disable-openmp` command-line options.
* Added `--enable-gpu-cache/--disable-gpu-cach`e command-line options for better
  control of GPU memory usage.
* Added `--adapter-ess-rel`  command-line option to avoid adaptation of proposal
  in SMC$^2$ when ESS too low.
* Several bug and compatibility fixes.

v1.0.1
------

* Added additional material to manual, including new section with guidance on
  tuning the proposal distribution and number of particles when using PMMH.
* Fixed sampling of joint distribution (`--target joint` now implies
  `--with-param-to-state`, just as `--target prior` and `--target prediction`
  do).
* Fixed reordering of actions and blocks when the same variable appears on
  the left more than once.
* Fixed bug in GPU implementation of multinomial resampler.
* Added `--dry-parse` option to remove parsing overhead when binaries have
  already been compiled.

v1.0.0
------

* First public release.
