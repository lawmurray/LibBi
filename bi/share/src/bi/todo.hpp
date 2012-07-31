/**
 * @todo * (L) Build system that combines genbi and libbi.
 *
 * @todo New interface for models.
 *
 * @todo * (L) Add --transform-obs-to-state command line argument, remove ns
 * dimension from input files, use np dimension instead.
 * 
 * @todo Allow to specify a minimum number of blocks.
 *
 * @todo * (A) Get PMMH to look at the actual number of observations.
 *
 * @todo * (L) Fix ESS threshold for MPI.
 *
 * @todo * (A) Make ESS threshold clear with adaptive particle filter.
 *
 * @todo Adapting number of samples of initial conditions.
 *
 * @todo Adapting the number of theta-particles in SMC^2.
 *
 * @todo Include meta-data on command in output NetCDF files, include *.bi
 * file also?
 *
 * @todo Output diagnostics (e.g. number of particles at each time for each
 * filter run during PMMH).
 *
 * @todo * (L) Matrix actions.
 * 
 * @todo * (L) Custom block/action. Bi C++ interface for writing custom functions
 * in Bi files.
 *
 * @todo * (L) pdf action as default for var ~ logpdf type expressions.
 *
 * @todo * (All) Example models for release. Look at POMP?
 *
 * @todo * (L) "bi package" command.
 *
 * @todo Test suite (list of bi commands that can be run to test
 * changes). Regression tests.
 *
 * @todo * (P) Output trajectories with SMC^2.
 *
 * @todo Swap GPU memory into host memory, swap host memory to disk.
 *
 * @todo * (P) Documentation on producing NetCDF files from e.g. CSV files.
 *
 * @todo Iterated filtering.
 *
 * @todo Fixed lag smoother.
 *
 * @todo Collapsed state-space model.
 *
 * @todo Log and logit transforms of parameters (e.g. --transform-*).
 * 
 * @todo Clock resampling.
 *
 * @todo Evidence estimation using PMCMC.
 *
 * @todo Univariate proposals.
 *
 * @todo Test functions for time T.
 *
 * @todo * (L) Multivariate normal
 *
 * @todo * (P) Exponential pdf, poisson pdf.
 *
 * @todo PMMH normalising constant estimates at each time.
 *
 * @todo * (L) Review OctBi.
 *
 * @todo * (A) Get OctBi working with MATLAB?
 *
 * @todo * (P) Review RBi.
 */
