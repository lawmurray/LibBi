/**
 * @todo Allow to specify a minimum number of blocks.
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
 * @todo * (L) Custom block/action. Bi C++ interface for writing custom functions
 * in Bi files.
 *
 * @todo * (All) Example models for release. Look at POMP?
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
 * @todo Fixed lag smoother.
 *
 * @todo Collapsed state-space model.
 *
 * @todo Log and logit transforms of parameters (e.g. --transform-*).
 * 
 * @todo Evidence estimation using PMCMC.
 *
 * @todo Univariate proposals.
 *
 * @todo * (L) Multivariate normal
 *
 * @todo * (P) Exponential pdf, poisson pdf.
 *
 * @todo PMMH normalising constant estimates at each time.
 *
 * @todo * (A) Get OctBi working with MATLAB.
 *
 * @todo * (P) Review RBi.
 *
 * @todo Use new NetCDF C++ interface. Or possible using C interface is an
 * easier step from the old C++ interface, and avoids having to specially
 * compile the new C++ interface in the current NetCDF distribution?
 *
 * @todo Add --cuda-arch flag to set sm_13, sm_20 or sm_30 etc.
 *
 * @todo Tidy up output file variable names, perhaps precede internal
 * variables with the program or schema name, e.g. simulate.time,
 * filter.logweight etc.
 *
 * @todo bi smooth
 *
 * @todo AVX support (should be very similar to SSE support, but with slightly
 * different intrinsics).
 *
 * @todo Replace BI_REAL(1.0/0.0) with BI_INF constant. Introduce named
 * literals such as true, false, pi, nan, inf into language.
 *
 * @todo Consider removing noise variables, or keep "noise" keyword for
 * semantics only, but treat as state variable internally.
 */
