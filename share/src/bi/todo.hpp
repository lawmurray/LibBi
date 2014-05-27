/**
 * @todo Allow to specify a minimum number of blocks.
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
 * @todo Custom block/action. Bi C++ interface for writing custom functions
 * in Bi files.
 *
 * @todo Fixed lag smoother.
 *
 * @todo Disturbance state-space model.
 *
 * @todo Log and logit transforms of parameters (e.g. --with-transform-*).
 * 
 * @todo Evidence estimation using PMCMC.
 *
 * @todo Univariate proposals.
 *
 * @todo Multivariate normal
 *
 * @todo Exponential pdf, poisson pdf.
 *
 * @todo Get OctBi working with MATLAB.
 *
 * @todo Review RBi.
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
 *
 * @todo Sparse updaters not working with ranges on left of actions.
 *
 * @todo Replace gpu_vector and host_vector with one vector class that
 * determines location based on allocator. Similar for matrix classes.
 *
 * @todo If high-level filter() method is now the same for all filters,
 * should they be wrapped in a Filter<F> type class that calls their
 * lower-level functions, save the repetition? Should use curiously recurring
 * to inherit from type F, so that it can call the lower-level interface.
 */
