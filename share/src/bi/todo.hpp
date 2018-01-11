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
 * @todo Tidy up output file variable names, perhaps precede internal
 * variables with the program or schema name, e.g. simulate.time,
 * filter.logweight etc.
 *
 * @todo bi smooth
 *
 * @todo Consider removing noise variables, or keep "noise" keyword for
 * semantics only, but treat as state variable internally.
 *
 * @todo Sparse updaters not working with ranges on left of actions.
 *
 * @todo Replace gpu_vector and host_vector with one vector class that
 * determines location based on allocator. Similar for matrix classes.
 */
