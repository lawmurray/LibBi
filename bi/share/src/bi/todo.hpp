/**
 * @todo Allow to specify a minimum number of blocks.
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
 * @todo * (L) Custom block/action. Bi C++ interface for writing custom functions
 * in Bi files.
 *
 * @todo * (L) pdf action as default for var ~ logpdf type expressions.
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
 *
 * @todo Use new NetCDF C++ interface.
 *
 * @todo Beta action maxLogDensity().
 *
 * @todo Automatically create directories for --output-file option.
 *
 * @todo Add --cuda-arch flag to set sm_13, sm_20 or sm_30 etc.
 *
 * @todo Give error if non-existent config file is given on command line.
 *
 * @todo Fix issue where hidden directory needs to be deleted if ./autogen.sh
 * or ./configure fails (or user hits Ctrl-C during one of these processes).
 *
 * @todo Need a locking system to ensure exclusive use of build directories.
 * Having a problem on cluster where multiple jobs are trying to build
 * simultaneously in the same directory and conflict with each other. Could,
 * for example, put a lock file in each directory to reserve it for a
 * particular process, then delete the file when that process is done with it.
 * If another process wants to use it, rather than waiting, it could simply
 * create a new directory with another name and work from there.
 *
 * @todo Allowing arbitrary indexing expressions on the right, not just
 * expressions of type k, k + n and k - n.
 */
