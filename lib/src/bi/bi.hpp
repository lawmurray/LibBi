/**
 * @file
 *
 * Documentation elements.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */

/**
 * @defgroup model Models
 *
 *   @defgroup model_high High-level interface
 *   @ingroup model
 *   High-level, runtime interface for constructing and querying models.
 *
 *   @defgroup model_low Low-level interface
 *   @ingroup model
 *   Low-level, static interface for constructing and querying models.
 *
 *     @defgroup model_trait Traits
 *     @ingroup model_low
 *     Node traits.
 *
 *     @defgroup model_typelist Type lists
 *     @ingroup model_low
 *     Node type lists.
 *
 * @defgroup method Methods
 * Inference methods.
 *
 *   @defgroup method_updater Updaters
 *   @ingroup method
 *   Lower level components of methods.
 *
 *   @defgroup method_strategy Strategies
 *   @ingroup method
 *   Encapsulation of node type specific operations.
 *
 * @defgroup math Math
 *
 *   @defgroup math_gpu Device vectors and matrices
 *   @ingroup math
 *
 *   @defgroup math_host Host vectors and matrices
 *   @ingroup math
 *
 *   @defgroup math_view Matrix and vector views
 *   @ingroup math
 *
 *   @defgroup math_op Matrix and vector operations
 *   @ingroup math
 *
 *   @defgroup math_primitive Vector primitives
 *   @ingroup math
 *
 *   @defgroup math_functor Functors
 *   @ingroup math
 *
 *   @defgroup math_pdf Probability density functions
 *   @ingroup math
 *
 *   @defgroup math_rng Random number generation
 *   @ingroup math
 *
 * @defgroup kd kd trees
 *
 * @defgroup state State
 *
 *   @defgroup state_host Host memory bindings
 *   @ingroup state
 *
 *   @defgroup state_gpu Device memory bindings
 *   @ingroup state
 *
 * @defgroup io I/O
 *
 *   @defgroup io_cache Caches
 *   @ingroup io
 *
 *   @defgroup io_mask Masks
 *   @ingroup io
 *
 * @defgroup concept Concepts
 *
 * @defgroup misc Miscellaneous
 *
 *   @defgroup misc_iterators Iterators
 *   @ingroup misc
 *
 *   @defgroup misc_allocators Allocators
 *   @ingroup misc
 */

/**
 * @page Headers
 *
 * @section Headers A note on headers
 *
 * Header files are given two extensions in bi:
 *
 * @li <i>*.hpp</i> headers may be safely included in any C++ (e.g.
 * <i>*.cpp</i>) or CUDA (<i>*.cu</i>) source files,
 * @li <i>*.cuh</i> headers may only be safely included in CUDA (<i>*.cu</i>)
 * source files. They include CUDA C extensions.
 *
 * Note that not all <i>*.hpp</i> headers can be safely included in
 * <i>*.cu</i> files either, due to CUDA compiler limitations, particularly
 * those headers that further include Boost headers which make extensive use
 * of templates. Efforts have been made to quarantine such incidences from
 * the CUDA compiler, but mileage may vary.
 *
 * Typically, it is worth writing a client program in <i>*.cpp</i> files as
 * much as possible. These should compile safely, although may be missing
 * template instantiations related to device code come link time. Compiling
 * and linking a single <i>*.cu</i> file with explicit template instantiations
 * of the methods being used (e.g. #bi::Simulator, #bi::ParticleFilter)
 * should be sufficient to satisfy the outstanding device code dependencies.
 */

/**
 * @page references References
 *
 * @section references References
 *
 * @anchor Andrieu2010
 * Andrieu, C.; Doucet, A. & Holenstein, R. Particle Markov chain Monte Carlo
 * methods. <i>Journal of the Royal Statistical Society Series B</i>,
 * <b>2010</b>, 72, 269-302.
 *
 * @anchor Gray2001
 * Gray, A. G. & Moore, A. W. `N-Body' Problems in Statistical
 * Learning. <i>Advances in Neural Information Processing Systems</i>,
 * <b>2001</b>, <i>13</i>.
 *
 * @anchor Haario2001
 * Haario, H.; Saksman, E. & Tamminen, J. An adaptive Metropolis algorithm.
 * <i>Bernoulli</i>, <b>2001</b>, 7, 223-242.
 *
 * @anchor Hairer1993
 * Hairer, E.; Norsett, S. N. & Wanner, G. Solving Ordinary Differential
 * Equations I: Nonstiff Problems. Springer-Verlag, <b>1993</b>.
 *
 * @anchor Jones2010
 * Jones, E.; Parslow, J. & Murray, L. A Bayesian approach to state and
 * parameter estimation in a Phytoplankton-Zooplankton model. <i>Australian
 * Meteorological and Oceanographic Journal</i>, <b>2010</b>, 59, 7-16.
 *
 * @anchor Kennedy2000
 * Kennedy, C. A.; Carpenter, M. H. & Lewis, R. M. Low-storage, explicit
 * Runge-Kutta schemes for the compressible Navier-Stokes equations.
 * <i>Applied Numerical Mathematics</i>, <b>2000</b>, 35, 177-219
 *
 * @anchor Kitagawa1996
 * Kitagawa, G. Monte Carlo %Filter and %Smoother for Non-Gaussian
 * Nonlinear %State Space Models. <i>Journal of Computational and
 * Graphical Statistics</i>, <b>1996</b>, 5, 1-25.
 *
 * @anchor Murray2011
 * Murray, L.M. GPU acceleration of Runge-Kutta integrators. <i>IEEE
 * Transactions on Parallel and Distributed Systems</i>, <b>2011</b>, to
 * appear.
 *
 * @anchor Murray2011a
 * Murray, L.M. GPU acceleration of the particle filter: The Metropolis
 * resampler. <i>DMMD: Distributed machine learning and sparse representation
 * with massive data sets</i>, <b>2011</b>.
 *
 * @anchor Murray2011b
 * Murray, L.M. & Storkey, A. Particle smoothing in continuous time: A fast
 * approach via density estimation. <i>IEEE Transactions on Signal
 * Processing</i>, <b>2011</b>, 59, 1017-1026.
 *
 * @anchor Pitt1999
 * Pitt, M. & Shephard, N. Filtering Via Simulation: Auxiliary Particle
 * Filters. <i>Journal of the American Statistical Association</i>,
 * <b>1999</b>, 94, 590-599.
 *
 * @anchor Pitt2002
 * Pitt, M. K. Smooth particle filters for likelihood evaluation and
 * maximisation. Technical Report 651, Department of Economics, The University
 * of Warwick, <b>2002</b>.
 *
 * @anchor Pitt2011
 * Pitt, M. K.; Silva, R. S.; Giordani, P. & Kohn, R. Auxiliary particle
 * filtering within adaptive Metropolis-Hastings sampling, <b>2010</b>.
 * http://arxiv.org/abs/1006.1914
 *
 * @anchor Sarkka2008
 * Särkkä, S. Unscented Rauch-Tung-Striebel Smoother. <i>IEEE Transactions on
 * Automated Control</i>, <b>2008</b>, 53, 845-849.
 *
 * @anchor Silverman1986
 * Silverman, B.W. <i>Density Estimation for Statistics and Data
 * Analysis</i>. Chapman and Hall, <b>1986</b>.
 */
