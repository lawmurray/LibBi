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
 * @defgroup math Math
 *
 *   @defgroup math_matvec Matrix and vector containers
 *   @ingroup math
 *
 *   @defgroup math_view Matrix and vector views
 *   @ingroup math
 *
 *   @defgroup math_op Matrix and vector operations
 *   @ingroup math
 *
 *   @defgroup math_multi_op Matrix and vector batch operations
 *   @ingroup math
 *
 *   These operations are BLAS-like operations that are performed on multiple
 *   input arguments simultaneously. Each accepts a @c P argument indicating
 *   the number of operations to be performed. Vector arguments are passed
 *   as matrices of @c P rows, where each row gives a vector. Matrix
 *   arguments are passed as matrices with @c P times as many rows as usual,
 *   with rows of each of the @c P matrices interleaved.
 *
 *   @defgroup math_ode Ordinary differential equations
 *   @ingroup math
 *
 *   @defgroup math_rng Random number generation
 *   @ingroup math
 *
 *   @defgroup math_pdf Probability density functions
 *   @ingroup math
 *
 * @defgroup primitive Primitives
 *
 *   @defgroup primitive_functor Functors
 *   @ingroup primitive
 *
 *   @defgroup primitive_iterators Iterators
 *   @ingroup primitive
 *
 *   @defgroup primitive_vector Vector primitives
 *   @ingroup primitive
 *
 *   @defgroup primitive_matrix Matrix primitives
 *   @ingroup primitive
 *
 *   @defgroup primitive_allocators Allocators
 *   @ingroup primitive
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
 *   @defgroup io_buffer Buffers
 *   @ingroup io
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
 */
