/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#error "Concept documentation only, should not be #included"

namespace concept {
/**
 * %Filter concept.
 *
 * @ingroup concept
 *
 * @note This is a phony class, representing a concept, for documentation
 * purposes only.
 */
struct Filter {
  /**
   * %Filter forward.
   *
   * @param tnxt Time to which to filter.
   */
  void filter(const real tnxt) = 0;

  /**
   * Filter forward conditioned on trajectory.
   *
   * @tparam M1 Matrix type.
   *
   * @param tnxt Time to which to filter.
   * @param xd Trajectory of d-nodes.
   * @param xc Trajectory of c-nodes.
   *
   * @p xc and @p xd are matrices where rows index variables and
   * columns index times. This method performs a <em>conditional</em> particle
   * filter as described in @ref Andrieu2010
   * "Andrieu, Doucet \& Holenstein (2010)".
   */
  template<class M1>
  void filter(const real tnxt, const M1& xd, const M1& xc) = 0;

  /**
   * Filter forward conditioned on starting state and stochastic trajectory.
   *
   * @tparam V1 Vector type.
   * @tparam M1 Matrix type.
   *
   * @param tnxt Time to which to filter.
   * @param xd0 Starting state of d-nodes in conditioned trajectory.
   * @param xc0 Starting state of c-nodes in conditioned trajectory.
   * @param xr Trajectory of r-nodes.
   *
   * @p xc and @p xd are vectors indexing variables, and @p xr is a matrix
   * where rows index variables and* columns index times.
   */
  template<class V1, class M1>
  void filter(const real tnxt, const V1& xd0, const V1& xc0, const M1& xr) = 0;

  /**
   * Compute summary information from last filter run.
   *
   * @tparam T1 Scalar type.
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param[out] ll Marginal log-likelihood.
   * @param[out] lls Log-likelihood at each observation.
   * @param[out] ess Effective sample size at each observation.
   *
   * Any number of the parameters may be @c NULL.
   */
  template<class T1, class V1, class V2>
  void summarise(T1* ll, V1* lls, V2* ess) = 0;

  /**
   * Read single particle trajectory.
   *
   * @tparam M1 Matrix type.
   *
   * @param[out] xd Trajectory of d-nodes.
   * @param[out] xc Trajectory of c-nodes.
   * @param[out] xr Trajectory of r-nodes.
   *
   * Reads a single particle trajectory from the smooth distribution.
   *
   * On output, @p xd, @p xc and @p xr are arranged such that rows index
   * variables, and columns index time points.
   */
  template<class M1>
  void drawTrajectory(M1& xd, M1& xc, M1& xr) = 0;

  /**
   * Reset filter for reuse.
   */
  void reset() = 0;

  /**
   * Get output buffer.
   *
   * @return The output buffer. NULL if there is no output.
   */
  IO3* getOutput() = 0;

};
}
