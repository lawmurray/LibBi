/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#error "Concept documentation only, should not be #included"

#include "../random/Random.hpp"
#include "../state/State.hpp"
#include "../buffer/Mask.hpp"
#include "../misc/location.hpp"

namespace concept {
/**
 * Block.
 *
 * @ingroup
 */
class Block {
public:
  /**
   * @name Static updates
   */
  //@{
  /**
   * Static deterministic update.
   *
   * @tparam L Location.
   *
   * @param[in,out] s State.
   */
  template<Location L>
  static void simulate(State<B,L>& s);

  /**
   * Static stochastic update.
   *
   * @tparam L Location.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   */
  template<Location L>
  static void samples(Random& rng, State<B,L>& s);

  /**
   * Static log-density update and static deterministic update.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param[in,out] lp Log-densities.
   */
  template<Location L, class V1>
  static void logDensities(State<B,L>& s, V1 lp);

  /**
   * Static maximum log-density update and static deterministic update.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param[in,out] lp Log-densities.
   */
  template<Location L, class V1>
  static void maxLogDensities(State<B,L>& s, V1 lp);
  //@}

  /**
   * @name Dynamic updates
   */
  //@{
  /**
   * Dynamic deterministic update.
   *
   * @tparam T1 Scalar type.
   * @tparam L Location.
   *
   * @param t1 Start of time interval.
   * @param t2 End of time interval.
   * @param onDelta Is the starting time an integer multiple of the time
   * step?
   * @param[in,out] s State.
   */
  template<class T1, Location L>
  static void simulate(const T1 t1, const T1 t2, const bool onDelta,
      State<B,L>& s);

  /**
   * Dynamic stochastic update.
   *
   * @tparam T1 Scalar type.
   * @tparam L Location.
   *
   * @param[in,out] rng Random number generator.
   * @param t1 Start of time interval.
   * @param t2 End of time interval.
   * @param onDelta Is the starting time an integer multiple of the time
   * step?
   * @param[in,out] s State.
   */
  template<class T1, Location L>
  static void samples(Random& rng, const T1 t1, const T1 t2,
      const bool onDelta, State<B,L>& s);

  /**
   * Dynamic log-density update and dynamic deterministic update.
   *
   * @tparam T1 Scalar type.
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param t1 Start of time interval.
   * @param t2 End of time interval.
   * @param onDelta Is the starting time an integer multiple of the time
   * step?
   * @param[in,out] s State.
   * @param[in,out] lp Log-density.
   */
  template<class T1, Location L, class V1>
  static void logDensities(const T1 t1, const T1 t2, const bool onDelta,
      State<B,L>& s, V1 lp);

  /**
   * Dynamic maximum log-density update and dynamic deterministic update.
   *
   * @tparam T1 Scalar type.
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param t1 Start of time interval.
   * @param t2 End of time interval.
   * @param onDelta Is the starting time an integer multiple of the time
   * step?
   * @param[in,out] s State.
   * @param[in,out] lp Log-density.
   */
  template<class T1, Location L, class V1>
  static void maxLogDensities(const T1 t1, const T1 t2, const bool onDelta,
      State<B,L>& s, V1 lp);
  //@}

  /**
   * @name Sparse-static updates
   */
  //@{
  /**
   * Sparse-static deterministic update.
   *
   * @tparam L Location.
   *
   * @param[in,out] s State.
   * @param mask Mask.
   */
  template<Location L>
  static void simulate(State<B,L>& s, const Mask<L>& mask);

  /**
   * Sparse-static stochastic update.
   *
   * @tparam L Location.
   *
   * @param[in,out] rng Random number generator.
   * @param[in,out] s State.
   * @param mask Mask.
   */
  template<Location L>
  static void samples(Random& rng, State<B,L>& s, const Mask<L>& mask);

  /**
   * Sparse-static log-density update and sparse-static deterministic update.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param mask Mask.
   * @param[in,out] lp Log-densities.
   */
  template<Location L, class V1>
  static void logDensities(State<B,L>& s, const Mask<L>& mask, V1 lp);

  /**
   * Sparse-static maximum log-density update and sparse-static deterministic
   * update.
   *
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param[in,out] s State.
   * @param mask Mask.
   * @param[in,out] lp Log-densities.
   */
  template<Location L, class V1>
  static void maxLogDensities(State<B,L>& s, const Mask<L>& mask, V1 lp);
  //@}
};
}

#endif
