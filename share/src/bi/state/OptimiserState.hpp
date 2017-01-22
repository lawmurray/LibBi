/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_OPTIMISERSTATE_HPP
#define BI_STATE_OPTIMISERSTATE_HPP

namespace bi {
/**
 * State for Optimiser.
 *
 * @ingroup state
 *
 * @tparam B Model type.
 * @tparam L Location.
 * @tparam S Filter state type.
 * @tparam IO1 Filter cache type.
 */
template<class B, Location L, class S, class IO1>
class OptimiserState {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param P Number of \f$x\f$-particles.
   * @param Y Number of observation times.
   * @param T Number of output times.
   */
  OptimiserState(B& m, const int P = 0, const int Y = 0, const int T = 0);

  /**
   * Shallow copy constructor.
   */
  OptimiserState(const OptimiserState<B,L,S,IO1>& o);

  /**
   * Assignment operator.
   */
  OptimiserState& operator=(const OptimiserState<B,L,S,IO1>& o);

  /**
   * Clear.
   */
  void clear();

  /**
   * Current state.
   */
  S s;

  /**
   * Filter output.
   */
  IO1 out;

  /**
   * Execution time.
   */
  long clock;

private:
  /**
   * Serialize.
   */
  template<class Archive>
  void save(Archive& ar, const unsigned version) const;

  /**
   * Restore from serialization.
   */
  template<class Archive>
  void load(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  friend class boost::serialization::access;
};
}

template<class B, bi::Location L, class S, class IO1>
bi::OptimiserState<B,L,S,IO1>::OptimiserState(B& m, const int P, const int Y,
    const int T) :
    s(P, Y, T), out(m, P, T) {
  //
}

template<class B, bi::Location L, class S, class IO1>
bi::OptimiserState<B,L,S,IO1>::OptimiserState(
    const OptimiserState<B,L,S,IO1>& o) :
    s(o.s), out(o.out) {
  //
}

template<class B, bi::Location L, class S, class IO1>
bi::OptimiserState<B,L,S,IO1>& bi::OptimiserState<B,L,S,IO1>::operator=(
    const OptimiserState<B,L,S,IO1>& o) {
  s = o.s;
  out = o.out;

  return *this;
}

template<class B, bi::Location L, class S, class IO1>
void bi::OptimiserState<B,L,S,IO1>::clear() {
  s.clear();
  out.clear();
}

template<class B, bi::Location L, class S, class IO1>
template<class Archive>
void bi::OptimiserState<B,L,S,IO1>::save(Archive& ar,
    const unsigned version) const {
  ar & s;
  ar & out;
}

template<class B, bi::Location L, class S, class IO1>
template<class Archive>
void bi::OptimiserState<B,L,S,IO1>::load(Archive& ar,
    const unsigned version) {
  ar & s;
  ar & out;
}

#endif
