/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_MARGINALMHSTATE_HPP
#define BI_STATE_MARGINALMHSTATE_HPP

namespace bi {
/**
 * State for MarginalMH.
 *
 * @ingroup state
 *
 * @tparam B Model type.
 * @tparam L Location.
 * @tparam S1 Filter state type.
 * @tparam IO1 Filter cache type.
 */
template<class B, Location L, class S1, class IO1>
class MarginalMHState {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param P Number of \f$x\f$-particles.
   * @param Y Number of observation times.
   * @param T Number of output times.
   */
  MarginalMHState(B& m, const int P = 0, const int Y = 0, const int T = 0);

  /**
   * Shallow copy constructor.
   */
  MarginalMHState(const MarginalMHState<B,L,S1,IO1>& o);

  /**
   * Assignment operator.
   */
  MarginalMHState& operator=(const MarginalMHState<B,L,S1,IO1>& o);

  /**
   * Clear.
   */
  void clear();

  /**
   * Swap.
   */
  void swap(MarginalMHState<B,L,S1,IO1>& o);

  /**
   * Current state.
   */
  S1 s1;

  /**
   * Proposed state.
   */
  S1 s2;

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

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalMHState<B,L,S1,IO1>::MarginalMHState(B& m, const int P, const int Y,
    const int T) :
    s1(P, Y, T), s2(P, Y, T), out(m, P, T) {
  //
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalMHState<B,L,S1,IO1>::MarginalMHState(
    const MarginalMHState<B,L,S1,IO1>& o) :
    s1(o.s1), s2(o.s2), out(o.out) {
  //
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalMHState<B,L,S1,IO1>& bi::MarginalMHState<B,L,S1,IO1>::operator=(
    const MarginalMHState<B,L,S1,IO1>& o) {
  s1 = o.s1;
  s2 = o.s2;
  out = o.out;

  return *this;
}

template<class B, bi::Location L, class S1, class IO1>
void bi::MarginalMHState<B,L,S1,IO1>::clear() {
  s1.clear();
  s2.clear();
  out.clear();
}

template<class B, bi::Location L, class S1, class IO1>
void bi::MarginalMHState<B,L,S1,IO1>::swap(MarginalMHState<B,L,S1,IO1>& o) {
  s1.swap(o.s1);
  s2.swap(o.s2);
  out.swap(o.out);
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalMHState<B,L,S1,IO1>::save(Archive& ar,
    const unsigned version) const {
  ar & s1;
  ar & s2;
  ar & out;
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalMHState<B,L,S1,IO1>::load(Archive& ar,
    const unsigned version) {
  ar & s1;
  ar & s2;
  ar & out;
}

#endif
