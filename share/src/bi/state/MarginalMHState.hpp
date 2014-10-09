/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_MARGINALMHSTATE_HPP
#define BI_STATE_MARGINALMHSTATE_HPP

#include "../state/SamplerState.hpp"

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
   * State type.
   */
  typedef SamplerState<B,L,S1,IO1> state_type;

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
   * Current state.
   */
  state_type theta1;

  /**
   * Proposed state.
   */
  state_type theta2;

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
    theta1(m, P, Y, T), theta2(m, P, Y, T) {
  //
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalMHState<B,L,S1,IO1>::MarginalMHState(
    const MarginalMHState<B,L,S1,IO1>& o) :
    theta1(o.theta1), theta2(o.theta2) {
  //
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalMHState<B,L,S1,IO1>& bi::MarginalMHState<B,L,S1,IO1>::operator=(
    const MarginalMHState<B,L,S1,IO1>& o) {
  theta1 = o.theta1;
  theta2 = o.theta2;

  return *this;
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalMHState<B,L,S1,IO1>::save(Archive& ar,
    const unsigned version) const {
  ar & theta1;
  ar & theta2;
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalMHState<B,L,S1,IO1>::load(Archive& ar,
    const unsigned version) {
  ar & theta1;
  ar & theta2;
}

#endif
