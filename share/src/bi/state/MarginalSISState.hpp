/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_MARGINALSISSTATE_HPP
#define BI_STATE_MARGINALSISSTATE_HPP

#include "MarginalMHState.hpp"

namespace bi {
/**
 * State for MarginalSIS.
 *
 * @ingroup state
 *
 * @tparam B Model type.
 * @tparam L Location.
 * @tparam S1 Filter state type.
 * @tparam IO1 Output type.
 */
template<class B, Location L, class S1, class IO1>
class MarginalSISState: public MarginalMHState<B,L,S1,IO1> {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param P Number of particles.
   * @param Y Number of observation times.
   * @param T Number of output times.
   */
  MarginalSISState(B& m, const int P = 0, const int Y = 0, const int T = 0);

  /**
   * Shallow copy constructor.
   */
  MarginalSISState(const MarginalSISState<B,L,S1,IO1>& o);

  /**
   * Assignment operator.
   */
  MarginalSISState& operator=(const MarginalSISState<B,L,S1,IO1>& o);

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
bi::MarginalSISState<B,L,S1,IO1>::MarginalSISState(B& m, const int P,
    const int Y, const int T) :
    MarginalMHState<B,L,S1,IO1>(m, P, Y, T) {
  //
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalSISState<B,L,S1,IO1>::MarginalSISState(
    const MarginalSISState<B,L,S1,IO1>& o) :
    MarginalMHState<B,L,S1,IO1>(o) {
  //
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalSISState<B,L,S1,IO1>& bi::MarginalSISState<B,L,S1,IO1>::operator=(
    const MarginalSISState<B,L,S1,IO1>& o) {
  MarginalMHState<B,L,S1,IO1>::operator=(o);
  return *this;
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalSISState<B,L,S1,IO1>::save(Archive& ar,
    const unsigned version) const {
  ar
      & boost::serialization::base_object < MarginalMHState<B,L,S1,IO1>
          > (*this);
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalSISState<B,L,S1,IO1>::load(Archive& ar,
    const unsigned version) {
  ar
      & boost::serialization::base_object < MarginalMHState<B,L,S1,IO1>
          > (*this);
}

#endif
