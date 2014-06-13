/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_MARGINALSRSSTATE_HPP
#define BI_STATE_MARGINALSRSSTATE_HPP

#include "SamplerState.hpp"

namespace bi {
/**
 * State for MarginalSRS.
 *
 * @ingroup state
 *
 * @tparam B Model type.
 * @tparam L Location.
 * @tparam S1 Filter state type.
 * @tparam IO1 Output type.
 */
template<class B, Location L, class S1, class IO1>
class MarginalSRSState: public SamplerState<B,L,S1,IO1> {
public:
  typedef SamplerState<B,L,S1,IO1> parent_type;

  /**
   * Constructor.
   *
   * @param m Model.
   * @param P Number of particles.
   * @param T Number of time points.
   */
  MarginalSRSState(B& m, const int P = 0, const int T = 0);

  /**
   * Shallow copy constructor.
   */
  MarginalSRSState(const MarginalSRSState<B,L,S1,IO1>& o);

  /**
   * Assignment operator.
   */
  MarginalSRSState& operator=(const MarginalSRSState<B,L,S1,IO1>& o);

  /**
   * Weight thresholds for each time.
   */
  typename State<B,L>::vector_type omegas;

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
bi::MarginalSRSState<B,L,S1,IO1>::MarginalSRSState(B& m, const int P,
    const int T) :
    parent_type(m, P, T), omegas(T) {
  //
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalSRSState<B,L,S1,IO1>::MarginalSRSState(
    const MarginalSRSState<B,L,S1,IO1>& o) :
    parent_type(o), omegas(o.omegas) {
  //
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalSRSState<B,L,S1,IO1>& bi::MarginalSRSState<B,L,S1,IO1>::operator=(
    const MarginalSRSState<B,L,S1,IO1>& o) {
  parent_type::operator=(o);
  omegas = o.omegas;

  return *this;
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalSRSState<B,L,S1,IO1>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < parent_type > (*this);
  save_resizable_vector(ar, version, omegas);
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalSRSState<B,L,S1,IO1>::load(Archive& ar,
    const unsigned version) {
  ar & boost::serialization::base_object < parent_type > (*this);
  load_resizable_vector(ar, version, omegas);
}

#endif
