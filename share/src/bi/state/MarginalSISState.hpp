/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_MARGINALSISSTATE_HPP
#define BI_STATE_MARGINALSISSTATE_HPP

#include "SamplerState.hpp"

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
 * @tparam Q1 Proposal type.
 */
template<class B, Location L, class S1, class IO1, class Q1>
class MarginalSISState: public SamplerState<B,L,S1,IO1> {
public:
  typedef SamplerState<B,L,S1,IO1> parent_type;

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
  MarginalSISState(const MarginalSISState<B,L,S1,IO1,Q1>& o);

  /**
   * Assignment operator.
   */
  MarginalSISState& operator=(const MarginalSISState<B,L,S1,IO1,Q1>& o);

  /**
   * Proposal distribution.
   */
  Q1 q;

  /*
   * Incremental log-likelihoods.
   */
  host_vector<double> logLikelihoods;

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

template<class B, bi::Location L, class S1, class IO1, class Q1>
bi::MarginalSISState<B,L,S1,IO1,Q1>::MarginalSISState(B& m, const int P,
    const int Y, const int T) :
    parent_type(m, P, Y, T), q(B::NP), logLikelihoods(Y) {
  //
}

template<class B, bi::Location L, class S1, class IO1, class Q1>
bi::MarginalSISState<B,L,S1,IO1,Q1>::MarginalSISState(
    const MarginalSISState<B,L,S1,IO1,Q1>& o) :
    parent_type(o), q(o.q), logLikelihoods(o.logLikelihoods) {
  //
}

template<class B, bi::Location L, class S1, class IO1, class Q1>
bi::MarginalSISState<B,L,S1,IO1,Q1>& bi::MarginalSISState<B,L,S1,IO1,Q1>::operator=(
    const MarginalSISState<B,L,S1,IO1,Q1>& o) {
  parent_type::operator=(o);
  q = o.q;
  logLikelihoods = o.logLikelihoods;

  return *this;
}

template<class B, bi::Location L, class S1, class IO1, class Q1>
template<class Archive>
void bi::MarginalSISState<B,L,S1,IO1,Q1>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < parent_type > (*this);
  ar & q;
  load_resizable_vector(ar, version, logLikelihoods);
}

template<class B, bi::Location L, class S1, class IO1, class Q1>
template<class Archive>
void bi::MarginalSISState<B,L,S1,IO1,Q1>::load(Archive& ar,
    const unsigned version) {
  ar & boost::serialization::base_object < parent_type > (*this);
  ar & q;
  save_resizable_vector(ar, version, logLikelihoods);
}

#endif
