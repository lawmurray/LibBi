/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_SAMPLERSTATE_HPP
#define BI_STATE_SAMPLERSTATE_HPP

#include "../math/matrix.hpp"

namespace bi {
/**
 * State for samplers.
 *
 * @ingroup state
 *
 * @tparam B Model type.
 * @tparam L Location.
 * @tparam S1 Filter state type.
 * @tparam IO1 Filter cache type.
 */
template<class B, Location L, class S1, class IO1>
class SamplerState: public S1 {
public:
  /**
   * Host vector type.
   */
  typedef host_vector<real> host_vector_type;

  /**
   * Matrix type.
   */
  typedef typename loc_temp_matrix<L,real>::type matrix_type;

  /**
   * Constructor.
   *
   * @param m Model.
   * @param P Number of \f$x\f$-particles.
   * @param T Number of time points.
   */
  SamplerState(B& m, const int P = 0, const int T = 0);

  /**
   * Shallow copy constructor.
   */
  SamplerState(const SamplerState<B,L,S1,IO1>& o);

  /**
   * Assignment operator.
   */
  SamplerState& operator=(const SamplerState<B,L,S1,IO1>& o);

  /**
   * Swap.
   */
  void swap(SamplerState<B,L,S1,IO1>& o);

  /**
   * Filter output.
   */
  IO1 out;

  /**
   * Current state sample.
   */
  matrix_type path;

  /**
   * Marginal log-likelihood of parameters.
   */
  double logLikelihood;

  /**
   * Log-prior density of parameters.
   */
  double logPrior;

  /**
   * Log-proposal density of parameters.
   */
  double logProposal;

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
bi::SamplerState<B,L,S1,IO1>::SamplerState(B& m, const int P, const int T) :
    S1(P), out(m), path(B::NR + B::ND, T), logLikelihood(-1.0 / 0.0), logPrior(
        -1.0 / 0.0), logProposal(-1.0 / 0.0) {
  //
}

template<class B, bi::Location L, class S1, class IO1>
bi::SamplerState<B,L,S1,IO1>::SamplerState(const SamplerState<B,L,S1,IO1>& o) :
    S1(o), out(o.out), path(o.path), logLikelihood(o.logLikelihood), logPrior(
        o.logPrior), logProposal(o.logProposal) {
  //
}

template<class B, bi::Location L, class S1, class IO1>
bi::SamplerState<B,L,S1,IO1>& bi::SamplerState<B,L,S1,IO1>::operator=(
    const SamplerState<B,L,S1,IO1>& o) {
  S1::operator=(o);
  out = o.out;
  path = o.path;
  logLikelihood = o.logLikelihood;
  logPrior = o.logPrior;
  logProposal = o.logProposal;

  return *this;
}

template<class B, bi::Location L, class S1, class IO1>
void bi::SamplerState<B,L,S1,IO1>::swap(SamplerState<B,L,S1,IO1>& o) {
  S1::swap(o);
  out.swap(o.out);
  path.swap(o.path);
  std::swap(logLikelihood, o.logLikelihood);
  std::swap(logPrior, o.logPrior);
  std::swap(logProposal, o.logProposal);
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::SamplerState<B,L,S1,IO1>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < S1 > (*this);
  ar & out;
  save_resizable_matrix(ar, version, path);
  ar & logLikelihood;
  ar & logPrior;
  ar & logProposal;
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::SamplerState<B,L,S1,IO1>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < S1 > (*this);
  ar & out;
  load_resizable_matrix(ar, version, path);
  ar & logLikelihood;
  ar & logPrior;
  ar & logProposal;
}

#endif
