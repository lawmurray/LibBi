/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_MARGINALSIRSTATE_HPP
#define BI_STATE_MARGINALSIRSTATE_HPP

namespace bi {
/**
 * State for MarginalSIR.
 *
 * @ingroup state
 *
 * @tparam B Model type.
 * @tparam L Location.
 * @tparam S1 Filter state type.
 * @tparam IO1 Filter cache type.
 */
template<class B, Location L, class S1, class IO1>
class MarginalSIRState {
public:
  /**
   * Constructor.
   *
   * @param m Model.
   * @param Ptheta Number of \f$\theta\f$-particles.
   * @param Px Number of \f$x\f$-particles.
   * @param T Number of time points.
   */
  MarginalSIRState(B& m, const int Ptheta = 0, const int Px = 0, const int T =
      0);

  /**
   * Shallow copy constructor.
   */
  MarginalSIRState(const MarginalSIRState<B,L,S1,IO1>& o);

  /**
   * Deep assignment operator.
   */
  MarginalSIRState& operator=(const MarginalSIRState<B,L,S1,IO1>& o);

  /**
   * Log-weights vector.
   */
  typename State<B,L>::vector_reference_type logWeights();

  /**
   * Log-weights vector.
   */
  const typename State<B,L>::vector_reference_type logWeights() const;

  /**
   * Ancestors vector.
   */
  typename State<B,L>::int_vector_reference_type ancestors();

  /**
   * Ancestors vector.
   */
  const typename State<B,L>::int_vector_reference_type ancestors() const;

  /**
   * Filter states.
   */
  std::vector<S1*> sFilters;

  /**
   * Filter outputs.
   */
  std::vector<IO1*> outFilters;

private:
  /**
   * Log-weights.
   */
  typename State<B,L>::vector_type lws;

  /**
   * Ancestors.
   */
  typename State<B,L>::int_vector_type as;

  /**
   * Incremental log-likelihood.
   */
  real incLogLikelihood;

  /**
   * Index of starting \f$\theta\f$-particle.
   */
  int p;

  /**
   * Number of \f$\theta\f$-particles.
   */
  int P;

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
bi::MarginalSIRState<B,L,S1,IO1>::MarginalSIRState(B& m, const int Ptheta,
    const int Px, const int T) :
    lws(Ptheta), as(Ptheta), incLogLikelihood(-1.0 / 0.0), p(0), P(Ptheta) {
  sFilters.reserve(Ptheta);
  outFilters.reserve(Ptheta);

  int i;
  for (i = 0; i < Ptheta; ++i) {
    sFilters[i] = new S1(Px, T);
  }
  for (i = 0; i < Ptheta; ++i) {
    outFilters[i] = new IO1(m);
  }
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalSIRState<B,L,S1,IO1>::MarginalSIRState(
    const MarginalSIRState<B,L,S1,IO1>& o) :
    lws(o.lws), as(o.as), incLogLikelihood(o.incLogLikelihood), p(o.p), P(o.P) {
  sFilters.reserve(o.sFilters.size());
  outFilters.reserve(o.outFilters.size());

  int i;
  for (i = 0; i < sFilters.size(); ++i) {
    sFilters[i] = new S1(*o.sFilters[i]);
  }
  for (i = 0; i < outFilters.size(); ++i) {
    outFilters[i] = new IO1(*o.outFilters[i]);
  }
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalSIRState<B,L,S1,IO1>& bi::MarginalSIRState<B,L,S1,IO1>::operator=(
    const MarginalSIRState<B,L,S1,IO1>& o) {
  lws = o.lws;
  as = o.as;
  incLogLikelihood = o.incLogLikelihood;
  p = o.p;
  P = o.P;

  int i;
  for (i = 0; i < sFilters.size(); ++i) {
    *sFilters[i] = *o.sFilters[i];
  }
  for (i = 0; i < outFilters.size(); ++i) {
    *outFilters[i] = *o.outFilters[i];
  }
  return *this;
}

template<class B, bi::Location L, class S1, class IO1>
typename bi::State<B,L>::vector_reference_type bi::MarginalSIRState<B,L,S1,IO1>::logWeights() {
  return subrange(lws, p, P);
}

template<class B, bi::Location L, class S1, class IO1>
const typename bi::State<B,L>::vector_reference_type bi::MarginalSIRState<B,L,
    S1,IO1>::logWeights() const {
  return subrange(lws, p, P);
}

template<class B, bi::Location L, class S1, class IO1>
typename bi::State<B,L>::int_vector_reference_type bi::MarginalSIRState<B,L,
    S1,IO1>::ancestors() {
  return subrange(as, p, P);
}

template<class B, bi::Location L, class S1, class IO1>
const typename bi::State<B,L>::int_vector_reference_type bi::MarginalSIRState<
    B,L,S1,IO1>::ancestors() const {
  return subrange(as, p, P);
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalSIRState<B,L,S1,IO1>::save(Archive& ar,
    const unsigned version) const {
  save_resizable_vector(ar, version, lws);
  save_resizable_vector(ar, version, as);

  int i;
  for (i = 0; i < sFilters.size(); ++i) {
    ar & *sFilters[i];
  }
  for (i = 0; i < outFilters.size(); ++i) {
    ar & *outFilters[i];
  }

  ar & incLogLikelihood & p & P;
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalSIRState<B,L,S1,IO1>::load(Archive& ar,
    const unsigned version) {
  load_resizable_vector(ar, version, lws);
  load_resizable_vector(ar, version, as);

  int i;
  for (i = 0; i < sFilters.size(); ++i) {
    ar & *sFilters[i];
  }
  for (i = 0; i < outFilters.size(); ++i) {
    ar & *outFilters[i];
  }

  ar & incLogLikelihood & p & P;
}

#endif
