/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_MARGINALSIRSTATE_HPP
#define BI_STATE_MARGINALSIRSTATE_HPP

#include <vector>

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
  static const Location location = L;
  static const bool on_device = (L == ON_DEVICE);

  typedef real value_type;
  typedef typename loc_vector<L,value_type>::type vector_type;
  typedef typename vector_type::vector_reference_type vector_reference_type;

  typedef int int_value_type;
  typedef typename loc_vector<L,int_value_type>::type int_vector_type;
  typedef typename int_vector_type::vector_reference_type int_vector_reference_type;

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
   * Number of \f$\theta\f$-particles.
   */
  int size() const;

  /**
   * Log-weights vector.
   */
  vector_reference_type logWeights();

  /**
   * Log-weights vector.
   */
  const vector_reference_type logWeights() const;

  /**
   * Ancestors vector.
   */
  int_vector_reference_type ancestors();

  /**
   * Ancestors vector.
   */
  const int_vector_reference_type ancestors() const;

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
  vector_type lws;

  /**
   * Ancestors.
   */
  int_vector_type as;

  /**
   * Incremental log-likelihood.
   */
  real incLogLikelihood;

  /**
   * Index of starting \f$\theta\f$-particle.
   */
  int ptheta;

  /**
   * Number of \f$\theta\f$-particles.
   */
  int Ptheta;

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
    sFilters(Ptheta), outFilters(Ptheta), lws(Ptheta), as(Ptheta), incLogLikelihood(
        -1.0 / 0.0), ptheta(0), Ptheta(Ptheta) {
  int p;
  for (p = 0; p < Ptheta; ++p) {
    sFilters[p] = new S1(Px, T);
  }
  for (p = 0; p < Ptheta; ++p) {
    outFilters[p] = new IO1(m);
  }
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalSIRState<B,L,S1,IO1>::MarginalSIRState(
    const MarginalSIRState<B,L,S1,IO1>& o) :
    sFilters(o.sFilters.size()), outFilters(o.outFilters.size()), lws(o.lws), as(
        o.as), incLogLikelihood(o.incLogLikelihood), ptheta(o.ptheta), Ptheta(
        o.Ptheta) {
  int p;
  for (p = 0; p < Ptheta; ++p) {
    sFilters[p] = new S1(*o.sFilters[p]);
  }
  for (p = 0; p < Ptheta; ++p) {
    outFilters[p] = new IO1(*o.outFilters[p]);
  }
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalSIRState<B,L,S1,IO1>& bi::MarginalSIRState<B,L,S1,IO1>::operator=(
    const MarginalSIRState<B,L,S1,IO1>& o) {
  /* pre-condition */
  BI_ASSERT(o.size() == size());

  lws = o.lws;
  as = o.as;
  incLogLikelihood = o.incLogLikelihood;
  ptheta = o.ptheta;
  Ptheta = o.Ptheta;

  int p;
  for (p = 0; p < Ptheta; ++p) {
    *sFilters[p] = *o.sFilters[p];
  }
  for (p = 0; p < Ptheta; ++p) {
    *outFilters[p] = *o.outFilters[p];
  }
  return *this;
}

template<class B, bi::Location L, class S1, class IO1>
int bi::MarginalSIRState<B,L,S1,IO1>::size() const {
  return Ptheta;
}

template<class B, bi::Location L, class S1, class IO1>
typename bi::MarginalSIRState<B,L,S1,IO1>::vector_reference_type bi::MarginalSIRState<
    B,L,S1,IO1>::logWeights() {
  return subrange(lws, ptheta, Ptheta);
}

template<class B, bi::Location L, class S1, class IO1>
const typename bi::MarginalSIRState<B,L,S1,IO1>::vector_reference_type bi::MarginalSIRState<
    B,L,S1,IO1>::logWeights() const {
  return subrange(lws, ptheta, Ptheta);
}

template<class B, bi::Location L, class S1, class IO1>
typename bi::MarginalSIRState<B,L,S1,IO1>::int_vector_reference_type bi::MarginalSIRState<
    B,L,S1,IO1>::ancestors() {
  return subrange(as, ptheta, Ptheta);
}

template<class B, bi::Location L, class S1, class IO1>
const bi::MarginalSIRState<B,L,S1,IO1>::int_vector_reference_type bi::MarginalSIRState<
    B,L,S1,IO1>::ancestors() const {
  return subrange(as, ptheta, Ptheta);
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalSIRState<B,L,S1,IO1>::save(Archive& ar,
    const unsigned version) const {
  save_resizable_vector(ar, version, lws);
  save_resizable_vector(ar, version, as);

  int p;
  for (p = 0; p < sFilters.size(); ++p) {
    ar & *sFilters[p];
  }
  for (p = 0; p < outFilters.size(); ++p) {
    ar & *outFilters[p];
  }
  ar & incLogLikelihood & ptheta & Ptheta;
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalSIRState<B,L,S1,IO1>::load(Archive& ar,
    const unsigned version) {
  load_resizable_vector(ar, version, lws);
  load_resizable_vector(ar, version, as);

  int p;
  for (p = 0; p < sFilters.size(); ++p) {
    ar & *sFilters[p];
  }
  for (p = 0; p < outFilters.size(); ++p) {
    ar & *outFilters[p];
  }
  ar & incLogLikelihood & ptheta & Ptheta;
}

#endif
