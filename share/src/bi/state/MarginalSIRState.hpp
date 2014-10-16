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
 * @tparam IO1 Output type.
 */
template<class B, Location L, class S1, class IO1>
class MarginalSIRState {
public:
  static const Location location = L;
  static const bool on_device = (L == ON_DEVICE);

  typedef real value_type;
  typedef host_vector<value_type> vector_type;
  typedef typename vector_type::vector_reference_type vector_reference_type;

  typedef int int_value_type;
  typedef host_vector<int_value_type> int_vector_type;
  typedef typename int_vector_type::vector_reference_type int_vector_reference_type;

  /**
   * Constructor.
   *
   * @param m Model.
   * @param Ptheta Number of \f$\theta\f$-particles.
   * @param Px Number of \f$x\f$-particles.
   * @param Y Number of observation times.
   * @param T Number of output times.
   */
  MarginalSIRState(B& m, const int Ptheta = 0, const int Px = 0, const int Y =
      0, const int T = 0);

  /**
   * Shallow copy constructor.
   */
  MarginalSIRState(const MarginalSIRState<B,L,S1,IO1>& o);

  /**
   * Deep assignment operator.
   */
  MarginalSIRState& operator=(const MarginalSIRState<B,L,S1,IO1>& o);

  /**
   * Clear.
   */
  void clear();

  /**
   * Swap.
   */
  void swap(MarginalSIRState<B,L,S1,IO1>& o);

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
   * \f$\theta\f$-particles.
   */
  std::vector<S1*> s1s;

  /**
   * Output buffers.
   */
  std::vector<IO1*> out1s;

  /**
   * Proposed state.
   */
  S1 s2;

  /**
   * Proposed output.
   */
  IO1 out2;

  /**
   * Log-evidence.
   */
  double logEvidence;

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
    const int Px, const int Y, const int T) :
    s1s(Ptheta), out1s(Ptheta), s2(m, Px, Y, T), out2(m, Px, T), logEvidence(
        0.0), lws(Ptheta), as(Ptheta), ptheta(0), Ptheta(Ptheta) {
  for (int p = 0; p < size(); ++p) {
    s1s[p] = new S1(m, Px, Y, T);
    out1s[p] = new IO1(m, Px, T);
  }
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalSIRState<B,L,S1,IO1>::MarginalSIRState(
    const MarginalSIRState<B,L,S1,IO1>& o) :
    s1s(o.s1s.size()), out1s(o.out1s.size()), s2(o.s2), out2(o.out2), logEvidence(
        o.logEvidence), lws(o.lws), as(o.as), ptheta(o.ptheta), Ptheta(
        o.Ptheta) {
  for (int p = 0; p < size(); ++p) {
    s1s[p] = new S1(*o.s1s[p]);
    out1s[p] = new IO1(*o.out1s[p]);
  }
}

template<class B, bi::Location L, class S1, class IO1>
bi::MarginalSIRState<B,L,S1,IO1>& bi::MarginalSIRState<B,L,S1,IO1>::operator=(
    const MarginalSIRState<B,L,S1,IO1>& o) {
  /* pre-condition */
  BI_ASSERT(o.size() == size());

  for (int p = 0; p < size(); ++p) {
    *s1s[p] = *o.s1s[p];
    *out1s[p] = *o.out1s[p];
  }
  s2 = o.s2;
  out2 = o.out2;
  logEvidence = o.logEvidence;
  lws = o.lws;
  as = o.as;
  ptheta = o.ptheta;
  Ptheta = o.Ptheta;

  return *this;
}

template<class B, bi::Location L, class S1, class IO1>
void bi::MarginalSIRState<B,L,S1,IO1>::clear() {
  for (int p = 0; p < size(); ++p) {
    s1s[p]->clear();
    out1s[p]->clear();
  }
  s2.clear();
  out2.clear();
  logEvidence = 0.0;
  logWeights().clear();
  seq_elements(ancestors(), 0);
}

template<class B, bi::Location L, class S1, class IO1>
void bi::MarginalSIRState<B,L,S1,IO1>::swap(MarginalSIRState<B,L,S1,IO1>& o) {
  std::swap(s1s, o.s1s);
  std::swap(out1s, o.out1s);
  s2.swap(o.s2);
  out2.swap(o.out2);
  std::swap(logEvidence, o.logEvidence);
  lws.swap(o.lws);
  as.swap(o.as);
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
const typename bi::MarginalSIRState<B,L,S1,IO1>::int_vector_reference_type bi::MarginalSIRState<
    B,L,S1,IO1>::ancestors() const {
  return subrange(as, ptheta, Ptheta);
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalSIRState<B,L,S1,IO1>::save(Archive& ar,
    const unsigned version) const {
  for (int p = 0; p < size(); ++p) {
    ar & *s1s[p];
    ar & *out1s[p];
  }
  ar & s2;
  ar & out2;
  ar & logEvidence;
  save_resizable_vector(ar, version, lws);
  save_resizable_vector(ar, version, as);
  ar & ptheta;
  ar & Ptheta;
}

template<class B, bi::Location L, class S1, class IO1>
template<class Archive>
void bi::MarginalSIRState<B,L,S1,IO1>::load(Archive& ar,
    const unsigned version) {
  for (int p = 0; p < size(); ++p) {
    ar & *s1s[p];
    ar & *out1s[p];
  }
  ar & s2;
  ar & out2;
  ar & logEvidence;
  load_resizable_vector(ar, version, lws);
  load_resizable_vector(ar, version, as);
  ar & ptheta;
  ar & Ptheta;
}

#endif
