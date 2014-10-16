/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_MARGINALSISSTATE_HPP
#define BI_STATE_MARGINALSISSTATE_HPP


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
class MarginalSISState: public S1 {
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
  MarginalSISState(const MarginalSISState<B,L,S1,IO1,Q1>& o);

  /**
   * Assignment operator.
   */
  MarginalSISState& operator=(const MarginalSISState<B,L,S1,IO1,Q1>& o);

  /**
   * Clear.
   */
  void clear();

  /**
   * Swap.
   */
  void swap(MarginalSISState<B,L,S1,IO1,Q1>& o);

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
    S1(m, P, Y, T), q(B::NP), logLikelihoods(Y) {
  //
}

template<class B, bi::Location L, class S1, class IO1, class Q1>
bi::MarginalSISState<B,L,S1,IO1,Q1>::MarginalSISState(
    const MarginalSISState<B,L,S1,IO1,Q1>& o) :
    S1(o), q(o.q), logLikelihoods(o.logLikelihoods) {
  //
}

template<class B, bi::Location L, class S1, class IO1, class Q1>
bi::MarginalSISState<B,L,S1,IO1,Q1>& bi::MarginalSISState<B,L,S1,IO1,Q1>::operator=(
    const MarginalSISState<B,L,S1,IO1,Q1>& o) {
  S1::operator=(o);
  q = o.q;
  logLikelihoods = o.logLikelihoods;

  return *this;
}

template<class B, bi::Location L, class S1, class IO1, class Q1>
void bi::MarginalSISState<B,L,S1,IO1,Q1>::clear() {
  S1::clear();
  q.clear();
  logLikelihoods.clear();
}

template<class B, bi::Location L, class S1, class IO1, class Q1>
void bi::MarginalSISState<B,L,S1,IO1,Q1>::swap(MarginalSISState<B,L,S1,IO1,Q1>& o) {
  S1::swap(o);
  q.swap(o.q);
  logLikelihoods.swap(o.logLikelihoods);
}

template<class B, bi::Location L, class S1, class IO1, class Q1>
template<class Archive>
void bi::MarginalSISState<B,L,S1,IO1,Q1>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < S1 > (*this);
  ar & q;
  load_resizable_vector(ar, version, logLikelihoods);
}

template<class B, bi::Location L, class S1, class IO1, class Q1>
template<class Archive>
void bi::MarginalSISState<B,L,S1,IO1,Q1>::load(Archive& ar,
    const unsigned version) {
  ar & boost::serialization::base_object < S1 > (*this);
  ar & q;
  save_resizable_vector(ar, version, logLikelihoods);
}

#endif
