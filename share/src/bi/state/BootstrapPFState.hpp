/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_BOOTSTRAPPFSTATE_HPP
#define BI_STATE_BOOTSTRAPPFSTATE_HPP

#include "FilterState.hpp"

namespace bi {
/**
 * State for BootstrapPF.
 *
 * @ingroup state
 */
template<class B, Location L>
class BootstrapPFState: public FilterState<B,L> {
public:
  /**
   * Constructor.
   *
   * @param P Number of \f$x\f$-particles.
   * @param Y Number of observation times.
   * @param T Number of output times.
   */
  BootstrapPFState(const int P = 0, const int Y = 0, const int T = 0);

  /**
   * Shallow copy constructor.
   */
  BootstrapPFState(const BootstrapPFState<B,L>& o);

  /**
   * Assignment operator.
   */
  BootstrapPFState& operator=(const BootstrapPFState<B,L>& o);

  /**
   * Clear.
   */
  void clear();

  /**
   * Swap.
   */
  void swap(BootstrapPFState<B,L>& o);

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
   * @copydoc State::trim()
   */
  void trim();

  /**
   * @copydoc State::resizeMax()
   */
  void resizeMax(const int maxP, const bool preserve = true);

  /**
   * Gather particles after resampling.
   *
   * @tparam V1 Vector type.
   *
   * @param now Current step in time schedule.
   * @param as Ancestry.
   */
  template<class V1>
  void gather(const ScheduleElement now, const V1 as);

  /**
   * Last ESS.
   */
  double ess;

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

template<class B, bi::Location L>
bi::BootstrapPFState<B,L>::BootstrapPFState(const int P, const int Y,
    const int T) :
    FilterState<B,L>(P, Y, T), ess(0.0), lws(P), as(P) {
  //
}

template<class B, bi::Location L>
bi::BootstrapPFState<B,L>::BootstrapPFState(const BootstrapPFState<B,L>& o) :
    FilterState<B,L>(o), ess(0.0), lws(o.lws), as(o.as) {
  //
}

template<class B, bi::Location L>
bi::BootstrapPFState<B,L>& bi::BootstrapPFState<B,L>::operator=(
    const BootstrapPFState<B,L>& o) {
  FilterState<B,L>::operator=(o);
  ess = o.ess;
  logWeights() = o.logWeights();
  ancestors() = o.ancestors();

  return *this;
}

template<class B, bi::Location L>
void bi::BootstrapPFState<B,L>::clear() {
  FilterState<B,L>::clear();
  ess = 0.0;
  logWeights().clear();
  seq_elements(ancestors(), 0);
}

template<class B, bi::Location L>
void bi::BootstrapPFState<B,L>::swap(BootstrapPFState<B,L>& o) {
  FilterState<B,L>::swap(o);
  std::swap(ess, o.ess);
  lws.swap(o.lws);
  as.swap(o.as);
}

template<class B, bi::Location L>
typename bi::State<B,L>::vector_reference_type bi::BootstrapPFState<B,L>::logWeights() {
  return subrange(lws, this->p, this->P);
}

template<class B, bi::Location L>
const typename bi::State<B,L>::vector_reference_type bi::BootstrapPFState<B,L>::logWeights() const {
  return subrange(lws, this->p, this->P);
}

template<class B, bi::Location L>
typename bi::State<B,L>::int_vector_reference_type bi::BootstrapPFState<B,L>::ancestors() {
  return subrange(as, this->p, this->P);
}

template<class B, bi::Location L>
const typename bi::State<B,L>::int_vector_reference_type bi::BootstrapPFState<
    B,L>::ancestors() const {
  return subrange(as, this->p, this->P);
}

template<class B, bi::Location L>
inline void bi::BootstrapPFState<B,L>::trim() {
  FilterState<B,L>::trim();
  lws.trim(this->p, this->P);
  as.trim(this->p, this->P);
}

template<class B, bi::Location L>
inline void bi::BootstrapPFState<B,L>::resizeMax(const int maxP,
    const bool preserve) {
  FilterState<B,L>::resizeMax(maxP, preserve);
  lws.resize(maxP, preserve);
  as.resize(maxP, preserve);
}

template<class B, bi::Location L>
template<class V1>
void bi::BootstrapPFState<B,L>::gather(const ScheduleElement now,
    const V1 as) {
  FilterState<B,L>::gather(as);
  if (now.hasOutput()) {
    ancestors() = as;
  } else {
    bi::gather(as, ancestors(), ancestors());
  }
}

template<class B, bi::Location L>
template<class Archive>
void bi::BootstrapPFState<B,L>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < FilterState<B,L> > (*this);
  ar & ess;
  save_resizable_vector(ar, version, lws);
  save_resizable_vector(ar, version, as);
}

template<class B, bi::Location L>
template<class Archive>
void bi::BootstrapPFState<B,L>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < FilterState<B,L> > (*this);
  ar & ess;
  load_resizable_vector(ar, version, lws);
  load_resizable_vector(ar, version, as);
}

#endif
