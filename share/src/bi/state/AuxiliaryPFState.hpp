/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_AUXILIARYPFSTATE_HPP
#define BI_STATE_AUXILIARYPFSTATE_HPP

#include "BootstrapPFState.hpp"

namespace bi {
/**
 * State for LookaheadPF.
 *
 * @ingroup state
 */
template<class B, Location L>
class AuxiliaryPFState: public BootstrapPFState<B,L> {
public:
  /**
   * Constructor.
   *
   * @param P Number of \f$x\f$-particles.
   * @param Y Number of observation times.
   * @param T Number of output times.
   */
  AuxiliaryPFState(const int P = 0, const int Y = 0, const int T = 0);

  /**
   * Shallow copy constructor.
   */
  AuxiliaryPFState(const AuxiliaryPFState<B,L>& o);

  /**
   * Assignment operator.
   */
  AuxiliaryPFState& operator=(const AuxiliaryPFState<B,L>& o);

  /**
   * Clear.
   */
  void clear();

  /**
   * Swap.
   */
  void swap(AuxiliaryPFState<B,L>& o);

  /**
   * Auxiliary log-weights vector.
   */
  typename State<B,L>::vector_reference_type logAuxWeights();

  /**
   * Auxiliary log-weights vector.
   */
  const typename State<B,L>::vector_reference_type logAuxWeights() const;

  /**
   * @copydoc BootstrapPFState::trim()
   */
  void trim();

  /**
   * @copydoc BootstrapPFState::resizeMax()
   */
  void resizeMax(const int maxP, const bool preserve = true);

  /**
   * @copydoc BootstrapPFState::gather()
   */
  template<class V1>
  void gather(const ScheduleElement now, const V1 as);

private:
  /**
   * Proposal log-weights.
   */
  typename State<B,L>::vector_type qlws;

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
bi::AuxiliaryPFState<B,L>::AuxiliaryPFState(const int P, const int Y, const int T) :
    BootstrapPFState<B,L>(P, Y, T), qlws(P) {
  //
}

template<class B, bi::Location L>
bi::AuxiliaryPFState<B,L>::AuxiliaryPFState(const AuxiliaryPFState<B,L>& o) :
    BootstrapPFState<B,L>(o), qlws(o.qlws) {
  //
}

template<class B, bi::Location L>
bi::AuxiliaryPFState<B,L>& bi::AuxiliaryPFState<B,L>::operator=(
    const AuxiliaryPFState<B,L>& o) {
  BootstrapPFState<B,L>::operator=(o);
  logAuxWeights() = o.logAuxWeights();

  return *this;
}

template<class B, bi::Location L>
void bi::AuxiliaryPFState<B,L>::clear() {
  BootstrapPFState<B,L>::clear();
  logAuxWeights().clear();
}

template<class B, bi::Location L>
void bi::AuxiliaryPFState<B,L>::swap(AuxiliaryPFState<B,L>& o) {
  BootstrapPFState<B,L>::swap(o);
  qlws.swap(o.qlws);
}

template<class B, bi::Location L>
typename bi::State<B,L>::vector_reference_type bi::AuxiliaryPFState<B,L>::logAuxWeights() {
  return subrange(qlws, this->p, this->P);
}

template<class B, bi::Location L>
const typename bi::State<B,L>::vector_reference_type bi::AuxiliaryPFState<B,L>::logAuxWeights() const {
  return subrange(qlws, this->p, this->P);
}

template<class B, bi::Location L>
inline void bi::AuxiliaryPFState<B,L>::trim() {
  qlws.trim(this->p, this->P);
  BootstrapPFState<B,L>::trim();
}

template<class B, bi::Location L>
inline void bi::AuxiliaryPFState<B,L>::resizeMax(const int maxP,
    const bool preserve) {
  qlws.resize(maxP, preserve);
  BootstrapPFState<B,L>::resizeMax(maxP, preserve);
}

template<class B, bi::Location L>
template<class V1>
void bi::AuxiliaryPFState<B,L>::gather(const ScheduleElement now,
    const V1 as) {
  BootstrapPFState<B,L>::gather(now, as);
  if (!now.hasOutput()) {
    bi::gather(as, logAuxWeights(), logAuxWeights());
  }
}

template<class B, bi::Location L>
template<class Archive>
void bi::AuxiliaryPFState<B,L>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < BootstrapPFState<B,L> > (*this);
  save_resizable_vector(ar, version, qlws);
}

template<class B, bi::Location L>
template<class Archive>
void bi::AuxiliaryPFState<B,L>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < BootstrapPFState<B,L> > (*this);
  load_resizable_vector(ar, version, qlws);
}

#endif
