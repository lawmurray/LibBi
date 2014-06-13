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
   */
  AuxiliaryPFState(const int P = 0);

  /**
   * Shallow copy constructor.
   */
  AuxiliaryPFState(const AuxiliaryPFState<B,L>& o);

  /**
   * Assignment operator.
   */
  AuxiliaryPFState& operator=(const AuxiliaryPFState<B,L>& o);

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
bi::AuxiliaryPFState<B,L>::AuxiliaryPFState(const int P) :
    BootstrapPFState<B,L>(P), qlws(P) {
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
  qlws = o.qlws;

  return *this;
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
