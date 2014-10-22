/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_BOOTSTRAPPFSTATE_HPP
#define BI_STATE_BOOTSTRAPPFSTATE_HPP

#include "State.hpp"

namespace bi {
/**
 * State for BootstrapPF.
 *
 * @ingroup state
 */
template<class B, Location L>
class BootstrapPFState: public State<B,L> {
public:
  /**
   * Constructor.
   *
   * @param P Number of \f$x\f$-particles.
   */
  BootstrapPFState(const int P = 0);

  /**
   * Shallow copy constructor.
   */
  BootstrapPFState(const BootstrapPFState<B,L>& o);

  /**
   * Assignment operator.
   */
  BootstrapPFState& operator=(const BootstrapPFState<B,L>& o);

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
bi::BootstrapPFState<B,L>::BootstrapPFState(const int P) :
    State<B,L>(P), lws(P), as(P) {
  //
}

template<class B, bi::Location L>
bi::BootstrapPFState<B,L>::BootstrapPFState(const BootstrapPFState<B,L>& o) :
    State<B,L>(o), lws(o.lws), as(o.as) {
  //
}

template<class B, bi::Location L>
bi::BootstrapPFState<B,L>& bi::BootstrapPFState<B,L>::operator=(
    const BootstrapPFState<B,L>& o) {
  State<B,L>::operator=(o);
  lws = o.lws;
  as = o.as;

  return *this;
}

template<class B, bi::Location L>
void bi::BootstrapPFState<B,L>::swap(BootstrapPFState<B,L>& o) {
  State<B,L>::swap(o);
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
  lws.trim(this->p, this->P);
  as.trim(this->p, this->P);
  State<B,L>::trim();
}

template<class B, bi::Location L>
inline void bi::BootstrapPFState<B,L>::resizeMax(const int maxP,
    const bool preserve) {
  lws.resize(maxP, preserve);
  as.resize(maxP, preserve);
  State<B,L>::resizeMax(maxP, preserve);
}

template<class B, bi::Location L>
template<class Archive>
void bi::BootstrapPFState<B,L>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < State<B,L> > (*this);
  save_resizable_vector(ar, version, lws);
  save_resizable_vector(ar, version, as);
}

template<class B, bi::Location L>
template<class Archive>
void bi::BootstrapPFState<B,L>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < State<B,L> > (*this);
  load_resizable_vector(ar, version, lws);
  load_resizable_vector(ar, version, as);
}

#endif
