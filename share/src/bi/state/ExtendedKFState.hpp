/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_EXTENDEDKFSTATE_HPP
#define BI_STATE_EXTENDEDKFSTATE_HPP

#include "State.hpp"

namespace bi {
/**
 * State for ExtendedKF.
 *
 * @ingroup state
 */
template<class B, Location L>
class ExtendedKFState: public State<B,L> {
public:
  /**
   * Constructor.
   *
   * @param P Number of \f$x\f$-particles.
   */
  ExtendedKFState();

  /**
   * Shallow copy constructor.
   */
  ExtendedKFState(const ExtendedKFState<B,L>& o);

  /**
   * Assignment operator.
   */
  ExtendedKFState& operator=(const ExtendedKFState<B,L>& o);

  /*
   * Uncorrected and correct means.
   */
  typename State<B,L>::vector_type mu1, mu2;

  /*
   * Square-roots of uncorrected and corrected covariance matrices,
   * cross-covariance matrix.
   */
  typename State<B,L>::matrix_type U1, U2, C;

  /* views */
  typename State<B,L>::matrix_reference_type F, Q, G, R;

private:
  /**
   * Number of dynamic variables.
   */
  static const int M = B::NR + B::ND;

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
bi::ExtendedKFState<B,L>::ExtendedKFState() :
    State<B,L>(), mu1(M), mu2(M), U1(M, M), U2(M, M), C(M, M), F(
        reshape(this->getVar<VarGroupF>(), M, M)), Q(
        reshape(this->getVar<VarGroupQ>(), M, M)), G(
        reshape(this->getVar<VarGroupG>(), M, B::NO)), R(
        reshape(this->getVar<VarGroupR>(), B::NO, B::NO)) {
  //
}

template<class B, bi::Location L>
bi::ExtendedKFState<B,L>::ExtendedKFState(const ExtendedKFState<B,L>& o) :
    State<B,L>(o), mu1(o.mu1), mu2(o.mu2), U1(o.U1), U2(o.U2), C(o.C), F(o.F), Q(
        o.Q), G(o.G), R(o.R) {
  //
}

template<class B, bi::Location L>
bi::ExtendedKFState<B,L>& bi::ExtendedKFState<B,L>::operator=(
    const ExtendedKFState<B,L>& o) {
  State<B,L>::operator=(o);
  mu1 = o.mu1;
  mu2 = o.mu2;
  U1 = o.U1;
  U2 = o.U2;
  C = o.C;

  return *this;
}

template<class B, bi::Location L>
template<class Archive>
void bi::ExtendedKFState<B,L>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < State<B,L> > (*this);
  save_resizable_vector(ar, version, mu1);
  save_resizable_vector(ar, version, mu2);
  save_resizable_matrix(ar, version, U1);
  save_resizable_matrix(ar, version, U2);
  save_resizable_matrix(ar, version, C);
  ar & F & Q & G & R;
}

template<class B, bi::Location L>
template<class Archive>
void bi::ExtendedKFState<B,L>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < State<B,L> > (*this);
  load_resizable_vector(ar, version, mu1);
  load_resizable_vector(ar, version, mu2);
  load_resizable_matrix(ar, version, U1);
  load_resizable_matrix(ar, version, U2);
  load_resizable_matrix(ar, version, C);
  ar & F & Q & G & R;
}

#endif
