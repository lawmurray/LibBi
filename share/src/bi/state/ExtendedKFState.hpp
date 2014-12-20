/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_EXTENDEDKFSTATE_HPP
#define BI_STATE_EXTENDEDKFSTATE_HPP

#include "FilterState.hpp"

namespace bi {
/**
 * State for ExtendedKF.
 *
 * @ingroup state
 */
template<class B, Location L>
class ExtendedKFState: public FilterState<B,L> {
public:
  /**
   * Constructor.
   *
   * @param P Number of \f$x\f$-particles.
   * @param Y Number of observation times.
   * @param T Number of output times.
   */
  ExtendedKFState(const int P = 0, const int Y = 0, const int T = 0);

  /**
   * Shallow copy constructor.
   */
  ExtendedKFState(const ExtendedKFState<B,L>& o);

  /**
   * Assignment operator.
   */
  ExtendedKFState& operator=(const ExtendedKFState<B,L>& o);

  /**
   * Clear.
   */
  void clear();

  /**
   * Swap.
   */
  void swap(ExtendedKFState<B,L>& o);

  /*
   * Views of Jacobian matrices etc.
   */
  typename State<B,L>::matrix_reference_type F();
  typename State<B,L>::matrix_reference_type Q();
  typename State<B,L>::matrix_reference_type G();
  typename State<B,L>::matrix_reference_type R();

  /*
   * Uncorrected and correct means.
   */
  typename State<B,L>::vector_type mu1, mu2;

  /*
   * Square-roots of uncorrected and corrected covariance matrices,
   * cross-covariance matrix.
   */
  typename State<B,L>::matrix_type U1, U2, C;

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
bi::ExtendedKFState<B,L>::ExtendedKFState(const int P, const int Y, const int T) :
    FilterState<B,L>(P, Y, T), mu1(M), mu2(M), U1(M, M), U2(M, M), C(M, M) {
  //
}

template<class B, bi::Location L>
bi::ExtendedKFState<B,L>::ExtendedKFState(const ExtendedKFState<B,L>& o) :
    FilterState<B,L>(o), mu1(o.mu1), mu2(o.mu2), U1(o.U1), U2(o.U2), C(o.C) {
  //
}

template<class B, bi::Location L>
bi::ExtendedKFState<B,L>& bi::ExtendedKFState<B,L>::operator=(
    const ExtendedKFState<B,L>& o) {
  FilterState<B,L>::operator=(o);
  mu1 = o.mu1;
  mu2 = o.mu2;
  U1 = o.U1;
  U2 = o.U2;
  C = o.C;

  return *this;
}

template<class B, bi::Location L>
void bi::ExtendedKFState<B,L>::clear() {
  FilterState<B,L>::clear();

  /* mean and Cholesky factor of initial state */
  mu1 = row(this->getDyn(), 0);
  U1 = Q();

  /* reset Jacobian, as it has now been multiplied in */
  ident(F());
  Q().clear();

  /* across-time covariance */
  C.clear();
}

template<class B, bi::Location L>
void bi::ExtendedKFState<B,L>::swap(ExtendedKFState<B,L>& o) {
  FilterState<B,L>::swap(o);
  mu1.swap(o.mu1);
  mu2.swap(o.mu2);
  U1.swap(o.U1);
  U2.swap(o.U2);
  C.swap(o.C);
}

template<class B, bi::Location L>
typename bi::State<B,L>::matrix_reference_type bi::ExtendedKFState<B,L>::F() {
  return reshape(this->template getVar<VarGroupF>(), M, M);
}

template<class B, bi::Location L>
typename bi::State<B,L>::matrix_reference_type bi::ExtendedKFState<B,L>::Q() {
  return reshape(this->template getVar<VarGroupQ>(), M, M);
}

template<class B, bi::Location L>
typename bi::State<B,L>::matrix_reference_type bi::ExtendedKFState<B,L>::G() {
  return reshape(this->template getVar<VarGroupG>(), M, B::NO);
}

template<class B, bi::Location L>
typename bi::State<B,L>::matrix_reference_type bi::ExtendedKFState<B,L>::R() {
  return reshape(this->template getVar<VarGroupR>(), B::NO, B::NO);
}

template<class B, bi::Location L>
template<class Archive>
void bi::ExtendedKFState<B,L>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < FilterState<B,L> > (*this);
  save_resizable_vector(ar, version, mu1);
  save_resizable_vector(ar, version, mu2);
  save_resizable_matrix(ar, version, U1);
  save_resizable_matrix(ar, version, U2);
  save_resizable_matrix(ar, version, C);
}

template<class B, bi::Location L>
template<class Archive>
void bi::ExtendedKFState<B,L>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < FilterState<B,L> > (*this);
  load_resizable_vector(ar, version, mu1);
  load_resizable_vector(ar, version, mu2);
  load_resizable_matrix(ar, version, U1);
  load_resizable_matrix(ar, version, U2);
  load_resizable_matrix(ar, version, C);
}

#endif
