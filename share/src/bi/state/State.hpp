/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_STATE_HPP
#define BI_STATE_STATE_HPP

#include "../model/Var.hpp"
#include "../traits/var_traits.hpp"
#include "../math/loc_vector.hpp"
#include "../math/loc_matrix.hpp"
#ifdef ENABLE_SSE
#include "../sse/math/scalar.hpp"
#endif

#include "boost/serialization/split_member.hpp"

namespace bi {
/**
 * %State of Model %model.
 *
 * @ingroup state
 *
 * @tparam B Model type.
 * @tparam L Location.
 *
 * @section State_Serialization Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 */
template<class B, Location L>
class State {
public:
  typedef real value_type;
  typedef typename loc_matrix<L,value_type,-1,-1,-1,1>::type matrix_type;
  typedef typename matrix_type::matrix_reference_type matrix_reference_type;

  static const bool on_device = (L == ON_DEVICE);

  /**
   * Constructor.
   *
   * @param P Number of trajectories to store.
   */
  CUDA_FUNC_BOTH
  State(const int P = 0);

  /**
   * Shallow copy constructor.
   */
  CUDA_FUNC_BOTH
  State(const State<B,L>& o);

  /**
   * Assignment operator.
   */
  State<B,L>& operator=(const State<B,L>& o);

  /**
   * Generic assignment operator.
   */
  template<Location L2>
  State<B,L>& operator=(const State<B,L2>& o);

  /**
   * Set the active range of trajectories in the state.
   *
   * @param p The starting index.
   * @param P The number of trajectories.
   *
   * It is required that <tt>p == roundup(p)</tt> and
   * <tt>P == roundup(P)</tt> to ensure correct memory alignment. See
   * #roundup.
   */
  CUDA_FUNC_BOTH
  void setRange(const int p, const int P);

  /**
   * Starting index of active range.
   */
  CUDA_FUNC_BOTH
  int start() const;

  /**
   * Number of trajectories in active range.
   */
  CUDA_FUNC_BOTH
  int size() const;

  /**
   * Resize buffers.
   *
   * @param P Number of trajectories to store.
   * @param preserve True to preserve existing values, false otherwise.
   *
   * Resizes the state to store at least @p P number of trajectories.
   * Storage for additional trajectories may be added in some contexts,
   * see #roundup. The active range will be set to the full buffer, with
   * previously active trajectories optionally preserved.
   */
  void resize(const int P, const bool preserve = true);

  /**
   * Maximum number of trajectories that can be presently stored in the
   * state.
   */
  CUDA_FUNC_BOTH
  int sizeMax() const;

  /**
   * Resize buffers.
   *
   * @param maxP Maximum number of trajectories to store.
   * @param preserve True to preserve existing values, false otherwise.
   *
   * Resizes the state to store at least @p maxP number of trajectories.
   * Storage for additional trajectories may be added in some contexts,
   * see #roundup. This affects the maximum size (see #maxSize), but not
   * the number of trajectories currently active (see #size and #setRange).
   * The active range of trajectories will be shrunk if necessary.
   */
  void resizeMax(const int maxP, const bool preserve = true);

  /**
   * Clear.
   */
  void clear();

  /**
   * Get buffer for net.
   *
   * @param type Node type.
   *
   * @return Given buffer.
   */
  CUDA_FUNC_BOTH
  matrix_reference_type get(const VarType type);

  /**
   * Get buffer for net.
   *
   * @param type Node type.
   *
   * @return Given buffer.
   */
  CUDA_FUNC_BOTH
  const matrix_reference_type get(const VarType type) const;

  /**
   * Get buffer for variable.
   *
   * @tparam X Variable type.
   *
   * @return Given buffer.
   */
  template<class X>
  CUDA_FUNC_BOTH matrix_reference_type getVar();

  /**
   * Get buffer for variable.
   *
   * @tparam X Variable type.
   *
   * @return Given buffer.
   */
  template<class X>
  CUDA_FUNC_BOTH const matrix_reference_type getVar() const;

  /**
   * Get alternative buffer for variable.
   *
   * @tparam X Variable type.
   *
   * @return Given buffer.
   */
  template<class X>
  CUDA_FUNC_BOTH matrix_reference_type getVarAlt();

  /**
   * Get alternative buffer for variable.
   *
   * @tparam X Variable type.
   *
   * @return Given buffer.
   */
  template<class X>
  CUDA_FUNC_BOTH const matrix_reference_type getVarAlt() const;

  /**
   * Get buffer for variable.
   *
   * @param var Variable.
   *
   * @return Given buffer.
   */
  CUDA_FUNC_BOTH
  matrix_reference_type getVar(const Var* var);

  /**
   * Get buffer for variable.
   *
   * @param var Variable.
   *
   * @return Given buffer.
   */
  CUDA_FUNC_BOTH
  const matrix_reference_type getVar(const Var* var) const;

  /**
   * Get variable.
   *
   * @tparam X Variable type.
   *
   * @param p Trajectory index.
   * @param ix Serial coordinate.
   *
   * @return Variable.
   */
  template<class X>
  CUDA_FUNC_BOTH real& getVar(const int p, const int ix);

  /**
   * Get variable.
   *
   * @tparam X Variable type.
   *
   * @param p Trajectory index.
   * @param ix Serial coordinate.
   *
   * @return Variable.
   */
  template<class X>
  CUDA_FUNC_BOTH const real& getVar(const int p, const int ix) const;

  /**
   * Get variable.
   *
   * @tparam X Variable type.
   *
   * @param p Trajectory index.
   * @param ix Serial coordinate.
   *
   * @return Variable.
   */
  template<class X>
  CUDA_FUNC_BOTH real& getVarAlt(const int p, const int ix);

  /**
   * Get variable.
   *
   * @tparam X Variable type.
   *
   * @param p Trajectory index.
   * @param ix Serial coordinate.
   *
   * @return Variable.
   */
  template<class X>
  CUDA_FUNC_BOTH const real& getVarAlt(const int p, const int ix) const;

  /**
   * Get buffer of param and param_aux_ variables.
   */
  CUDA_FUNC_BOTH
  matrix_reference_type getCommon();

  /**
   * Get buffer of param and param_aux_ variables.
   */
  CUDA_FUNC_BOTH
  const matrix_reference_type getCommon() const;

  /**
   * Get buffer of state and noise variables.
   */
  CUDA_FUNC_BOTH
  matrix_reference_type getDyn();

  /**
   * Get buffer of state and noise variables.
   */
  CUDA_FUNC_BOTH
  const matrix_reference_type getDyn() const;

  /**
   * Round up number of trajectories as required by implementation.
   *
   * @param P Minimum number of trajectories.
   *
   * @return Number of trajectories.
   *
   * The following rules are applied:
   *
   * @li for @p L on device, @p P must be either less than 32, or a
   * multiple of 32, and
   * @li for @p L on host with SSE enabled, @p P must be zero, one or a
   * multiple of four (single precision) or two (double precision).
   */
  static CUDA_FUNC_BOTH int roundup(const int P);

private:
  /**
   * Storage for dense non-shared variables.
   */
  matrix_type Xdn;

  /**
   * Storage for dense shared variables.
   */
  matrix_type Kdn;

  /**
   * Index of starting trajectory in @p Xdn.
   */
  int p;

  /**
   * Number of trajectories.
   */
  int P;

  /* net sizes, for convenience */
  static const int NR = B::NR;
  static const int ND = B::ND;
  static const int NP = B::NP;
  static const int NF = B::NF;
  static const int NO = B::NO;
  static const int NDX = B::NDX;
  static const int NPX = B::NPX;

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

#include "../math/view.hpp"

#include "boost/typeof/typeof.hpp"

template<class B, bi::Location L>
bi::State<B,L>::State(const int P) :
    Xdn(roundup(P), NR + ND + NO + NDX + NR + ND),  // includes dy- and ry-vars
    Kdn(1, NP + NPX + NF + NP + NO),  // includes py- and oy-vars
    p(0), P(roundup(P)) {
  clear();
}

template<class B, bi::Location L>
bi::State<B,L>::State(const State<B,L>& o) :
    Xdn(o.Xdn), Kdn(o.Kdn), p(o.p), P(o.P) {
  //
}

template<class B, bi::Location L>
bi::State<B,L>& bi::State<B,L>::operator=(const State<B,L>& o) {
  rows(Xdn, p, P) = rows(o.Xdn, o.p, o.P);
  Kdn = o.Kdn;

  return *this;
}

template<class B, bi::Location L>
template<bi::Location L2>
bi::State<B,L>& bi::State<B,L>::operator=(const State<B,L2>& o) {
  rows(Xdn, p, P) = rows(o.Xdn, o.p, o.P);
  Kdn = o.Kdn;

  return *this;
}

template<class B, bi::Location L>
inline void bi::State<B,L>::setRange(const int p, const int P) {
  /* pre-condition */
  BI_ASSERT(p >= 0 && p == roundup(p));
  BI_ASSERT(P >= 0 && P == roundup(P));

  if (p + P > sizeMax()) {
    resizeMax(p + P, true);
  }

  this->p = p;
  this->P = P;
}

template<class B, bi::Location L>
inline int bi::State<B,L>::start() const {
  return p;
}

template<class B, bi::Location L>
inline int bi::State<B,L>::size() const {
  return P;
}

template<class B, bi::Location L>
inline void bi::State<B,L>::resize(const int P, const bool preserve) {
  const int P1 = roundup(P);

  if (preserve && p != 0) {
    /* move active range to start of buffer, being careful of overlap */
    int n = 0, N = bi::min(this->P, P1);
    while (p < N - n) {
      /* be careful of overlap */
      rows(Xdn, n, p) = rows(Xdn, p + n, p);
      n += p;
    }
    rows(Xdn, n, N - n) = rows(Xdn, p + n, N - n);
  }

  Xdn.resize(P1, Xdn.size2(), preserve);
  p = 0;
  this->P = P1;
}

template<class B, bi::Location L>
inline int bi::State<B,L>::sizeMax() const {
  return Xdn.size1();
}

template<class B, bi::Location L>
inline void bi::State<B,L>::resizeMax(const int maxP, const bool preserve) {
  const int maxP1 = roundup(maxP);

  Xdn.resize(maxP1, Xdn.size2(), preserve);
  if (p > sizeMax()) {
    p = sizeMax();
  }
  if (p + P > sizeMax()) {
    P = sizeMax() - p;
  }
}

template<class B, bi::Location L>
inline void bi::State<B,L>::clear() {
  rows(Xdn, 0, P).clear();
  Kdn.clear();
}

template<class B, bi::Location L>
inline typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::get(
    const VarType type) {
  switch (type) {
  case R_VAR:
    return subrange(Xdn.ref(), p, P, 0, NR);
  case D_VAR:
    return subrange(Xdn.ref(), p, P, NR, ND);
  case O_VAR:
    return subrange(Xdn.ref(), p, P, NR + ND, NO);
  case DX_VAR:
    return subrange(Xdn.ref(), p, P, NR + ND + NO, NDX);
  case RY_VAR:
    return subrange(Xdn.ref(), p, P, NR + ND + NO + NDX, NR);
  case DY_VAR:
    return subrange(Xdn.ref(), p, P, NR + ND + NO + NDX + NR, ND);
  case P_VAR:
    return columns(Kdn.ref(), 0, NP);
  case PX_VAR:
    return columns(Kdn.ref(), NP, NPX);
  case F_VAR:
    return columns(Kdn.ref(), NP + NPX, NF);
  case PY_VAR:
    return columns(Kdn.ref(), NP + NPX + NF, NP);
  case OY_VAR:
    return columns(Kdn.ref(), NP + NPX + NF + NP, NO);
  default:
    BI_ASSERT(false);
    return subrange(Xdn.ref(), 0, 0, 0, 0);
  }
}

template<class B, bi::Location L>
inline const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::get(
    const VarType type) const {
  switch (type) {
  case R_VAR:
    return subrange(Xdn.ref(), p, P, 0, NR);
  case D_VAR:
    return subrange(Xdn.ref(), p, P, NR, ND);
  case O_VAR:
    return subrange(Xdn.ref(), p, P, NR + ND, NO);
  case DX_VAR:
    return subrange(Xdn.ref(), p, P, NR + ND + NO, NDX);
  case RY_VAR:
    return subrange(Xdn.ref(), p, P, NR + ND + NO + NDX, NR);
  case DY_VAR:
    return subrange(Xdn.ref(), p, P, NR + ND + NO + NDX + NR, ND);
  case P_VAR:
    return columns(Kdn.ref(), 0, NP);
  case PX_VAR:
    return columns(Kdn.ref(), NP, NPX);
  case F_VAR:
    return columns(Kdn.ref(), NP + NPX, NF);
  case PY_VAR:
    return columns(Kdn.ref(), NP + NPX + NF, NP);
  case OY_VAR:
    return columns(Kdn.ref(), NP + NPX + NF + NP, NO);
  default:
    BI_ASSERT(false);
    return subrange(Xdn.ref(), 0, 0, 0, 0);
  }
}

template<class B, bi::Location L>
template<class X>
inline typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getVar() {
  const VarType type = var_type<X>::value;
  const int start = var_start<X>::value;
  const int size = var_size<X>::value;

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
template<class X>
inline const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getVar() const {
  const VarType type = var_type<X>::value;
  const int start = var_start<X>::value;
  const int size = var_size<X>::value;

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
template<class X>
inline typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getVarAlt() {
  const VarType type = alt_type<var_type<X>::value>::value;
  const int start = var_start<X>::value;
  const int size = var_size<X>::value;

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
template<class X>
inline const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getVarAlt() const {
  const VarType type = alt_type<var_type<X>::value>::value;
  const int start = var_start<X>::value;
  const int size = var_size<X>::value;

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
inline typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getVar(
    const Var* var) {
  return columns(get(var->getType()), var->getStart(), var->getSize());
}

template<class B, bi::Location L>
inline const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getVar(
    const Var* var) const {
  return columns(get(var->getType()), var->getStart(), var->getSize());
}

template<class B, bi::Location L>
template<class X>
inline real& bi::State<B,L>::getVar(const int p, const int ix) {
  const VarType type = var_type<X>::value;
  const int start = var_start<X>::value;

  switch (type) {
  case R_VAR:
    return Xdn(this->p + p, start + ix);
  case D_VAR:
    return Xdn(this->p + p, NR + start + ix);
  case O_VAR:
    return Xdn(this->p + p, NR + ND + start + ix);
  case DX_VAR:
    return Xdn(this->p + p, NR + ND + NO + start + ix);
  case RY_VAR:
    return Xdn(this->p + p, NR + ND + NO + NDX + start + ix);
  case DY_VAR:
    return Xdn(this->p + p, NR + ND + NO + NDX + NR + start + ix);
  case P_VAR:
    return Kdn(0, start + ix);
  case PX_VAR:
    return Kdn(0, NP + start + ix);
  case F_VAR:
    return Kdn(0, NP + NPX + start + ix);
  case PY_VAR:
    return Kdn(0, NP + NPX + NF + start + ix);
  case OY_VAR:
    return Kdn(0, NP + NPX + NF + NP + start + ix);
  default:
    BI_ASSERT(false);
    return Xdn(this->p + p, 0);
  }
}

template<class B, bi::Location L>
template<class X>
inline const real& bi::State<B,L>::getVar(const int p, const int ix) const {
  const VarType type = var_type<X>::value;
  const int start = var_start<X>::value;

  switch (type) {
  case R_VAR:
    return Xdn(this->p + p, start + ix);
  case D_VAR:
    return Xdn(this->p + p, NR + start + ix);
  case O_VAR:
    return Xdn(this->p + p, NR + ND + start + ix);
  case DX_VAR:
    return Xdn(this->p + p, NR + ND + NO + start + ix);
  case RY_VAR:
    return Xdn(this->p + p, NR + ND + NO + NDX + start + ix);
  case DY_VAR:
    return Xdn(this->p + p, NR + ND + NO + NDX + NR + start + ix);
  case P_VAR:
    return Kdn(0, start + ix);
  case PX_VAR:
    return Kdn(0, NP + start + ix);
  case F_VAR:
    return Kdn(0, NP + NPX + start + ix);
  case PY_VAR:
    return Kdn(0, NP + NPX + NF + start + ix);
  case OY_VAR:
    return Kdn(0, NP + NPX + NF + NP + start + ix);
  default:
    BI_ASSERT(false);
    return Xdn(this->p + p, 0);
  }
}

template<class B, bi::Location L>
template<class X>
inline real& bi::State<B,L>::getVarAlt(const int p, const int ix) {
  const VarType type = alt_type<var_type<X>::value>::value;
  const int start = var_start<X>::value;

  switch (type) {
  case R_VAR:
    return Xdn(this->p + p, start + ix);
  case D_VAR:
    return Xdn(this->p + p, NR + start + ix);
  case O_VAR:
    return Xdn(this->p + p, NR + ND + start + ix);
  case DX_VAR:
    return Xdn(this->p + p, NR + ND + NO + start + ix);
  case RY_VAR:
    return Xdn(this->p + p, NR + ND + NO + NDX + start + ix);
  case DY_VAR:
    return Xdn(this->p + p, NR + ND + NO + NDX + NR + start + ix);
  case P_VAR:
    return Kdn(0, start + ix);
  case PX_VAR:
    return Kdn(0, NP + start + ix);
  case F_VAR:
    return Kdn(0, NP + NPX + start + ix);
  case PY_VAR:
    return Kdn(0, NP + NPX + NF + start + ix);
  case OY_VAR:
    return Kdn(0, NP + NPX + NF + NP + start + ix);
  default:
    BI_ASSERT(false);
    return Xdn(this->p + p, 0);
  }
}

template<class B, bi::Location L>
template<class X>
inline const real& bi::State<B,L>::getVarAlt(const int p,
    const int ix) const {
  const VarType type = alt_type<var_type<X>::value>::value;
  const int start = var_start<X>::value;

  switch (type) {
  case R_VAR:
    return Xdn(this->p + p, start + ix);
  case D_VAR:
    return Xdn(this->p + p, NR + start + ix);
  case O_VAR:
    return Xdn(this->p + p, NR + ND + start + ix);
  case DX_VAR:
    return Xdn(this->p + p, NR + ND + NO + start + ix);
  case RY_VAR:
    return Xdn(this->p + p, NR + ND + NO + NDX + start + ix);
  case DY_VAR:
    return Xdn(this->p + p, NR + ND + NO + NDX + NR + start + ix);
  case P_VAR:
    return Kdn(0, start + ix);
  case PX_VAR:
    return Kdn(0, NP + start + ix);
  case F_VAR:
    return Kdn(0, NP + NPX + start + ix);
  case PY_VAR:
    return Kdn(0, NP + NPX + NF + start + ix);
  case OY_VAR:
    return Kdn(0, NP + NPX + NF + NP + start + ix);
  default:
    BI_ASSERT(false);
    return Xdn(this->p + p, 0);
  }
}

template<class B, bi::Location L>
inline typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getCommon() {
  return columns(Kdn.ref(), 0, Kdn.size2());
}

template<class B, bi::Location L>
inline const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getCommon() const {
  return columns(Kdn.ref(), 0, Kdn.size2());
}

template<class B, bi::Location L>
inline typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getDyn() {
  return subrange(Xdn.ref(), p, P, 0, NR + ND);
}

template<class B, bi::Location L>
inline const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getDyn() const {
  return subrange(Xdn.ref(), p, P, 0, NR + ND);
}

template<class B, bi::Location L>
int bi::State<B,L>::roundup(const int P) {
  int P1 = P;
  if (L == ON_DEVICE) {
    /* either < 32 or a multiple of 32 number of trajectories required */
    if (P1 > 32) {
      P1 = ((P1 + 31) / 32) * 32;
    }
  } else {
#ifdef ENABLE_SSE
    /* zero, one or a multiple of 4 (single precision) or 2 (double
     * precision) required */
    if (P1 > 1) {
      P1 = ((P1 + BI_SSE_SIZE - 1)/BI_SSE_SIZE)*BI_SSE_SIZE;
    }
#endif
  }

  return P1;
}

template<class B, bi::Location L>
template<class Archive>
void bi::State<B,L>::save(Archive& ar, const unsigned version) const {
  save_resizable_matrix(ar, version, Xdn);
  save_resizable_matrix(ar, version, Kdn);
  ar & p;
  ar & P;
}

template<class B, bi::Location L>
template<class Archive>
void bi::State<B,L>::load(Archive& ar, const unsigned version) {
  load_resizable_matrix(ar, version, Xdn);
  load_resizable_matrix(ar, version, Kdn);
  ar & p;
  ar & P;
}

#endif
