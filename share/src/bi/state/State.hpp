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
#include "../math/loc_temp_vector.hpp"
#include "../math/loc_temp_matrix.hpp"

#include "boost/serialization/split_member.hpp"

namespace bi {
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
int roundup(const int P);
}

#ifdef ENABLE_SSE
#include "../sse/math/scalar.hpp"
#endif

inline int bi::roundup(const int P) {
  int P1 = P;

#if defined(ENABLE_CUDA)
  /* either < 32 or a multiple of 32 number of trajectories required */
  if (P1 > 32) {
    P1 = ((P1 + 31) / 32) * 32;
  }
#elif defined(ENABLE_SSE)
  /* zero, one or a multiple of 4 (single precision) or 2 (double
   * precision) required */
  if (P1 > 1) {
    P1 = ((P1 + BI_SIMD_SIZE - 1)/BI_SIMD_SIZE)*BI_SIMD_SIZE;
  }
#endif

  return P1;
}

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
  static const Location location = L;
  static const bool on_device = (L == ON_DEVICE);

  typedef real value_type;
  typedef typename loc_vector<L,value_type>::type vector_type;
  typedef typename loc_matrix<L,value_type>::type matrix_type;
  typedef typename vector_type::vector_reference_type vector_reference_type;
  typedef typename matrix_type::matrix_reference_type matrix_reference_type;

  typedef typename loc_temp_vector<L,value_type>::type temp_vector_type;
  typedef typename loc_temp_matrix<L,value_type>::type temp_matrix_type;

  typedef int int_value_type;
  typedef typename loc_vector<L,int_value_type>::type int_vector_type;
  typedef typename loc_matrix<L,int_value_type>::type int_matrix_type;
  typedef typename int_vector_type::vector_reference_type int_vector_reference_type;
  typedef typename int_matrix_type::matrix_reference_type int_matrix_reference_type;

  typedef typename loc_temp_vector<L,int_value_type>::type temp_int_vector_type;
  typedef typename loc_temp_matrix<L,int_value_type>::type temp_int_matrix_type;

  /**
   * Constructor.
   *
   * @param P Number of \f$x\f$-particles.
   * @param Y Number of observation times.
   * @param T Number of output times.
   */
  CUDA_FUNC_BOTH
  State(const int P = 1, const int Y = 0, const int T = 0);

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
   * Swap.
   */
  void swap(State<B,L>& o);

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
   * Trim buffers down to the active range.
   */
  void trim();

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
   * This affects the maximum size (see #sizeMax), and if this size is
   * reduced, may truncate the active range.
   */
  void resizeMax(const int maxP, const bool preserve = true);

  /**
   * Clear.
   */
  void clear();

  /**
   * Select single particle.
   */
  vector_reference_type select(const int p);

  /**
   * Gather particles.
   *
   * @tparam V1 Vector type.
   *
   * @param as Ancestry.
   */
  template<class V1>
  void gather(const V1 as);

  /**
   * @name Built-in variables
   */
  //@{
  /**
   * Get current time.
   */
  CUDA_FUNC_BOTH
  real getTime() const;

  /**
   * Set current time.
   */
  void setTime(const real t);

  /**
   * Get time of last input.
   */
  CUDA_FUNC_BOTH
  real getLastInputTime() const;

  /**
   * Set time of last input.
   */
  void setLastInputTime(const real t);

  /**
   * Get time of next observation.
   */
  CUDA_FUNC_BOTH
  real getNextObsTime() const;

  /**
   * Set time of next observation.
   */
  void setNextObsTime(const real t);
  //@}

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
   * Log-prior density of parameters.
   */
  double logPrior;

  /**
   * Log-proposal density of parameters.
   */
  double logProposal;

  /**
   * Execution time.
   */
  long clock;

protected:
  /* net sizes, for convenience */
  static const int NR = B::NR;
  static const int ND = B::ND;
  static const int NP = B::NP;
  static const int NF = B::NF;
  static const int NO = B::NO;
  static const int NDX = B::NDX;
  static const int NPX = B::NPX;
  static const int NB = B::NB;

  /**
   * Storage for dense non-common variables.
   */
  matrix_type Xdn;

  /**
   * Storage for dense common variables.
   */
  matrix_type Kdn;

  /**
   * Storage for built-in variables.
   */
  real builtin[NB];

  /**
   * Index of starting trajectory in @p Xdn.
   */
  int p;

  /**
   * Number of trajectories.
   */
  int P;

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

#include "../math/view.hpp"
#include "../math/constant.hpp"
#include "../primitive/matrix_primitive.hpp"

template<class B, bi::Location L>
bi::State<B,L>::State(const int P, const int Y, const int T) :
    logPrior(-BI_INF), logProposal(-BI_INF), clock(0),
    Xdn(P, NR + ND + NDX + NR + ND),  // includes dy- and ry-vars
    Kdn(1, NP + NPX + NF + NP + 2 * NO),// includes py- and oy-vars
    p(0), P(P) {
      /* pre-condition */
      BI_ASSERT(P == roundup(P));

      clear();
    }

template<class B, bi::Location L>
bi::State<B,L>::State(const State<B,L>& o) :
    logPrior(o.logPrior), logProposal(o.logProposal), clock(o.clock), Xdn(
        o.Xdn), Kdn(o.Kdn), p(o.p), P(o.P) {
  for (int i = 0; i < NB; ++i) {
    builtin[i] = o.builtin[i];
  }
}

template<class B, bi::Location L>
bi::State<B,L>& bi::State<B,L>::operator=(const State<B,L>& o) {
  logPrior = o.logPrior;
  logProposal = o.logProposal;
  clock = o.clock;
  rows(Xdn, p, P) = rows(o.Xdn, o.p, o.P);
  Kdn = o.Kdn;
  for (int i = 0; i < NB; ++i) {
    builtin[i] = o.builtin[i];
  }
  return *this;
}

template<class B, bi::Location L>
template<bi::Location L2>
bi::State<B,L>& bi::State<B,L>::operator=(const State<B,L2>& o) {
  logPrior = o.logPrior;
  logProposal = o.logProposal;
  clock = o.clock;
  rows(Xdn, p, P) = rows(o.Xdn, o.p, o.P);
  Kdn = o.Kdn;
  for (int i = 0; i < NB; ++i) {
    builtin[i] = o.builtin[i];
  }
  return *this;
}

template<class B, bi::Location L>
void bi::State<B,L>::swap(State<B,L>& o) {
  std::swap(logPrior, o.logPrior);
  std::swap(logProposal, o.logProposal);
  std::swap(clock, o.clock);
  Xdn.swap(o.Xdn);
  Kdn.swap(o.Kdn);
  for (int i = 0; i < NB; ++i) {
    std::swap(builtin[i], o.builtin[i]);
  }
}

template<class B, bi::Location L>
inline void bi::State<B,L>::setRange(const int p, const int P) {
  /* pre-condition */
  BI_ASSERT(p >= 0 && p == roundup(p));
  BI_ASSERT(P >= 0 && P == roundup(P));
  BI_ASSERT(p + P <= sizeMax());

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
inline void bi::State<B,L>::trim() {
  Xdn.trim(p, P, 0, Xdn.size2());
  p = 0;
}

template<class B, bi::Location L>
inline int bi::State<B,L>::sizeMax() const {
  return Xdn.size1();
}

template<class B, bi::Location L>
inline void bi::State<B,L>::resizeMax(const int maxP, const bool preserve) {
  /* pre-condition */
  BI_ASSERT(maxP == roundup(maxP));

  Xdn.resize(maxP, Xdn.size2(), preserve);
  if (p > maxP) {
    p = maxP;
  }
  if (p + P > maxP) {
    P = maxP - p;
  }
}

template<class B, bi::Location L>
inline void bi::State<B,L>::clear() {
  logPrior = -BI_INF;
  logProposal = -BI_INF;
  clock = 0;
  rows(Xdn, p, P).clear();
  Kdn.clear();
}

template<class B, bi::Location L>
real bi::State<B,L>::getTime() const {
  return builtin[0];
}

template<class B, bi::Location L>
void bi::State<B,L>::setTime(const real t) {
  this->builtin[0] = t;
}

template<class B, bi::Location L>
real bi::State<B,L>::getLastInputTime() const {
  return builtin[1];
}

template<class B, bi::Location L>
void bi::State<B,L>::setLastInputTime(const real t) {
  this->builtin[1] = t;
}

template<class B, bi::Location L>
real bi::State<B,L>::getNextObsTime() const {
  return builtin[2];
}

template<class B, bi::Location L>
void bi::State<B,L>::setNextObsTime(const real t) {
  this->builtin[2] = t;
}

template<class B, bi::Location L>
inline typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::get(
    const VarType type) {
  switch (type) {
  case R_VAR:
    return subrange(Xdn.ref(), p, P, 0, NR);
  case D_VAR:
    return subrange(Xdn.ref(), p, P, NR, ND);
  case DX_VAR:
    return subrange(Xdn.ref(), p, P, NR + ND, NDX);
  case RY_VAR:
    return subrange(Xdn.ref(), p, P, NR + ND + NDX, NR);
  case DY_VAR:
    return subrange(Xdn.ref(), p, P, NR + ND + NDX + NR, ND);
  case P_VAR:
    return columns(Kdn.ref(), 0, NP);
  case PX_VAR:
    return columns(Kdn.ref(), NP, NPX);
  case F_VAR:
    return columns(Kdn.ref(), NP + NPX, NF);
  case PY_VAR:
    return columns(Kdn.ref(), NP + NPX + NF, NP);
  case O_VAR:
    return columns(Kdn.ref(), NP + NPX + NF + NP, NO);
  case OY_VAR:
    return columns(Kdn.ref(), NP + NPX + NF + NP + NO, NO);
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
  case DX_VAR:
    return subrange(Xdn.ref(), p, P, NR + ND, NDX);
  case RY_VAR:
    return subrange(Xdn.ref(), p, P, NR + ND + NDX, NR);
  case DY_VAR:
    return subrange(Xdn.ref(), p, P, NR + ND + NDX + NR, ND);
  case P_VAR:
    return columns(Kdn.ref(), 0, NP);
  case PX_VAR:
    return columns(Kdn.ref(), NP, NPX);
  case F_VAR:
    return columns(Kdn.ref(), NP + NPX, NF);
  case PY_VAR:
    return columns(Kdn.ref(), NP + NPX + NF, NP);
  case O_VAR:
    return columns(Kdn.ref(), NP + NPX + NF + NP, NO);
  case OY_VAR:
    return columns(Kdn.ref(), NP + NPX + NF + NP + NO, NO);
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
  case DX_VAR:
    return Xdn(this->p + p, NR + ND + start + ix);
  case RY_VAR:
    return Xdn(this->p + p, NR + ND + NDX + start + ix);
  case DY_VAR:
    return Xdn(this->p + p, NR + ND + NDX + NR + start + ix);
  case P_VAR:
    return Kdn(0, start + ix);
  case PX_VAR:
    return Kdn(0, NP + start + ix);
  case F_VAR:
    return Kdn(0, NP + NPX + start + ix);
  case PY_VAR:
    return Kdn(0, NP + NPX + NF + start + ix);
  case O_VAR:
    return Kdn(0, NP + NPX + NF + NP + start + ix);
  case OY_VAR:
    return Kdn(0, NP + NPX + NF + NP + NO + start + ix);
  case B_VAR:
    return builtin[start + ix];
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
  case DX_VAR:
    return Xdn(this->p + p, NR + ND + start + ix);
  case RY_VAR:
    return Xdn(this->p + p, NR + ND + NDX + start + ix);
  case DY_VAR:
    return Xdn(this->p + p, NR + ND + NDX + NR + start + ix);
  case P_VAR:
    return Kdn(0, start + ix);
  case PX_VAR:
    return Kdn(0, NP + start + ix);
  case F_VAR:
    return Kdn(0, NP + NPX + start + ix);
  case PY_VAR:
    return Kdn(0, NP + NPX + NF + start + ix);
  case O_VAR:
    return Kdn(0, NP + NPX + NF + NP + start + ix);
  case OY_VAR:
    return Kdn(0, NP + NPX + NF + NP + NO + start + ix);
  case B_VAR:
    return builtin[start + ix];
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
  case DX_VAR:
    return Xdn(this->p + p, NR + ND + start + ix);
  case RY_VAR:
    return Xdn(this->p + p, NR + ND + NDX + start + ix);
  case DY_VAR:
    return Xdn(this->p + p, NR + ND + NDX + NR + start + ix);
  case P_VAR:
    return Kdn(0, start + ix);
  case PX_VAR:
    return Kdn(0, NP + start + ix);
  case F_VAR:
    return Kdn(0, NP + NPX + start + ix);
  case PY_VAR:
    return Kdn(0, NP + NPX + NF + start + ix);
  case O_VAR:
    return Kdn(0, NP + NPX + NF + NP + start + ix);
  case OY_VAR:
    return Kdn(0, NP + NPX + NF + NP + NO + start + ix);
  case B_VAR:
    return builtin[start + ix];
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
  case DX_VAR:
    return Xdn(this->p + p, NR + ND + start + ix);
  case RY_VAR:
    return Xdn(this->p + p, NR + ND + NDX + start + ix);
  case DY_VAR:
    return Xdn(this->p + p, NR + ND + NDX + NR + start + ix);
  case P_VAR:
    return Kdn(0, start + ix);
  case PX_VAR:
    return Kdn(0, NP + start + ix);
  case F_VAR:
    return Kdn(0, NP + NPX + start + ix);
  case PY_VAR:
    return Kdn(0, NP + NPX + NF + start + ix);
  case O_VAR:
    return Kdn(0, NP + NPX + NF + NP + start + ix);
  case OY_VAR:
    return Kdn(0, NP + NPX + NF + NP + NO + start + ix);
  case B_VAR:
    return builtin[start + ix];
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
typename bi::State<B,L>::vector_reference_type bi::State<B,L>::select(
    const int p) {
  return row(getDyn(), p);
}

template<class B, bi::Location L>
template<class V1>
void bi::State<B,L>::gather(const V1 as) {
  bi::gather_rows(as, getDyn(), getDyn());
}

template<class B, bi::Location L>
template<class Archive>
void bi::State<B,L>::save(Archive& ar, const unsigned version) const {
  ar & logPrior;
  ar & logProposal;
  ar & clock;
  save_resizable_matrix(ar, version, Xdn);
  save_resizable_matrix(ar, version, Kdn);
  ar & builtin;
  ar & p;
  ar & P;
}

template<class B, bi::Location L>
template<class Archive>
void bi::State<B,L>::load(Archive& ar, const unsigned version) {
  ar & logPrior;
  ar & logProposal;
  ar & clock;
  load_resizable_matrix(ar, version, Xdn);
  load_resizable_matrix(ar, version, Kdn);
  ar & builtin;
  ar & p;
  ar & P;
}

#endif
