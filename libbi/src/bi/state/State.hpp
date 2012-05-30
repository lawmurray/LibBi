/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_STATE_HPP
#define BI_STATE_STATE_HPP

#include "../buffer/Mask.hpp"
#include "../traits/var_traits.hpp"
#include "../traits/net_traits.hpp"
#include "../math/loc_vector.hpp"
#include "../math/loc_matrix.hpp"
#ifdef ENABLE_SSE
#include "../sse/sse.hpp"
#endif

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
  typedef typename loc_matrix<L,value_type>::type matrix_type;
  typedef typename matrix_type::matrix_reference_type matrix_reference_type;
  typedef typename loc_vector<L,value_type>::type vector_type;
  typedef typename vector_type::vector_reference_type vector_reference_type;

  static const bool on_device = (L == ON_DEVICE);

  /**
   * Constructor.
   *
   * @param m Model.
   * @param P Number of trajectories to store.
   */
  State(const B& m, const int P = 1);

  /**
   * Copy constructor (deep copy).
   */
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
  void setRange(const int p, const int P);

  /**
   * Starting index of active range.
   */
  int start() const;

  /**
   * Number of trajectories in active range.
   */
  int size() const;

  /**
   * Maximum number of trajectories that can be presently stored in the
   * state.
   */
  int sizeMax() const;

  /**
   * Resize buffers.
   *
   * @param maxP Maximum number of trajectories to store.
   * @param preserve True to preserve existing values, false otherwise.
   *
   * Resizes the state to store at least @p P number of trajectories.
   * Storage for additional trajectories may be added in some contexts,
   * see #roundup. This affects the maximum size (see #maxSize), but not
   * the number of trajectories currently active (see #size and #setRange).
   * The active range of trajectories will be shrunk if necessary.
   */
  void resizeMax(const int maxP, const bool preserve = true);

  /**
   * Resize sparse buffers for observations.
   *
   * @param W Number of observations.
   * @param preserve True to preserve existing values, false otherwise.
   *
   * Resizes the state to store at least @p W observations.
   */
  void resizeSparse(const int W, const bool preserve = true);

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
  matrix_reference_type get(const VarType type);

  /**
   * Get buffer for net.
   *
   * @param type Node type.
   *
   * @return Given buffer.
   */
  const matrix_reference_type get(const VarType type) const;

  /**
   * Get alternative buffer for net.
   *
   * @param type Node type.
   *
   * @return Given buffer.
   */
  matrix_reference_type getAlt(const VarType type);

  /**
   * Get alternative buffer for net.
   *
   * @param type Node type.
   *
   * @return Given buffer.
   */
  const matrix_reference_type getAlt(const VarType type) const;

  /**
   * Get buffer for variable.
   *
   * @tparam X Variable type.
   *
   * @return Given buffer.
   */
  template<class X>
  matrix_reference_type getVar();

  /**
   * Get buffer for variable.
   *
   * @tparam X Variable type.
   *
   * @return Given buffer.
   */
  template<class X>
  const matrix_reference_type getVar() const;

  /**
   * Get buffer for variable.
   *
   * @param type Node type.
   * @param id Variable id.
   *
   * @return Given buffer.
   */
  matrix_reference_type getVar(const VarType type, const int id);

  /**
   * Get buffer for variable.
   *
   * @param type Node type.
   * @param id Variable id.
   *
   * @return Given buffer.
   */
  const matrix_reference_type getVar(const VarType type, const int id) const;

  /**
   * Get alternative buffer for variable.
   *
   * @tparam X Variable type.
   *
   * @return Given buffer.
   */
  template<class X>
  matrix_reference_type getVarAlt();

  /**
   * Get alternative buffer for variable.
   *
   * @tparam X Variable type.
   *
   * @return Given buffer.
   */
  template<class X>
  const matrix_reference_type getVarAlt() const;

  /**
   * Get alternative buffer for variable.
   *
   * @param type Node type.
   * @param id Variable id.
   *
   * @return Given buffer.
   */
  matrix_reference_type getVarAlt(const VarType type, const int id);

  /**
   * Get alternative buffer for variable.
   *
   * @param type Node type.
   * @param id Variable id.
   *
   * @return Given buffer.
   */
  const matrix_reference_type getVarAlt(const VarType type, const int id) const;

  /**
   * Get buffer for sparse variable.
   *
   * @tparam X Variable type.
   *
   * @return Given buffer.
   */
  template<class X>
  matrix_reference_type getSparseVar(const Mask<L>& mask);

  /**
   * Get buffer for sparse variable.
   *
   * @tparam X Variable type.
   *
   * @param mask Mask.
   *
   * @return Given buffer.
   */
  template<class X>
  const matrix_reference_type getSparseVar(const Mask<L>& mask) const;

  /**
   * Get buffer for sparse variable.
   *
   * @param type Node type.
   * @param id Variable id.
   * @param mask Mask.
   *
   * @return Given buffer.
   */
  matrix_reference_type getSparseVar(const VarType type, const int id, const Mask<L>& mask);

  /**
   * Get buffer for sparse variable.
   *
   * @param type Node type.
   * @param id Variable id.
   * @param mask Mask.
   *
   * @return Given buffer.
   */
  const matrix_reference_type getSparseVar(const VarType type, const int id, const Mask<L>& mask) const;

  /**
   * Get alternative buffer for sparse variable.
   *
   * @tparam X Variable type.
   *
   * @return Given buffer.
   */
  template<class X>
  matrix_reference_type getSparseVarAlt(const Mask<L>& mask);

  /**
   * Get alternative buffer for sparse variable.
   *
   * @tparam X Variable type.
   *
   * @param mask Mask.
   *
   * @return Given buffer.
   */
  template<class X>
  const matrix_reference_type getSparseVarAlt(const Mask<L>& mask) const;

  /**
   * Get alternative buffer for sparse variable.
   *
   * @param type Node type.
   * @param id Variable id.
   * @param mask Mask.
   *
   * @return Given buffer.
   */
  matrix_reference_type getSparseVarAlt(const VarType type, const int id, const Mask<L>& mask);

  /**
   * Get alternative buffer for sparse variable.
   *
   * @param type Node type.
   * @param id Variable id.
   * @param mask Mask.
   *
   * @return Given buffer.
   */
  const matrix_reference_type getSparseVarAlt(const VarType type, const int id, const Mask<L>& mask) const;

  /**
   * Get buffer of state and noise variables.
   */
  matrix_reference_type getDyn();

  /**
   * Get buffer of state and noise variables.
   */
  const matrix_reference_type getDyn() const;

  /**
   * Get buffer of all square-root covariance variables.
   */
  matrix_reference_type getStd();

  /**
   * Get buffer of all square-root covariance variables.
   */
  const matrix_reference_type getStd() const;

  /**
   * Model.
   */
  const B& m;

  /**
   * Storage for dense non-shared variables (d- and r-).
   */
  matrix_type Xdn;

  /**
   * Storage for sparse non-shared variables (o-).
   */
  matrix_type Xsp;

  /**
   * Storage for dense shared variables (f-, s- and p-).
   */
  matrix_type Kdn;

  /**
   * Storage for sparse shared variables (oy-).
   */
  matrix_type Ksp;

  /**
   * Index of starting trajectory in @p Xdn and @p Xsp.
   */
  int p;

  /**
   * Number of trajectories.
   */
  int P;

  /**
   * Number of observations in sparse buffers.
   */
  int W;

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
  static int roundup(const int P);

  /* net sizes, for convenience */
  static const int NR = net_size<typename B::RTypeList>::value;
  static const int ND = net_size<typename B::DTypeList>::value;
  static const int NP = net_size<typename B::PTypeList>::value;
  static const int NF = net_size<typename B::FTypeList>::value;
  static const int NO = net_size<typename B::OTypeList>::value;
  static const int NDX = net_size<typename B::DXTypeList>::value;
  static const int NPX = net_size<typename B::PXTypeList>::value;

  #ifndef __CUDACC__
  /**
   * Serialize.
   */
  template<class Archive>
  void serialize(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  friend class boost::serialization::access;
  #endif
};
}

template<class B, bi::Location L>
bi::State<B,L>::State(const B& m, const int P) :
    m(m),
    Xdn(roundup(P), NR + ND + NO + NDX + NR + ND), // includes dy- and ry-vars
    Xsp(Xdn.size1(), 0),
    Kdn(1, 2*NP + NF + NPX), // includes py-vars
    Ksp(1, 0),
    p(0),
    P(P),
    W(0) {
  clear();
}

template<class B, bi::Location L>
bi::State<B,L>::State(const State<B,L>& o) :
    m(o.m),
    Xdn(o.Xdn.size1(), o.Xdn.size2()),
    Xsp(o.Xsp.size1(), o.Xsp.size2()),
    Kdn(o.Kdn.size1(), o.Kdn.size2()),
    Ksp(o.Ksp.size1(), o.Ksp.size2()),
    p(o.p),
    P(o.P),
    W(o.W) {
  *this = o;
}

template<class B, bi::Location L>
bi::State<B,L>& bi::State<B,L>::operator=(const State<B,L>& o) {
  Xdn = o.Xdn;
  Xsp = o.Xsp;
  Kdn = o.Kdn;
  Ksp = o.Ksp;
  p = o.p;
  P = o.P;
  W = o.W;

  return *this;
}

template<class B, bi::Location L>
template<bi::Location L2>
bi::State<B,L>& bi::State<B,L>::operator=(const State<B,L2>& o) {
  Xdn = o.X;
  Xsp = o.Xsp;
  Kdn = o.K;
  Ksp = o.Ksp;
  p = o.p;
  P = o.P;
  W = o.W;

  return *this;
}

template<class B, bi::Location L>
inline void bi::State<B,L>::setRange(const int p, const int P) {
  /* pre-condition */
  assert (p >= 0 && p == roundup(p));
  assert (p >= 0 && P == roundup(P));

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
inline int bi::State<B,L>::sizeMax() const {
  return Xdn.size1();
}

template<class B, bi::Location L>
inline void bi::State<B,L>::resizeMax(const int maxP, const bool preserve) {
  const int maxP1 = roundup(maxP);

  Xdn.resize(maxP1, Xdn.size2(), preserve);
  Xsp.resize(maxP1, Xsp.size2(), preserve);
  if (p > sizeMax()) {
    p = sizeMax();
  }
  if (p + P > sizeMax()) {
    P = sizeMax() - p;
  }
}

template<class B, bi::Location L>
inline void bi::State<B,L>::resizeSparse(const int W, bool preserve) {
  Xsp.resize(Xsp.size1(), W, preserve);
  Ksp.resize(Ksp.size1(), W, preserve);
  this->W = W;
}

template<class B, bi::Location L>
inline void bi::State<B,L>::clear() {
  rows(Xdn, 0, P).clear();
  rows(Xsp, 0, P).clear();
  Kdn.clear();
  Ksp.clear();
}

template<class B, bi::Location L>
inline typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::get(const VarType type) {
  switch (type) {
  case R_VAR:
    return subrange(Xdn, p, P, 0, NR);
  case D_VAR:
    return subrange(Xdn, p, P, NR, ND);
  case O_VAR:
    return subrange(Xdn, p, P, NR + ND, NO);
  case DX_VAR:
    return subrange(Xdn, p, P, NR + ND + NO, NDX);
  case RY_VAR:
    return subrange(Xdn, p, P, NR + ND + NO + NDX, NR);
  case DY_VAR:
    return subrange(Xdn, p, P, NR + ND + NO + NDX + NR, ND);
  case P_VAR:
    return columns(Kdn, 0, NP);
  case PY_VAR:
    return columns(Kdn, NP, NP);
  case F_VAR:
    return columns(Kdn, 2*NP, NF);
  case PX_VAR:
    return columns(Kdn, 2*NP + NF, NPX);
  case OY_VAR:
    return Ksp;
  default:
    BI_ERROR(false, "Unsupported type");
  }
}

template<class B, bi::Location L>
inline const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::get(const VarType type) const {
  switch (type) {
  case R_VAR:
    return subrange(Xdn, p, P, 0, NR);
  case D_VAR:
    return subrange(Xdn, p, P, NR, ND);
  case O_VAR:
    return subrange(Xdn, p, P, NR + ND, NO);
  case DX_VAR:
    return subrange(Xdn, p, P, NR + ND + NO, NDX);
  case RY_VAR:
    return subrange(Xdn, p, P, NR + ND + NO + NDX, NR);
  case DY_VAR:
    return subrange(Xdn, p, P, NR + ND + NO + NDX + NR, ND);
  case P_VAR:
    return columns(Kdn, 0, NP);
  case PY_VAR:
    return columns(Kdn, NP, NP);
  case F_VAR:
    return columns(Kdn, 2*NP, NF);
  case PX_VAR:
    return columns(Kdn, 2*NP + NF, NPX);
  case OY_VAR:
    return Ksp;
  default:
    BI_ERROR(false, "Unsupported type");
  }
}

template<class B, bi::Location L>
inline typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getAlt(
    const VarType type) {
  return get(m.getAltType(type));
}

template<class B, bi::Location L>
inline const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getAlt(
    const VarType type) const {
  return get(m.getAltType(type));
}

template<class B, bi::Location L>
template<class X>
typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getVar() {
  static const VarType type = var_type<X>::value;
  static const int start = var_net_start<B,X>::value;
  static const int size = var_size<X>::value;

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
template<class X>
const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getVar() const {
  static const VarType type = var_type<X>::value;
  static const int start = var_net_start<B,X>::value;
  static const int size = var_size<X>::value;

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getVar(
    const VarType type, const int id) {
  const int start = m.getNodeStart(type, id);
  const int size = m.getNode(type, id)->getSize();

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getVar(
    const VarType type, const int id) const {
  const int start = m.getNodeStart(type, id);
  const int size = m.getNode(type, id)->getSize();

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
template<class X>
typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getVarAlt() {
  static const VarType type = alt_type<var_type<X>::value>::value;
  static const int start = var_net_start<B,X>::value;
  static const int size = var_size<X>::value;

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
template<class X>
const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getVarAlt() const {
  static const VarType type = alt_type<var_type<X>::value>::value;
  static const int start = var_net_start<B,X>::value;
  static const int size = var_size<X>::value;

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getVarAlt(
    const VarType type, const int id) {
  const VarType alt = m.getAltType(type);
  const int start = m.getNodeStart(alt, id);
  const int size = m.getNode(alt, id)->getSize();

  return columns(get(alt), start, size);
}

template<class B, bi::Location L>
const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getVarAlt(
    const VarType type, const int id) const {
  const VarType alt = m.getAltType(type);
  const int start = m.getNodeStart(alt, id);
  const int size = m.getNode(alt, id)->getSize();

  return columns(get(alt), start, size);
}

template<class B, bi::Location L>
template<class X>
typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getSparseVar(
    const Mask<L>& mask) {
  static const VarType type = var_type<X>::value;
  static const int id = var_id<X>::value;
  const int start = mask.getStart(id);
  const int size = mask.getSize(id);

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
template<class X>
const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getSparseVar(
    const Mask<L>& mask) const {
  static const VarType type = var_type<X>::value;
  static const int id = var_id<X>::value;
  const int start = mask.getStart(id);
  const int size = mask.getSize(id);

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getSparseVar(
    const VarType type, const int id, const Mask<L>& mask) {
  const int start = mask.getStart(id);
  const int size = mask.getSize(id);

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getSparseVar(
    const VarType type, const int id, const Mask<L>& mask) const {
  const int start = mask.getStart(id);
  const int size = mask.getSize(id);

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
template<class X>
typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getSparseVarAlt(
    const Mask<L>& mask) {
  static const VarType type = alt_type<var_type<X>::value>::value;
  static const int id = var_id<X>::value;
  const int start = mask.getStart(id);
  const int size = mask.getSize(id);

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
template<class X>
const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getSparseVarAlt(
    const Mask<L>& mask) const {
  static const VarType type = alt_type<var_type<X>::value>::value;
  static const int id = var_id<X>::value;
  const int start = mask.getStart(id);
  const int size = mask.getSize(id);

  return columns(get(type), start, size);
}

template<class B, bi::Location L>
typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getSparseVarAlt(
    const VarType type, const int id, const Mask<L>& mask) {
  const VarType alt = m.getAltType(type);
  const int start = mask.getStart(id);
  const int size = mask.getSize(id);

  return columns(get(alt), start, size);
}

template<class B, bi::Location L>
const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getSparseVarAlt(
    const VarType type, const int id, const Mask<L>& mask) const {
  const VarType alt = m.getAltType(type);
  const int start = mask.getStart(id);
  const int size = mask.getSize(id);

  return columns(get(alt), start, size);
}

template<class B, bi::Location L>
typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getDyn() {
  return subrange(Xdn, p, P, 0, NR + ND);
}

template<class B, bi::Location L>
const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getDyn()
    const {
  return subrange(Xdn, p, P, 0, NR + ND);
}

template<class B, bi::Location L>
typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getStd() {
  const int start = B::getStdStart();
  const int size = B::getStdSize();
  const int sqrt_size = std::sqrt(size);

  return reshape(subrange(get(DX_VAR), p, P, start, size), P*sqrt_size, sqrt_size);
}

template<class B, bi::Location L>
const typename bi::State<B,L>::matrix_reference_type bi::State<B,L>::getStd()
    const {
  const int start = B::getStdStart();
  const int size = B::getStdSize();
  const int sqrt_size = std::sqrt(size);

  return reshape(subrange(get(DX_VAR), p, P, start, size), P*sqrt_size, sqrt_size);
}

template<class B, bi::Location L>
int bi::State<B,L>::roundup(const int P) {
  int P1 = P;
  if (L == ON_DEVICE) {
    /* either < 32 or a multiple of 32 number of trajectories required */
    if (P1 > 32) {
      P1 = ((P1 + 31)/32)*32;
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

#ifndef __CUDACC__
template<class B, bi::Location L>
template<class Archive>
void bi::State<B,L>::serialize(Archive& ar, const unsigned version) {
  ar & Xdn & Xsp & Kdn & Ksp & W;
}
#endif

#endif
