/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STATE_MASK_HPP
#define BI_STATE_MASK_HPP

#include "../cuda/cuda.hpp"
#include "../math/loc_temp_vector.hpp"
#include "../math/loc_temp_matrix.hpp"
#include "../math/view.hpp"

#include "boost/typeof/typeof.hpp"

#include <list>

namespace bi {
/**
 * Mask.
 *
 * @ingroup state
 *
 * @tparam L Location.
 */
template<Location L = ON_HOST>
class Mask {
  friend class Mask<ON_HOST>;
  friend class Mask<ON_DEVICE>;

public:
  /**
   * Vector type.
   */
  typedef typename loc_temp_vector<L,int>::type vector_type;

  /**
   * Matrix type.
   */
  typedef typename loc_temp_matrix<L,int>::type matrix_type;

  /**
   * Default constructor.
   *
   * @param numVars Number of variables.
   */
  Mask(const int numVars = 0);

  /**
   * Shallow copy constructor.
   */
  Mask(const Mask<L>& o);

  /**
   * Generic (deep) copy constructor.
   */
  template<Location L2>
  Mask(const Mask<L2>& o);

  /**
   * Destructor.
   */
  ~Mask();

  /**
   * Assignment operator.
   */
  Mask<L>& operator=(const Mask<L>& o);

  /**
   * Generic assignment operator.
   *
   * @tparam L2 Location.
   */
  template<Location L2>
  Mask<L>& operator=(const Mask<L2>& o);

  /**
   * Add dense mask over variable.
   *
   * @param id Variable id.
   * @param size Mask size (same as variable size, as the mask is dense).
   */
  void addDenseMask(const int id, const int size);

  /**
   * Add sparse mask over variables.
   *
   * @param id Variable id.
   * @param size Mask size.
   *
   * Once the sparse mask is added, getIndices() should be used to retrieve
   * and write into the vector of serialised coordinates to indicate the
   * precise elements of the vector that form part of the mask.
   */
  void addSparseMask(const int id, const int size);

  /**
   * Get number of variables.
   */
  CUDA_FUNC_BOTH int getNumVars() const;

  /**
   * Size of mask.
   */
  CUDA_FUNC_BOTH int size() const;

  /**
   * Resize mask.
   *
   * @param numVars Number of variables.
   * @param preserve True to preserve current contents, false to clear for
   * new use.
   */
  void resize(const int numVars, const bool preserve = true);

  /**
   * Clear mask.
   */
  void clear();

  /**
   * Is a variable active in the mask and dense?
   *
   * @param id Variable id.
   *
   * @return True if the variable is active in the mask and dense, false
   * otherwise.
   */
  CUDA_FUNC_BOTH bool isDense(const int id) const;

  /**
   * Is a variable active in the mask and sparse?
   *
   * @param id Variable id.
   *
   * @return True if the variable is active in the mask and sparse, false
   * otherwise.
   */
  CUDA_FUNC_BOTH bool isSparse(const int id) const;

  /**
   * Get size of a variable in the mask.
   *
   * @param id Variable id.
   *
   * @return Size of the variable.
   */
  CUDA_FUNC_BOTH int getSize(const int id) const;

  /**
   * Translate a sparse index in the mask into a dense index.
   *
   * @param id Variable id.
   * @param i Sparse index.
   *
   * @return Dense index.
   */
  CUDA_FUNC_BOTH int getIndex(const int id, const int i) const;

  /**
   * Get serial indices for sparse variable.
   *
   * @param id Variable id.
   *
   * @return Serial indices.
   */
  CUDA_FUNC_BOTH const typename vector_type::vector_reference_type getIndices(
      const int id) const;

private:
  /**
   * Mask information. This is stored as a three-row matrix rather than as
   * three vectors to reduce the size of Mask for passing as an argument to
   * CUDA kernels. Objects of this type are 144 bytes with separate vectors,
   * 80 bytes with this matrix. Given the formal parameter limit of 256 bytes
   * as of CUDA 5.0, the saving is worth having.
   *
   * The first row is for the dense mask, sizes indexed by variable id, zero
   * if not in mask.
   *
   * The second row is for the sparse mask, sizes indexed by variable id,
   * zero if not in mask.
   *
   * The third row is offsets into #ixs, indexed by variable id.
   */
  matrix_type info;

  /**
   * Serialised coordinates for sparsely masked variables.
   */
  vector_type ixs;

  /**
   * Dense mask size.
   */
  int denseSize;

  /**
   * Sparse mask size.
   */
  int sparseSize;
};
}

template<bi::Location L>
bi::Mask<L>::Mask(const int numVars) : info(3, numVars) {
  clear();
}

template<bi::Location L>
bi::Mask<L>::Mask(const Mask<L>& o) :
    info(o.info),
    ixs(o.ixs),
    denseSize(o.denseSize),
    sparseSize(o.sparseSize) {
  //
}

template<bi::Location L>
template<bi::Location L2>
bi::Mask<L>::Mask(const Mask<L2>& o) {
  info.resize(o.info.size1(), o.info.size2());
  operator=(o);
}

template<bi::Location L>
bi::Mask<L>::~Mask() {
  //
}

template<bi::Location L>
bi::Mask<L>& bi::Mask<L>::operator=(const Mask<L>& o) {
  info.resize(3, o.getNumVars(), false);
  ixs.resize(o.ixs.size(), false);

  info = o.info;
  ixs = o.ixs;
  denseSize = o.denseSize;
  sparseSize = o.sparseSize;

  return *this;
}

template<bi::Location L>
template<bi::Location L2>
bi::Mask<L>& bi::Mask<L>::operator=(const Mask<L2>& o) {
  info.resize(3, o.getNumVars(), false);
  ixs.resize(o.ixs.size(), false);

  info = o.info;
  ixs = o.ixs;
  denseSize = o.denseSize;
  sparseSize = o.sparseSize;

  return *this;
}

template<bi::Location L>
inline int bi::Mask<L>::getNumVars() const {
  return info.size2();
}

template<bi::Location L>
void bi::Mask<L>::resize(const int numVars, const bool preserve) {
  if (preserve) {
    int oldSize = info.size2();
    info.resize(3, numVars, true);
    if (numVars > oldSize) {
      columns(info, oldSize, numVars - oldSize).clear();
    }
  } else {
    info.resize(3, numVars, false);
    clear();
  }
}

template<bi::Location L>
inline int bi::Mask<L>::size() const {
  return denseSize + sparseSize;
}

template<bi::Location L>
inline void bi::Mask<L>::clear() {
  info.clear();
  ixs.resize(0, false);
  denseSize = 0;
  sparseSize = 0;
}

template<bi::Location L>
void bi::Mask<L>::addDenseMask(const int id, const int size) {
  /* pre-condition */
  BI_ASSERT(L == ON_HOST);

  info(0, id) = size;
  denseSize += size;
}

template<bi::Location L>
void bi::Mask<L>::addSparseMask(const int id, const int size) {
  /* pre-condition */
  BI_ASSERT(L == ON_HOST);

  int start = ixs.size();

  ixs.resize(start + size, true);
  sparseSize += size;
  info(1, id) = size;
  info(2, id) = start;
}

template<bi::Location L>
inline bool bi::Mask<L>::isDense(const int id) const {
  return info(0, id) > 0;
}

template<bi::Location L>
inline bool bi::Mask<L>::isSparse(const int id) const {
  return info(1, id) > 0;
}

template<bi::Location L>
inline int bi::Mask<L>::getSize(const int id) const {
  if (isDense(id)) {
    return info(0, id);
  } else if (isSparse(id)) {
    return info(1, id);
  } else {
    return 0;
  }
}

template<bi::Location L>
inline int bi::Mask<L>::getIndex(const int id, const int i) const {
  if (isSparse(id)) {
    return ixs(info(2, id) + i);
  } else {
    return i;
  }
}

template<bi::Location L>
const typename bi::Mask<L>::vector_type::vector_reference_type bi::Mask<L>::getIndices(
    const int id) const {
  /* host_matrix::operator()() not available on device, so this gives compile
   * error, access directly instead */
  //const int start = info(2, id);
  //const int size = info(1, id);
  const int start = *(info.buf() + id*info.lead() + 2*info.inc());
  const int size = *(info.buf() + id*info.lead() + 1*info.inc());

  return subrange(ixs, start, size);
}

#endif
