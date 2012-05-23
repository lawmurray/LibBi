/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_BUFFER_MASK_HPP
#define BI_BUFFER_MASK_HPP

#include "../cuda/cuda.hpp"
#include "../math/loc_temp_vector.hpp"

#include "boost/typeof/typeof.hpp"

#include <list>

namespace bi {
/**
 * Mask.
 *
 * @ingroup io_mask
 *
 * @tparam L Location.
 */
template<Location L = ON_HOST>
class Mask {
  friend class Mask<ON_HOST>;
  friend class Mask<ON_DEVICE>;

public:
  typedef typename loc_temp_vector<L,int>::type vector_type;

  /**
   * Default constructor.
   *
   * @param N Number of variables.
   */
  Mask(const int N = 0);

  /**
   * Copy constructor.
   */
  Mask(const Mask<L>& o);

  /**
   * Generic copy constructor.
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
   * @param size Variable size.
   */
  void addDenseMask(const int id, const int size);

  /**
   * Add sparse mask over variables.
   *
   * @tparam V1 Integer vector type.
   * @tparam V2 Integer vector type.
   *
   * @param ids Ids of variables.
   * @param indices Serial indices of active coordinates.
   */
  template<class V1, class V2>
  void addSparseMask(const V1 ids, const V2 indices);

  /**
   * Number of active variables in mask.
   */
  int size() const;

  /**
   * Resize. Set number of variables.
   */
  void resize(const int size);

  /**
   * Clear mask.
   */
  void clear();

  /**
   * Perform all precomputes after adding masks.
   */
  void init();

  /**
   * Is a variable active in the mask?
   *
   * @param id Variable id.
   *
   * @return True if the variable is active in the mask, false otherwise.
   */
  bool isActive(const int id) const;

  /**
   * Is a variable active in the mask and dense?
   *
   * @param id Variable id.
   *
   * @return True if the variable is active in the mask and dense, false
   * otherwise.
   */
  bool isDense(const int id) const;

  /**
   * Is a variable active in the mask and sparse?
   *
   * @param id Variable id.
   *
   * @return True if the variable is active in the mask and sparse, false
   * otherwise.
   */
  bool isSparse(const int id) const;

  /**
   * Get starting index for variable in the mask.
   *
   * @param id Variable id.
   *
   * @return Starting index of the variable with the given id.
   */
  int getStart(const int id) const;

  /**
   * Get size of a variable in the mask.
   *
   * @param id Variable id.
   *
   * @return Size of the variable.
   */
  int getSize(const int id) const;

  /**
   * Translate a sparse index in the mask into a dense index.
   *
   * @param id Variable id.
   * @param i Sparse index.
   *
   * @return Dense index.
   */
  int getIndex(const int id, const int i) const;

  /**
   * Get serial indices for sparse variable.
   *
   * @param id Variable id.
   *
   * @return Serial indices.
   */
  const vector_type* getIndices(const int id) const;

private:
  /**
   * Dense mask, sizes indexed by id, zero if not in mask.
   */
  std::vector<int> denseMask;

  /**
   * Sparse mask, indexes into #indices indexed by id, -1 if not in mask.
   */
  std::vector<int> sparseMask;

  /**
   * Starting indices, indexed by id, -1 if not in mask.
   */
  std::vector<int> starts;

  /**
   * Size of mask.
   */
  int sz;

  /**
   * Indices buffers.
   */
  std::vector<vector_type*> indices;

  /**
   * Destroy buffers.
   */
  void destroy();
};
}

template<bi::Location L>
bi::Mask<L>::Mask(const int N) : denseMask(N, 0), sparseMask(N, -1),
    starts(N, -1), sz(0) {
  //
}

template<bi::Location L>
bi::Mask<L>::Mask(const Mask<L>& o) {
  operator=(o);
}

template<bi::Location L>
template<bi::Location L2>
bi::Mask<L>::Mask(const Mask<L2>& o) {
  operator=(o);
}

template<bi::Location L>
bi::Mask<L>::~Mask() {
  destroy();
}

template<bi::Location L>
bi::Mask<L>& bi::Mask<L>::operator=(const Mask<L>& o) {
  destroy();
  indices.clear();
  denseMask = o.denseMask;
  sparseMask = o.sparseMask;
  starts = o.starts;
  sz = o.sz;

  BOOST_AUTO(iter1, indices.begin());
  BOOST_AUTO(iter2, o.indices.begin());
  for (; iter1 != indices.end(); ++iter1, ++iter2) {
    *iter1 = new vector_type(**iter2);
  }

  return *this;
}

template<bi::Location L>
template<bi::Location L2>
bi::Mask<L>& bi::Mask<L>::operator=(const Mask<L2>& o) {
  destroy();
  indices.clear();
  denseMask = o.denseMask;
  sparseMask = o.sparseMask;
  starts = o.starts;
  sz = o.sz;

  BOOST_AUTO(iter1, indices.begin());
  BOOST_AUTO(iter2, o.indices.begin());
  for (; iter1 != indices.end(); ++iter1, ++iter2) {
    *iter1 = new vector_type(**iter2);
  }

  return *this;
}

template<bi::Location L>
inline int bi::Mask<L>::size() const {
  return sz;
}

template<bi::Location L>
void bi::Mask<L>::resize(const int size) {
  denseMask.resize(size, 0);
  sparseMask.resize(size, -1);
  starts.resize(size, -1);
}

template<bi::Location L>
inline void bi::Mask<L>::clear() {
  destroy();
  std::fill(denseMask.begin(), denseMask.end(), 0);
  std::fill(sparseMask.begin(), sparseMask.end(), -1);
  std::fill(starts.begin(), starts.end(), -1);
  sz = 0;
  indices.clear();
}

template<bi::Location L>
inline void bi::Mask<L>::init() {
  int id, start = 0;
  for (id = 0; id < (int)starts.size(); ++id) {
    if (denseMask[id] > 0) {
      starts[id] = start;
      start += denseMask[id];
    } else if (sparseMask[id] != -1) {
      starts[id] = start;
      start += indices[sparseMask[id]]->size();
    } else {
      starts[id] = -1;
    }
  }
  sz = start;
}

template<bi::Location L>
void bi::Mask<L>::addDenseMask(const int id, const int size) {
  denseMask[id] = size;
  sparseMask[id] = -1;
}

template<bi::Location L>
template<class V1, class V2>
void bi::Mask<L>::addSparseMask(const V1 ids, const V2 indices) {
  const int i = this->indices.size();
  this->indices.push_back(new vector_type(indices));

  BOOST_AUTO(iter, ids.begin());
  for (; iter != ids.end(); ++iter) {
    denseMask[*iter] = 0;
    sparseMask[*iter] = i;
  }
}

template<bi::Location L>
inline void bi::Mask<L>::destroy() {
  BOOST_AUTO(iter, indices.begin());
  for (; iter != indices.end(); ++iter) {
    delete *iter;
  }
}

template<bi::Location L>
inline bool bi::Mask<L>::isActive(const int id) const {
  return starts[id] >= 0;
}

template<bi::Location L>
inline bool bi::Mask<L>::isDense(const int id) const {
  return denseMask[id] > 0;
}

template<bi::Location L>
inline bool bi::Mask<L>::isSparse(const int id) const {
  return sparseMask[id] >= 0;
}

template<bi::Location L>
inline int bi::Mask<L>::getStart(const int id) const {
  return starts[id];
}

template<bi::Location L>
inline int bi::Mask<L>::getSize(const int id) const {
  if (sparseMask[id] >= 0) {
    return indices[sparseMask[id]]->size();
  } else {
    return denseMask[id];
  }
}

template<bi::Location L>
inline int bi::Mask<L>::getIndex(const int id, const int i) const {
  if (sparseMask[id] >= 0) {
    return (*indices[sparseMask[id]])(i);
  } else {
    return i;
  }
}

template<bi::Location L>
const typename bi::Mask<L>::vector_type* bi::Mask<L>::getIndices(
    const int id) const {
  if (sparseMask[id] >= 0) {
    return indices[sparseMask[id]];
  } else {
    return NULL;
  }
}

#endif
