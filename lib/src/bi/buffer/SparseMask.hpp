/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1318 $
 * $Date: 2011-03-01 16:27:47 +0800 (Tue, 01 Mar 2011) $
 */
#ifndef BI_BUFFER_SPARSEMASK_HPP
#define BI_BUFFER_SPARSEMASK_HPP

#include "Mask.hpp"
#include "DenseBlock.hpp"
#include "SparseBlock.hpp"

namespace bi {
/**
 * Mask over matrix, union of dense and sparse.
 *
 * @ingroup io_mask
 *
 * @tparam L Location.
 */
template<Location L = ON_HOST>
class SparseMask {
public:
  /**
   * Type of dense blocks.
   */
  typedef DenseBlock<L> dense_block_type;

  /**
   * Type of sparse blocks.
   */
  typedef SparseBlock<L> sparse_block_type;

  /**
   * Type of dense mask component.
   */
  typedef Mask<dense_block_type> dense_mask_type;

  /**
   * Type of sparse mask component.
   */
  typedef Mask<sparse_block_type> sparse_mask_type;

  /**
   * Default constructor.
   */
  SparseMask();

  /**
   * Copy constructor.
   */
  SparseMask(const SparseMask<L>& o);

  /**
   * Generic copy constructor.
   */
  template<Location L2>
  SparseMask(const SparseMask<L2>& o);

  /**
   * Assignment operator.
   */
  SparseMask<L>& operator=(const SparseMask<L>& o);

  /**
   * Generic assignment operator.
   *
   * @tparam V2 Integral vector type.
   * @tparam M2 Integral matrix type.
   */
  template<bi::Location L2>
  SparseMask<L>& operator=(const SparseMask<L2>& o);

  /**
   * Get dense mask component.
   */
  dense_mask_type& getDenseMask();

  /**
   * Get sparse mask component.
   */
  sparse_mask_type& getSparseMask();

  /**
   * Get dense mask component.
   */
  const dense_mask_type& getDenseMask() const;

  /**
   * Get sparse mask component.
   */
  const sparse_mask_type& getSparseMask() const;

  /**
   * Clear mask.
   */
  void clear();

  /**
   * Number of active variables in mask.
   */
  int size() const;

  /**
   * Translate sparse linear index into variable id and coordinate.
   *
   * @param i Sparse linear index.
   * @param[out] id Variable id.
   * @param[out] coord Coordinate.
   */
  void coord(const int i, int& id, Coord& coord) const;

  /**
   * Translate sparse linear index into variable id.
   *
   * @param i Sparse linear index.
   *
   * @return Variable id.
   */
  int id(const int i) const;

private:
  /**
   * Dense mask component.
   */
  dense_mask_type denseMask;

  /**
   * Sparse mask component.
   */
  sparse_mask_type sparseMask;
};
}

template<bi::Location L>
inline bi::SparseMask<L>::SparseMask() {
  //
}

template<bi::Location L>
inline bi::SparseMask<L>::SparseMask(const SparseMask<L>& o) :
    denseMask(o.denseMask), sparseMask(o.sparseMask) {
  //
}

template<bi::Location L>
template<bi::Location L2>
inline bi::SparseMask<L>::SparseMask(const SparseMask<L2>& o) :
    denseMask(o.getDenseMask()), sparseMask(o.getSparseMask()) {
  //
}

template<bi::Location L>
inline bi::SparseMask<L>& bi::SparseMask<L>::operator=(const SparseMask<L>& o) {
  denseMask = o.denseMask;
  sparseMask = o.sparseMask;

  return *this;
}

template<bi::Location L>
template<bi::Location L2>
inline bi::SparseMask<L>& bi::SparseMask<L>::operator=(const SparseMask<L2>& o) {
  denseMask = o.getDenseMask();
  sparseMask = o.getSparseMask();

  return *this;
}

template<bi::Location L>
inline typename bi::SparseMask<L>::dense_mask_type& bi::SparseMask<L>::getDenseMask() {
  return denseMask;
}

template<bi::Location L>
inline typename bi::SparseMask<L>::sparse_mask_type& bi::SparseMask<L>::getSparseMask() {
  return sparseMask;
}

template<bi::Location L>
inline const typename bi::SparseMask<L>::dense_mask_type& bi::SparseMask<L>::getDenseMask() const {
  return denseMask;
}

template<bi::Location L>
inline const typename bi::SparseMask<L>::sparse_mask_type& bi::SparseMask<L>::getSparseMask() const {
  return sparseMask;
}

template<bi::Location L>
inline void bi::SparseMask<L>::clear() {
  denseMask.clear();
  sparseMask.clear();
}

template<bi::Location L>
inline int bi::SparseMask<L>::size() const {
  return denseMask.size() + sparseMask.size();
}

template<bi::Location L>
inline void bi::SparseMask<L>::coord(const int i, int& id, Coord& coord) const {
  /* pre-condition */
  assert (i >= 0 && i < size());

  if (i < denseMask.size()) {
    denseMask.coord(i, id, coord);
  } else {
    sparseMask.coord(i - denseMask.size(), id, coord);
  }
}

template<bi::Location L>
inline int bi::SparseMask<L>::id(const int i) const {
  /* pre-condition */
  assert (i >= 0 && i < size());

  if (i < denseMask.size()) {
    return denseMask.id(i);
  } else {
    return sparseMask.id(i - denseMask.size());
  }
}

#endif
