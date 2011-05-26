/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1254 $
 * $Date: 2011-02-02 17:36:34 +0800 (Wed, 02 Feb 2011) $
 */
#ifndef BI_BUFFER_MASK_HPP
#define BI_BUFFER_MASK_HPP

#include "../state/Coord.hpp"
#include "../cuda/cuda.hpp"

#include "boost/typeof/typeof.hpp"

#include <list>

namespace bi {
/**
 * Mask over matrix.
 *
 * @ingroup io_mask
 *
 * @tparam T Block type.
 */
template<class T>
class Mask {
public:
  typedef T block_type;
  typedef std::list<T*> blocks_type;

  /**
   * Default constructor.
   */
  Mask();

  /**
   * Copy constructor.
   */
  Mask(const Mask<T>& o);

  /**
   * Generic copy constructor.
   */
  template<class T2>
  Mask(const Mask<T2>& o);

  /**
   * Destructor.
   */
  ~Mask();

  /**
   * Assignment operator.
   */
  Mask<T>& operator=(const Mask<T>& o);

  /**
   * Generic assignment operator.
   *
   * @tparam T2 Block type.
   */
  template<class T2>
  Mask<T>& operator=(const Mask<T2>& o);

  /**
   * Clear mask.
   */
  void clear();

  /**
   * Number of active variables in mask.
   */
  int size() const;

  /**
   * Add block.
   *
   * @param block Block. Callee claims ownership.
   */
  void add(T* block);

  /**
   * Iterator to start of blocks.
   */
  typename blocks_type::iterator begin();

  /**
   * Iterator to end of blocks.
   */
  typename blocks_type::iterator end();

  /**
   * Iterator to start of blocks.
   */
  const typename blocks_type::const_iterator begin() const;

  /**
   * Iterator to end of blocks.
   */
  const typename blocks_type::const_iterator end() const;

  /**
   * Translate sparse linear index into coordinate.
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

  /**
   * Blocks.
   */
  blocks_type blocks;
};
}

template<class T>
bi::Mask<T>::Mask() {
  //
}

template<class T>
bi::Mask<T>::Mask(const Mask<T>& o) {
  operator=(o);
}

template<class T>
template<class T2>
bi::Mask<T>::Mask(const Mask<T2>& o) {
  operator=(o);
}

template<class T>
bi::Mask<T>::~Mask() {
  clear();
}

template<class T>
bi::Mask<T>& bi::Mask<T>::operator=(const Mask<T>& o) {
  blocks.resize(o.blocks.size(), NULL);
  BOOST_AUTO(iter1, begin());
  BOOST_AUTO(iter2, o.begin());
  while (iter2 != o.end()) {
    delete *iter1;
    *iter1 = new block_type();
    **iter1 = **iter2;
    ++iter1;
    ++iter2;
  }
  return *this;
}

template<class T>
template<class T2>
bi::Mask<T>& bi::Mask<T>::operator=(const Mask<T2>& o) {
  blocks.resize(o.blocks.size(), NULL);
  BOOST_AUTO(iter1, begin());
  BOOST_AUTO(iter2, o.begin());
  while (iter2 != o.end()) {
    delete *iter1;
    *iter1 = new block_type();
    **iter1 = **iter2;
    ++iter1;
    ++iter2;
  }
  return *this;
}

template<class T>
inline void bi::Mask<T>::clear() {
  BOOST_AUTO(iter, begin());
  while (iter != end()) {
    delete *iter;
    ++iter;
  }
  blocks.clear();
}

template<class T>
inline int bi::Mask<T>::size() const {
  int size = 0;
  BOOST_AUTO(iter, begin());
  while (iter != end()) {
    size += (*iter)->size();
    ++iter;
  }
  return size;
}

template<class T>
inline void bi::Mask<T>::add(T* block) {
  blocks.push_back(block);
}

template<class T>
inline typename bi::Mask<T>::blocks_type::iterator bi::Mask<T>::begin() {
  return blocks.begin();
}

template<class T>
inline typename bi::Mask<T>::blocks_type::iterator bi::Mask<T>::end() {
  return blocks.end();
}

template<class T>
inline const typename bi::Mask<T>::blocks_type::const_iterator bi::Mask<T>::begin() const {
  return blocks.begin();
}

template<class T>
inline const typename bi::Mask<T>::blocks_type::const_iterator bi::Mask<T>::end() const {
  return blocks.end();
}

template<class T>
inline void bi::Mask<T>::coord(const int i, int& id, Coord& coord) const {
  /* pre-condition */
  assert (i >= 0 && i < size());

  BOOST_AUTO(iter, begin());
  int index = i, size;
  while (iter != end()) {
    size = (*iter)->size();
    if (index < size) {
      break;
    } else {
      index -= size;
      ++iter;
    }
  }
  assert (iter != end());
  (*iter)->coord(index, id, coord);
}

template<class T>
inline int bi::Mask<T>::id(const int i) const {
  /* pre-condition */
  assert (i >= 0 && i < size());

  BOOST_AUTO(iter, begin());
  int index = i, size;
  while (iter != end()) {
    size = (*iter)->size();
    if (index < size) {
      break;
    } else {
      index -= size;
      ++iter;
    }
  }
  assert (iter != end());
  return iter->id(index);
}

#endif
