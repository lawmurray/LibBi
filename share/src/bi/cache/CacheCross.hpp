/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_CROSSCACHE_HPP
#define BI_CACHE_CROSSCACHE_HPP

#include "Cache.hpp"
#include "../math/loc_matrix.hpp"

namespace bi {
/**
 * Cached matrix, where rows are writable and columns are readable. Useful
 * for stacking discontiguous writes in memory to later ensure a contiguous
 * write to disk.
 *
 * @ingroup io_cache
 *
 * @tparam T1 Scalar type.
 * @tparam CL Cache location.
 */
template<class T1, Location CL = ON_HOST>
class CacheCross: public Cache {
public:
  /**
   * Matrix type.
   */
  typedef typename loc_matrix<CL,T1>::type matrix_type;

  /**
   * Matrix reference type.
   */
  typedef typename matrix_type::matrix_reference_type matrix_reference_type;

  /**
   * Vector reference type.
   */
  typedef typename matrix_type::vector_reference_type vector_reference_type;

  /**
   * Constructor.
   *
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  CacheCross(const int rows = 0, const int cols = 0);

  /**
   * Shallow copy constructor.
   */
  CacheCross(const CacheCross<T1,CL>& o);

  /**
   * Deep assignment operator.
   */
  CacheCross<T1,CL>& operator=(const CacheCross<T1,CL>& o);

  /**
   * Read row.
   *
   * @param i Index of row.
   *
   * @return Row range.
   */
  const vector_reference_type get(const int i) const;

  /**
   * Read rows.
   *
   * @param i Index of first row.
   * @param len Number of rows.
   *
   * @return Row range.
   */
  const matrix_reference_type get(const int i, const int len) const;

  /**
   * Write row.
   *
   * @tparam V1 Vector type.
   *
   * @param i Row index.
   * @param x Row.
   */
  template<class V1>
  void set(const int i, const V1 x);

  /**
   * Write rows.
   *
   * @tparam M1 Matrix type.
   *
   * @param i Index of first row.
   * @param len Number of rows.
   * @param X Rows.
   */
  template<class M1>
  void set(const int i, const int len, const M1 X);

  /**
   * Resize cache.
   */
  void resize(const int rows, const int cols);

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(CacheCross<T1,CL>& o);

  /**
   * Empty cache.
   */
  void empty();

private:
  /**
   * Contents of cache.
   */
  matrix_type X;

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

template<class T1, bi::Location CL>
inline bi::CacheCross<T1,CL>::CacheCross(const int rows, const int cols) :
    Cache(rows), X(rows, cols) {
  //
}

template<class T1, bi::Location CL>
inline bi::CacheCross<T1,CL>::CacheCross(const CacheCross<T1,CL>& o) :
    Cache(o), X(o.X) {
  //
}

template<class T1, bi::Location CL>
inline bi::CacheCross<T1,CL>& bi::CacheCross<T1,CL>::operator=(
    const CacheCross<T1,CL>& o) {
  Cache::operator=(o);

  X.resize(o.X.size1(), o.X.size2(), false);
  X = o.X;

  return *this;
}

template<class T1, bi::Location CL>
inline const typename bi::CacheCross<T1,CL>::vector_reference_type bi::CacheCross<
    T1,CL>::get(const int i) const {
  /* pre-condition */
  BI_ASSERT(isValid(i));

  return row(X, i);
}

template<class T1, bi::Location CL>
inline const typename bi::CacheCross<T1,CL>::matrix_reference_type bi::CacheCross<
    T1,CL>::get(const int i, const int len) const {
  /* pre-condition */
  BI_ASSERT(isValid(i, len));

  return rows(X, i, len);
}

template<class T1, bi::Location CL>
template<class V1>
inline void bi::CacheCross<T1,CL>::set(const int p, const V1 x) {
  /* pre-condition */
  BI_ASSERT(p >= 0);

  if (p >= size()) {
    resize(p, X.size2());
  }
  row(X, p) = x;
  setDirty(p, true);
  setValid(p, true);
}

template<class T1, bi::Location CL>
template<class M1>
inline void bi::CacheCross<T1,CL>::set(const int p, const int len,
    const M1 X) {
  /* pre-condition */
  BI_ASSERT(p >= 0);

  if (p + len > size()) {
    resize(p + len, X.size2());
  }
  rows(this->X, p, len) = X;
  setDirty(p, len, true);
  setValid(p, len, true);
}

template<class T1, bi::Location CL>
void bi::CacheCross<T1,CL>::resize(const int rows, const int cols) {
  Cache::resize(rows);
  X.resize(rows, cols, true);
}

template<class T1, bi::Location CL>
void bi::CacheCross<T1,CL>::swap(CacheCross<T1,CL>& o) {
  Cache::swap(o);
  X.swap(o.X);
}

template<class T1, bi::Location CL>
void bi::CacheCross<T1,CL>::empty() {
  Cache::empty();
  X.resize(0, 0);
}

template<class T1, bi::Location CL>
template<class Archive>
void bi::CacheCross<T1,CL>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object < Cache > (*this);
  save_resizable_matrix(ar, version, X);
}

template<class T1, bi::Location CL>
template<class Archive>
void bi::CacheCross<T1,CL>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < Cache > (*this);
  load_resizable_matrix(ar, version, X);
}

#endif
