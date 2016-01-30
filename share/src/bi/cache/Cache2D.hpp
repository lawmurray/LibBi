/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_CACHE2D_HPP
#define BI_CACHE_CACHE2D_HPP

#include "Cache.hpp"
#include "../math/loc_matrix.hpp"

namespace bi {
/**
 * 2d cache of scalar values.
 *
 * @ingroup io_cache
 *
 * @tparam T1 Scalar type.
 * @tparam CL Cache location.
 */
template<class T1, Location CL = ON_HOST>
class Cache2D: public Cache {
public:
  /**
   * Matrix type.
   */
  typedef typename loc_matrix<CL,T1>::type matrix_type;

  /**
   * Constructor.
   *
   * @param len Length of each page (number of rows).
   * @param size Number of pages (number of columns).
   */
  Cache2D(const int len = 0, const int size = 0);

  /**
   * Shallow copy constructor.
   */
  Cache2D(const Cache2D<T1,CL>& o);

  /**
   * Deep assignment operator.
   */
  Cache2D<T1,CL>& operator=(const Cache2D<T1,CL>& o);

  /**
   * Get page.
   *
   * @param j Page index.
   *
   * @return Page.
   */
  const typename matrix_type::vector_reference_type get(const int p) const;

  /**
   * Get range of pages.
   *
   * @param p Starting index of range.
   * @param len Length of range.
   *
   * @return Range of pages.
   */
  const typename matrix_type::matrix_reference_type get(const int p, const int len) const;

  /**
   * Write page.
   *
   * @tparam T2 Scalar type.
   *
   * @param p Column (page) index.
   * @param x Page.
   */
  template<class V1>
  void set(const int p, const V1 x);

  /**
   * Resize cache.
   *
   * @param len Length of each page (number of rows).
   * @param size Number of pages (number of columns).
   */
  void resize(const int len = 0, const int size = 0);

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(Cache2D<T1,CL>& o);

  /**
   * Empty cache.
   */
  void empty();

private:
  /**
   * Pages.
   */
  matrix_type pages;

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
inline bi::Cache2D<T1,CL>::Cache2D(const int len, const int size) :
    pages(len, size) {
  //
}

template<class T1, bi::Location CL>
bi::Cache2D<T1,CL>::Cache2D(const Cache2D<T1,CL>& o) :
    Cache(o), pages(o.pages) {
  //
}

template<class T1, bi::Location CL>
bi::Cache2D<T1,CL>& bi::Cache2D<T1,CL>::operator=(const Cache2D<T1,CL>& o) {
  pages.resize(o.pages.size1(), o.pages.size2(), false);
  pages = o.pages;
  Cache::operator=(o);

  return *this;
}

template<class T1, bi::Location CL>
inline const typename bi::Cache2D<T1,CL>::matrix_type::vector_reference_type bi::Cache2D<
    T1,CL>::get(const int p) const {
  /* pre-condition */
  BI_ASSERT(isValid(p));

  return column(pages, p);
}

template<class T1, bi::Location CL>
inline const typename bi::Cache2D<T1,CL>::matrix_type::matrix_reference_type bi::Cache2D<
    T1,CL>::get(const int p, const int len) const {
  /* pre-condition */
  BI_ASSERT(isValid(p, len));

  return columns(pages, p, len);
}

template<class T1, bi::Location CL>
template<class V1>
void bi::Cache2D<T1,CL>::set(const int p, const V1 x) {
  /* pre-conditions */
  BI_ASSERT(p >= 0);
  BI_ASSERT(pages.size1() == 0 || x.size() == pages.size1());

  if (pages.size1() == 0) {
    pages.resize(x.size(), pages.size2());
  }
  if (p >= size()) {
    resize(pages.size1(), bi::max(p + 1, 2 * size()));
  }
  setDirty(p);
  setValid(p);
  column(pages, p) = x;

  /* post-condition */
  BI_ASSERT(isValid(p) && isDirty(p));
}

template<class T1, bi::Location CL>
void bi::Cache2D<T1,CL>::resize(const int len, const int size) {
  pages.resize(len, size, true);
  Cache::resize(size);
}

template<class T1, bi::Location CL>
void bi::Cache2D<T1,CL>::swap(Cache2D<T1,CL>& o) {
  Cache::swap(o);
  pages.swap(o.pages);
}

template<class T1, bi::Location CL>
void bi::Cache2D<T1,CL>::empty() {
  Cache::empty();
  pages.resize(0, 0);
}

template<class T1, bi::Location CL>
template<class Archive>
void bi::Cache2D<T1,CL>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object < Cache > (*this);
  save_resizable_matrix(ar, version, pages);
}

template<class T1, bi::Location CL>
template<class Archive>
void bi::Cache2D<T1,CL>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < Cache > (*this);
  load_resizable_matrix(ar, version, pages);
}

#endif
