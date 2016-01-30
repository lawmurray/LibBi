/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_CACHE1D_HPP
#define BI_CACHE_CACHE1D_HPP

#include "Cache.hpp"
#include "../math/matrix.hpp"
#include "../math/loc_vector.hpp"

namespace bi {
/**
 * 1d cache of scalar values, stored in a vector.
 *
 * @ingroup io_cache
 *
 * @tparam T1 Scalar type.
 * @tparam CL Cache location.
 */
template<class T1, Location CL = ON_HOST>
class Cache1D: public Cache {
public:
  /**
   * Vector type.
   */
  typedef typename loc_vector<CL,T1>::type vector_type;

  /**
   * Vector reference type.
   */
  typedef typename vector_type::vector_reference_type vector_reference_type;

  /**
   * Constructor.
   *
   * @param Initial size of cache. Resized as necessary.
   */
  Cache1D(const int size = 0);

  /**
   * Shallow copy constructor.
   */
  Cache1D(const Cache1D<T1,CL>& o);

  /**
   * Deep assignment operator.
   */
  Cache1D<T1,CL>& operator=(const Cache1D<T1,CL>& o);

  /**
   * Read page.
   *
   * @param p Page index.
   *
   * @return Page.
   */
  T1 get(const int p) const;

  /**
   * Get range of pages.
   *
   * @param p Starting index of range.
   * @param len Length of range.
   *
   * @return Range.
   */
  const vector_reference_type get(const int p, const int len) const;

  /**
   * Write page.
   *
   * @param p Page index.
   * @param x Page.
   */
  void set(const int p, const T1& x);

  /**
   * Set subrange of pages.
   *
   * @tparam V1 Vector type.
   *
   * @param p Starting index of range.
   * @param len Length of range.
   * @param x Range.
   */
  template<class V1>
  void set(const int p, const int len, const V1 x);

  /**
   * Resize cache.
   */
  void resize(const int size);

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(Cache1D<T1,CL>& o);

  /**
   * Empty cache.
   */
  void empty();

private:
  /**
   * Pages.
   */
  vector_type pages;

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
inline bi::Cache1D<T1,CL>::Cache1D(const int size) :
    Cache(size), pages(size) {
  //
}

template<class T1, bi::Location CL>
bi::Cache1D<T1,CL>::Cache1D(const Cache1D<T1,CL>& o) :
    Cache(o), pages(o.pages) {
  //
}

template<class T1, bi::Location CL>
bi::Cache1D<T1,CL>& bi::Cache1D<T1,CL>::operator=(const Cache1D<T1,CL>& o) {
  Cache::operator=(o);

  pages.resize(o.pages.size(), false);
  pages = o.pages;

  return *this;
}

template<class T1, bi::Location CL>
inline T1 bi::Cache1D<T1,CL>::get(const int p) const {
  /* pre-condition */
  BI_ASSERT(isValid(p));

  return *(pages.begin() + p);
}

template<class T1, bi::Location CL>
inline const typename bi::Cache1D<T1,CL>::vector_reference_type bi::Cache1D<
    T1,CL>::get(const int p, const int len) const {
  /* pre-condition */
  BI_ASSERT(isValid(p, len));

  return subrange(pages, p, len);
}

template<class T1, bi::Location CL>
void bi::Cache1D<T1,CL>::set(const int p, const T1& x) {
  /* pre-condition */
  BI_ASSERT(p >= 0);

  if (p >= size()) {
    resize(bi::max(p + 1, 2 * size()));
  }
  *(pages.begin() + p) = x;
  setDirty(p);
  setValid(p);
}

template<class T1, bi::Location CL>
template<class V1>
void bi::Cache1D<T1,CL>::set(const int p, const int len, const V1 x) {
  /* pre-conditions */
  BI_ASSERT(p >= 0);
  BI_ASSERT(x.size() == len);

  if (p + len > size()) {
    resize(bi::max(p + len, 2 * size()));
  }
  subrange(pages, p, len) = x;
  setDirty(p, len);
  setValid(p, len);
}

template<class T1, bi::Location CL>
void bi::Cache1D<T1,CL>::resize(const int size) {
  pages.resize(size, true);
  Cache::resize(size);
}

template<class T1, bi::Location CL>
void bi::Cache1D<T1,CL>::swap(Cache1D<T1,CL>& o) {
  Cache::swap(o);
  pages.swap(o.pages);
}

template<class T1, bi::Location CL>
void bi::Cache1D<T1,CL>::empty() {
  pages.resize(0);
  Cache::empty();
}

template<class T1, bi::Location CL>
template<class Archive>
void bi::Cache1D<T1,CL>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object < Cache > (*this);
  save_resizable_vector(ar, version, pages);
}

template<class T1, bi::Location CL>
template<class Archive>
void bi::Cache1D<T1,CL>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < Cache > (*this);
  load_resizable_vector(ar, version, pages);
}

#endif
