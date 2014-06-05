/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_CACHE_HPP
#define BI_CACHE_CACHE_HPP

#include "../math/vector.hpp"

namespace bi {
/**
 * Abstract cache.
 *
 * @ingroup io_cache
 */
class Cache {
public:
  /**
   * Constructor.
   *
   * @param size Number of pages in cache.
   */
  Cache(const int size = 0);

  /**
   * Shallow copy constructor.
   */
  Cache(const Cache& o);

  /**
   * Deep assignment operator.
   */
  Cache& operator=(const Cache& o);

  /**
   * Size of cache.
   */
  int size() const;

  /**
   * Is a particular page valid?
   *
   * @param p Page index.
   *
   * A page is @em valid if its contents are consistent with that on disk.
   */
  bool isValid(const int p) const;

  /**
   * Set validity flag for page.
   *
   * @param p Page index.
   * @param valid Is page valid?
   */
  void setValid(const int p, const bool valid = true);

  /**
   * Is a particular range of pages valid?
   *
   * @param p Index of first page.
   * @param len Number of pages.
   */
  bool isValid(const int p, const int len) const;

  /**
   * Set validity flag for range of pages.
   *
   * @param p Index of first page.
   * @param len Number of pages.
   * @param valid Are pages valid?
   */
  void setValid(const int p, const int len, const bool valid = true);

  /**
   * Is a particular page dirty?
   *
   * @param p Page index.
   *
   * A page is @em dirty if its contents are up-to-date, but not consistent
   * with that on disk (i.e. its contents are yet to be written to disk).
   */
  bool isDirty(const int p) const;

  /**
   * Set dirty flag for page.
   *
   * @param p Page index.
   * @param dirty Is page dirty?
   */
  void setDirty(const int p, const bool dirty = true);

  /**
   * Is a particular range of pages dirty?
   *
   * @param p Page index.
   * @param len Number of pages.
   */
  bool isDirty(const int p, const int len) const;

  /**
   * Set dirty flag for range of pages.
   *
   * @param p Index of first page.
   * @param len Number of pages.
   * @param dirty Are pages dirty?
   */
  void setDirty(const int p, const int len, const bool dirty = true);

  /**
   * Resize cache.
   */
  void resize(const int size);

  /**
   * Clean cache. All pages are no longer dirty. Typically called after cache
   * is written out to file.
   */
  void flush();

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(Cache& o);

  /**
   * Clear cache.
   */
  void clear();

  /**
   * Empty cache.
   */
  void empty();

private:
  /**
   * Validity of each page.
   */
  host_vector<bool> valids;

  /**
   * Dirtiness of pages. Same indexing as #valids applies.
   */
  host_vector<bool> dirties;

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

inline int bi::Cache::size() const {
  return valids.size();
}

inline bool bi::Cache::isValid(const int p) const {
  /* pre-condition */
  BI_ASSERT(p >= 0);

  return p < (int)valids.size() && valids[p];
}

inline void bi::Cache::setValid(const int p, const bool valid) {
  /* pre-condition */
  BI_ASSERT(p >= 0 && p < (int )valids.size());

  valids[p] = valid;
}

inline bool bi::Cache::isDirty(const int p) const {
  /* pre-condition */
  BI_ASSERT(p >= 0);

  return p < (int)dirties.size() && dirties[p];
}

inline void bi::Cache::setDirty(const int p, const bool dirty) {
  /* pre-condition */
  BI_ASSERT(p >= 0 && p < (int )dirties.size());

  dirties[p] = dirty;
}

template<class Archive>
void bi::Cache::save(Archive& ar, const unsigned version) const {
  save_resizable_vector(ar, version, valids);
  save_resizable_vector(ar, version, dirties);
}

template<class Archive>
void bi::Cache::load(Archive& ar, const unsigned version) {
  load_resizable_vector(ar, version, valids);
  load_resizable_vector(ar, version, dirties);
}

#endif
