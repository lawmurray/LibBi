/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_CACHE_HPP
#define BI_CACHE_CACHE_HPP

#include "../math/matrix.hpp"

namespace bi {
/**
 * Abstract cache.
 *
 * @ingroup io_cache
 */
class Cache {
public:
  /**
   * Is page valid?
   *
   * @param p Page index.
   *
   * A page is @em valid if its contents are consistent with that on disk.
   */
  bool isValid(const int p) const;

  /**
   * Are all pages valid?
   */
  bool isValid() const;

  /**
   * Set validity flag for page.
   *
   * @param p Page index.
   * @param valid Is page valid?
   */
  void setValid(const int p, const bool valid = true);

  /**
   * Is page dirty?
   *
   * @param p Page index.
   *
   * A page is @em dirty if its contents are up-to-date, but not consistent
   * with that on disk (i.e. its contents are yet to be written to disk).
   */
  bool isDirty(const int p) const;

  /**
   * Is any page dirty?
   */
  bool isDirty() const;

  /**
   * Set dirty flag for page.
   *
   * @param p Page index.
   * @param dirty Is page dirty?
   */
  void setDirty(const int p, const bool dirty = true);

  /**
   * Clear cache.
   */
  void clear();

  /**
   * Clean cache. All pages are no longer dirty. Typically called after cache
   * is written out to file.
   */
  void clean();

  /**
   * Resize cache.
   */
  void resize(const int size);

  /**
   * Empty cache.
   */
  void empty();

private:
  /**
   * Validity of each page.
   */
  std::vector<bool> valids;

  /**
   * Dirtiness of pages. Same indexing as #valids applies.
   */
  std::vector<bool> dirties;
};
}

inline bool bi::Cache::isValid(const int p) const {
  /* pre-condition */
  BI_ASSERT(p >= 0);

  return p < (int)valids.size() && valids[p];
}

inline bool bi::Cache::isValid() const {
  return find(valids.begin(), valids.end(), false) == valids.end();
}

inline void bi::Cache::setValid(const int p, const bool valid) {
  /* pre-condition */
  BI_ASSERT(p >= 0 && p < (int)valids.size());

  valids[p] = valid;
}

inline bool bi::Cache::isDirty(const int p) const {
  /* pre-condition */
  BI_ASSERT(p >= 0);

  return p < (int)dirties.size() && dirties[p];
}

inline bool bi::Cache::isDirty() const {
  return find(dirties.begin(), dirties.end(), true) != dirties.end();
}

inline void bi::Cache::setDirty(const int p, const bool dirty) {
  /* pre-condition */
  BI_ASSERT(p >= 0 && p < (int)dirties.size());

  dirties[p] = dirty;
}

inline void bi::Cache::clear() {
  std::fill(valids.begin(), valids.end(), false);
  std::fill(dirties.begin(), dirties.end(), false);
}

inline void bi::Cache::clean() {
  std::fill(dirties.begin(), dirties.end(), false);
}

inline void bi::Cache::resize(const int size) {
  /* pre-condition */
  BI_ASSERT(size >= 0);

  valids.resize(size, false);
  dirties.resize(size, false);
}

inline void bi::Cache::empty() {
  resize(0);
}

#endif
