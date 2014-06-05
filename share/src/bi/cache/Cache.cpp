/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Cache.hpp"

#include "../math/view.hpp"
#include "../primitive/vector_primitive.hpp"

bi::Cache::Cache(const int size) :
    valids(size), dirties(size) {
  clear();
}

bi::Cache::Cache(const Cache& o) :
    valids(o.valids), dirties(o.dirties) {
  //
}

bi::Cache& bi::Cache::operator=(const Cache& o) {
  valids.resize(o.valids.size(), false);
  dirties.resize(o.dirties.size(), false);

  valids = o.valids;
  dirties = o.dirties;

  return *this;
}

bool bi::Cache::isValid(const int p, const int len) const {
  BOOST_AUTO(tmp, subrange(valids, p, len));
  return std::find(tmp.begin(), tmp.end(), false) == tmp.end();
}

void bi::Cache::setValid(const int p, const int len, const bool valid) {
  /* pre-condition */
  BI_ASSERT(p >= 0 && p + len <= (int )valids.size());

  set_elements(subrange(valids, p, len), valid);
}

bool bi::Cache::isDirty(const int p, const int len) const {
  BOOST_AUTO(tmp, subrange(dirties, p, len));
  return std::find(tmp.begin(), tmp.end(), true) != tmp.end();
}

void bi::Cache::setDirty(const int p, const int len, const bool dirty) {
  /* pre-condition */
  BI_ASSERT(p >= 0 && p + len <= (int )dirties.size());

  set_elements(subrange(dirties, p, len), dirty);
}

void bi::Cache::resize(const int size) {
  /* pre-condition */
  BI_ASSERT(size >= 0);

  int oldSize = this->size();
  valids.resize(size, true);  // true is to preserve contents here
  dirties.resize(size, true);
  if (size > oldSize) {
    set_elements(subrange(valids, oldSize, size - oldSize), false);
    set_elements(subrange(dirties, oldSize, size - oldSize), false);
  }
}

void bi::Cache::swap(Cache& o) {
  valids.swap(o.valids);
  dirties.swap(o.dirties);
}

void bi::Cache::flush() {
  set_elements(dirties, false);
}

void bi::Cache::clear() {
  set_elements(valids, false);
  set_elements(dirties, false);
}

void bi::Cache::empty() {
  resize(0);
}
