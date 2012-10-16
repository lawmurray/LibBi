/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "AncestryCache.hpp"

bi::AncestryCache::AncestryCache(const Model& m) :
    particles(0, m.getDynSize()),
    size1(0) {
  //
}

void bi::AncestryCache::clear() {
  set_elements(legacies, -1);
  current.resize(0, false);
  size1 = 0;
}

void bi::AncestryCache::empty() {
  particles.resize(0, 0, false);
  ancestors.resize(0, false);
  legacies.resize(0, false);
  current.resize(0, false);
  size1 = 0;
}
