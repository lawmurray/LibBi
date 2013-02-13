/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "AncestryCache.hpp"

bi::AncestryCache::AncestryCache() :
    particles(0, 0),
    size1(0) {
  //
}

bi::AncestryCache::AncestryCache(const AncestryCache& o) :
    particles(o.particles),
    ancestors(o.ancestors),
    legacies(o.legacies),
    current(o.current),
    size1(o.size1) {
  //
}

bi::AncestryCache& bi::AncestryCache::operator=(const AncestryCache& o) {
  particles.resize(o.particles.size1(), o.particles.size2(), false);
  ancestors.resize(o.ancestors.size(), false);
  legacies.resize(o.legacies.size(), false);
  current.resize(o.current.size(), false);

  particles = o.particles;
  ancestors = o.ancestors;
  legacies = o.legacies;
  current = o.current;
  size1 = o.size1;

  return *this;
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
