/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "AncestryCache.hpp"

#include <iomanip>

bi::AncestryCache::AncestryCache() :
    particles(0, 0), size1(0) {
  //
}

bi::AncestryCache::AncestryCache(const AncestryCache& o) :
    particles(o.particles), ancestors(o.ancestors), legacies(o.legacies), current(
        o.current), size1(o.size1) {
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

int bi::AncestryCache::numSlots() const {
  return particles.size1();
}

int bi::AncestryCache::numNodes() const {
  return numSlots()
      - op_reduce(legacies, less_constant_functor<int>(size() - 1), 0,
          thrust::plus<int>());
}

int bi::AncestryCache::numFreeBlocks() const {
  int q, len = 0, blocks = 0;
  for (q = 0; q < legacies.size(); ++q) {
    if (legacies(q) < size() - 1) {
      ++len;
    } else {
      if (len > 0) {
        ++blocks;
      }
      len = 0;
    }
  }
  if (len > 0) {
    ++blocks;
  }
  return blocks;
}
int bi::AncestryCache::largestFreeBlock() const {
  int q, len = 0, maxLen = 0;
  for (q = 0; q < legacies.size(); ++q) {
    if (legacies(q) < size() - 1) {
      ++len;
      if (len > maxLen) {
        maxLen = len;
      }
    } else {
      len = 0;
    }
  }
  return maxLen;
}

void bi::AncestryCache::report() const {
  int slots = numSlots();
  int nodes = numNodes();
  int freeBlocks = numFreeBlocks();
  double largest = largestFreeBlock();
  double frag;
  if (slots == nodes) {
    frag = 0.0;
  } else {
    frag = 1.0 - largest / (slots - nodes);
  }

  std::cerr << "AncestryCache: ";
  std::cerr << slots << " slots, ";
  std::cerr << nodes << " nodes, ";
  std::cerr << freeBlocks << " free blocks with ";
  std::cerr << std::setprecision(4) << frag << "% fragmentation.";
  std::cerr << std::endl;
}
