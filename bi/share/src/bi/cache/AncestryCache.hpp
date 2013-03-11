/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_ANCESTRYCACHE_HPP
#define BI_CACHE_ANCESTRYCACHE_HPP

#include "../math/vector.hpp"
#include "../math/matrix.hpp"
#include "../misc/location.hpp"
#include "../misc/TicToc.hpp"
#include "../state/State.hpp"
#include "../model/Model.hpp"

#include <vector>
#include <list>

namespace bi {
/**
 * Cache for particle ancestry tree.
 *
 * @ingroup io_cache
 *
 * @tparam CL Cache location.
 *
 * Particles are stored in a matrix with rows indexing particles and
 * columns indexing variables. Whenever particles for a new time are added,
 * their ancestry is traced back to identify those particles at earlier
 * times which have no descendant at the new time (no @em legacy in the
 * nomenclature here). The rows in the matrix corresponding to such particles
 * are marked, and may be overwritten by new particles.
 *
 * The implementation uses a single matrix in order to reduce memory
 * allocations and deallocations, which have dominated execution time in
 * previous implementations.
 */
template<Location CL = ON_HOST>
class AncestryCache {
public:
  /**
   * Matrix type.
   */
  typedef typename loc_matrix<CL,real>::type matrix_type;

  /**
   * Integer vector type.
   */
  typedef typename loc_temp_vector<CL,int>::type int_vector_type;

  /**
   * Integer vector type on host.
   */
  typedef typename temp_host_vector<int>::type host_int_vector_type;

  /**
   * Constructor.
   */
  AncestryCache();

  /**
   * Shallow copy constructor.
   */
  AncestryCache(const AncestryCache<CL>& o);

  /**
   * Deep assignment operator.
   */
  AncestryCache<CL>& operator=(const AncestryCache<CL>& o);

  /**
   * Size of the cache (number of time points represented).
   */
  int size() const;

  /**
   * Prune expired slots from cache.
   *
   * @tparam V1 Integer vector type.
   * @tparam M1 Matrix type.
   *
   * @param X Particles.
   * @param as Ancestors.
   * @param r Was resampling performed?
   */
  template<class M1, class V1>
  void prune(const M1 X, const V1 as, const bool r);

  /**
   * Enlarge the cache to accommodate a new set of particles.
   *
   * @tparam M1 Matrix type.
   *
   * @param X Particles.
   */
  template<class M1>
  void enlarge(const M1 X);

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(AncestryCache<CL>& o);

  /**
   * Clear the cache.
   */
  void clear();

  /**
   * Empty the cache.
   */
  void empty();

  /**
   * Read single trajectory from the cache.
   *
   * @tparam M1 Matrix type.
   *
   * @param p Index of particle at current time.
   * @param[out] X Trajectory. Rows index variables, columns index times.
   */
  template<class M1>
  void readTrajectory(const int p, M1 X) const;

  /**
   * Add particles at a new time to the cache.
   *
   * @tparam B Model type.
   * @tparam L Location.
   * @tparam V1 Vector type.
   *
   * @param t Time index.
   * @param s State.
   * @param as Ancestors.
   * @param r Was resampling performed?
   */
  template<class B, Location L, class V1>
  void writeState(const int t, const State<B,L>& s, const V1 as,
      const bool r);

  /**
   * @name Diagnostics
   */
  //@{
  /**
   * Report to stderr.
   */
  void report() const;

  /**
   * Number of slots in the cache.
   */
  int numSlots() const;

  /**
   * Number of slots in the cache which contain active nodes.
   */
  int numNodes() const;

  /**
   * Number of free blocks (one or more continugous free slots) in the cache.
   */
  int numFreeBlocks() const;

  /**
   * Largest free block in the cache.
   */
  int largestFreeBlock() const;
  //@}

private:
  /**
   * Host implementation of writeState().
   *
   * @tparam M1 Host matrix type.
   * @tparam V1 Host vector type.
   *
   * @param X State.
   * @param as Ancestors.
   * @param r Was resampling performed?
   */
  template<class M1, class V1>
  void writeState(const M1 X, const V1 as, const bool r);

  /**
   * All cached particles. Rows index particles, columns index variables.
   */
  matrix_type particles;

  /**
   * Ancestry index. Each entry, corresponding to a row in @p particles,
   * gives the index of the row in @p particles which holds the ancestor of
   * that particle.
   */
  host_int_vector_type ancestors;

  /**
   * Legacies. Each entry, corresponding to a row in @p particles, gives the
   * latest time at which that particle is known to have a descendant.
   */
  host_int_vector_type legacies;

  /**
   * Current time index. Each entry gives the index of a row in @p particles,
   * that it holds a particle for the current time.
   */
  host_int_vector_type current;

  /**
   * Size of the cache (number of time points represented).
   */
  int size1;

  /**
   * The legacy that is considered current.
   */
  int maxLegacy;

  /**
   * Time taken for last write, in microseconds.
   */
  int usecs;

  /**
   * Number of occupied slots in the cache.
   */
  int numOccupied;

  /**
   * Current position in buffer.
   */
  int q;

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

#include "../math/temp_vector.hpp"
#include "../math/temp_matrix.hpp"
#include "../math/view.hpp"
#include "../math/serialization.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

#include <iomanip>

template<bi::Location CL>
bi::AncestryCache<CL>::AncestryCache() :
    particles(0, 0), size1(0), maxLegacy(0), numOccupied(0), q(0) {
  //
}

template<bi::Location CL>
bi::AncestryCache<CL>::AncestryCache(const AncestryCache<CL>& o) :
    particles(o.particles), ancestors(o.ancestors), legacies(o.legacies), current(
        o.current), size1(o.size1), maxLegacy(0), numOccupied(o.numOccupied), q(
        o.q) {
  //
}

template<bi::Location CL>
bi::AncestryCache<CL>& bi::AncestryCache<CL>::operator=(const AncestryCache<CL>& o) {
  particles.resize(o.particles.size1(), o.particles.size2(), false);
  ancestors.resize(o.ancestors.size(), false);
  legacies.resize(o.legacies.size(), false);
  current.resize(o.current.size(), false);

  particles = o.particles;
  ancestors = o.ancestors;
  legacies = o.legacies;
  current = o.current;
  size1 = o.size1;
  maxLegacy = o.maxLegacy;
  numOccupied = o.numOccupied;
  q = o.q;

  return *this;
}

template<bi::Location CL>
inline int bi::AncestryCache<CL>::size() const {
  return size1;
}

template<bi::Location CL>
template<class M1, class V1>
void bi::AncestryCache<CL>::prune(const M1 X, const V1 as, const bool r) {
  /* pre-conditions */
  BI_ASSERT(!V1::on_device);

  host_int_vector_type current1(current);

  if (maxLegacy == 0 || (r/* && numSlots() - numNodes() < X.size1()*/)) {
    ++maxLegacy;
    numOccupied = 0;
    if (current.size()) {
      for (int p = 0; p < as.size(); ++p) {
        int a = current(as(p));
        while (a != -1 && legacies(a) < maxLegacy) {
          legacies(a) = maxLegacy;
          a = ancestors(a);
          ++numOccupied;
        }
      }
    }
  }
}

template<bi::Location CL>
template<class M1>
void bi::AncestryCache<CL>::enlarge(const M1 X) {
  int oldSize = particles.size1();
  int newSize = numOccupied + X.size1();
  if (newSize > oldSize) {
    newSize = /*oldSize + X.size1()*/2 * bi::max(oldSize, X.size1());
    particles.resize(newSize, X.size2(), true);
    ancestors.resize(newSize, true);
    legacies.resize(newSize, true);
    subrange(legacies, oldSize, newSize - oldSize).clear();
  }

  /* post-conditions */
  BI_ASSERT(numSlots() - numNodes() >= X.size1());
  BI_ASSERT(particles.size1() == ancestors.size());
  BI_ASSERT(particles.size1() == legacies.size());
}

template<bi::Location CL>
void bi::AncestryCache<CL>::swap(AncestryCache<CL>& o) {
  particles.swap(o.particles);
  ancestors.swap(o.ancestors);
  legacies.swap(o.legacies);
  current.swap(o.current);
  std::swap(size1, o.size1);
  std::swap(maxLegacy, o.maxLegacy);
  std::swap(usecs, o.usecs);
  std::swap(numOccupied, o.numOccupied);
  std::swap(q, o.q);
}

template<bi::Location CL>
void bi::AncestryCache<CL>::clear() {
  legacies.clear();
  current.resize(0, false);
  size1 = 0;
  maxLegacy = 0;
  numOccupied = 0;
  q = 0;
}

template<bi::Location CL>
void bi::AncestryCache<CL>::empty() {
  particles.resize(0, 0, false);
  ancestors.resize(0, false);
  legacies.resize(0, false);
  current.resize(0, false);
  size1 = 0;
  maxLegacy = 0;
  numOccupied = 0;
  q = 0;
}

template<bi::Location CL>
template<class M1>
void bi::AncestryCache<CL>::readTrajectory(const int p, M1 X) const {
  /* pre-conditions */
  BI_ASSERT(X.size1() == particles.size2());
  BI_ASSERT(X.size2() >= size());
  BI_ASSERT(p >= 0 && p < current.size());

  ///@todo Implement this with scatter, so that one kernel call on device
  if (size() > 0) {
    int a = current(p);
    int t = size() - 1;
    while (a != -1) {
      BI_ASSERT(t >= 0);
      column(X, t) = row(particles, a);
      a = ancestors(a);
      --t;
    }
    BI_ASSERT(t == -1);
  }
}

template<bi::Location CL>
template<class B, bi::Location L, class V1>
void bi::AncestryCache<CL>::writeState(const int t, const State<B,L>& s,
    const V1 as, const bool r) {
  /* pre-condition */
  BI_ASSERT(t == this->size());

  host_int_vector_type as1(as);
  synchronize(V1::on_device);
  writeState(s.getDyn(), as1, r);
}

template<bi::Location CL>
template<class M1, class V1>
void bi::AncestryCache<CL>::writeState(const M1 X, const V1 as, const bool r) {
  /* pre-conditions */
  BI_ASSERT(X.size1() == as.size());
  BI_ASSERT(!V1::on_device);

#ifdef ENABLE_DIAGNOSTICS
  synchronize();
  TicToc clock;
#endif

  const int P = X.size1();
  host_int_vector_type newAs(P);
  int p, len, a;

  /* update ancestors and legacies */
  prune(X, as, r);

  /* enlarge cache if necessary */
  enlarge(X);

  /* remap ancestors */
  if (current.size()) {
    for (int p = 0; p < as.size(); ++p) {
      a = current(as(p));
      newAs(p) = a;
    }
  } else {
    set_elements(newAs, -1);
  }

  /* write new particles */
  current.resize(P, false);
  for (p = 0; p < P; ++p) {
    /* search for free slot for this particle */
    while (legacies(q) == maxLegacy) {
      q = (q + 1) % legacies.size();
    }
    legacies(q) = maxLegacy;
    current(p) = q;
    q = (q + 1) % legacies.size();
  }
  bi::scatter(current, newAs, ancestors);
  int_vector_type current1(current);
  bi::scatter_rows(current1, X, particles);
  numOccupied += P;
  ++size1;

#ifdef ENABLE_DIAGNOSTICS
  synchronize();
  usecs = clock.toc();
  report();
#endif
}

template<bi::Location CL>
void bi::AncestryCache<CL>::report() const {
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
  std::cerr << freeBlocks << " free blocks, ";
  std::cerr << std::setprecision(4) << frag << "% fragmentation, ";
  std::cerr << usecs << " us last write.";
  std::cerr << std::endl;
}

template<bi::Location CL>
inline int bi::AncestryCache<CL>::numSlots() const {
  return particles.size1();
}

template<bi::Location CL>
inline int bi::AncestryCache<CL>::numNodes() const {
  return numOccupied;
}

template<bi::Location CL>
int bi::AncestryCache<CL>::numFreeBlocks() const {
  int q, len = 0, blocks = 0;
  for (q = 0; q < legacies.size(); ++q) {
    if (legacies(q) < maxLegacy) {
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

template<bi::Location CL>
int bi::AncestryCache<CL>::largestFreeBlock() const {
  int q, len = 0, maxLen = 0;
  for (q = 0; q < legacies.size(); ++q) {
    if (legacies(q) < maxLegacy) {
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

template<bi::Location CL>
template<class Archive>
void bi::AncestryCache<CL>::save(Archive& ar, const unsigned version) const {
  save_resizable_matrix(ar, version, particles);
  save_resizable_vector(ar, version, ancestors);
  save_resizable_vector(ar, version, legacies);
  save_resizable_vector(ar, version, current);
  ar & size1;
  ar & usecs;
  ar & numOccupied;
  ar & q;
}

template<bi::Location CL>
template<class Archive>
void bi::AncestryCache<CL>::load(Archive& ar, const unsigned version) {
  load_resizable_matrix(ar, version, particles);
  load_resizable_vector(ar, version, ancestors);
  load_resizable_vector(ar, version, legacies);
  load_resizable_vector(ar, version, current);
  ar & size1;
  ar & usecs;
  ar & numOccupied;
  ar & q;
}

#endif
