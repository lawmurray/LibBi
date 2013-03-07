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
 * @tparam B Model type.
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
class AncestryCache {
public:
  /**
   * Constructor.
   */
  AncestryCache();

  /**
   * Shallow copy constructor.
   */
  AncestryCache(const AncestryCache& o);

  /**
   * Deep assignment operator.
   */
  AncestryCache& operator=(const AncestryCache& o);

  /**
   * Size of the cache (number of time points represented).
   */
  int size() const;

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
  void swap(AncestryCache& o);

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
   */
  template<class B, Location L, class V1>
  void writeState(const int t, const State<B,L>& s, const V1 as);

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
   */
  template<class M1, class V1>
  void writeState(const M1 X, const V1 as);

  /**
   * All cached particles. Rows index particles, columns index variables.
   */
  host_matrix<real,-1,-1,-1,1> particles;

  /**
   * Ancestry index. Each entry, corresponding to a row in @p particles,
   * gives the index of the row in @p particles which holds the ancestor of
   * that particle.
   */
  host_vector<int,-1,1> ancestors;

  /**
   * Legacies. Each entry, corresponding to a row in @p particles, gives the
   * latest time at which that particle is known to have a descendant.
   */
  host_vector<int,-1,1> legacies;

  /**
   * Current time index. Each entry gives the index of a row in @p particles,
   * that it holds a particle for the current time.
   */
  host_vector<int,-1,1> current;

  /**
   * Size of the cache (number of time points represented).
   */
  int size1;

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

inline int bi::AncestryCache::size() const {
  return size1;
}

template<class M1>
void bi::AncestryCache::enlarge(const M1 X) {
  int oldSize = particles.size1();
  int newSize = numOccupied + X.size1();
  if (newSize > oldSize) {
    newSize = 2*bi::max(oldSize, X.size1());
    particles.resize(newSize, X.size2(), true);
    ancestors.resize(newSize, true);
    legacies.resize(newSize, true);

    set_elements(subrange(legacies, oldSize, newSize - oldSize), -1);
  }

  /* post-conditions */
  BI_ASSERT(particles.size1() == ancestors.size());
  BI_ASSERT(particles.size1() == legacies.size());
}

template<class M1>
void bi::AncestryCache::readTrajectory(const int p, M1 X) const {
  /* pre-conditions */
  BI_ASSERT(X.size1() == particles.size2());
  BI_ASSERT(X.size2() >= size());
  BI_ASSERT(p >= 0 && p < current.size());

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

template<class B, bi::Location L, class V1>
void bi::AncestryCache::writeState(const int t, const State<B,L>& s,
    const V1 as) {
  /* pre-condition */
  BI_ASSERT(t == this->size());

  typedef typename temp_host_matrix<real>::type host_matrix_type;
  typedef typename temp_host_vector<int>::type host_vector_type;

  if (L == ON_DEVICE || V1::on_device) {
    const host_matrix_type X1(s.getDyn());
    const host_vector_type as1(as);
    synchronize();
    writeState(X1, as1);
  } else {
    writeState(s.getDyn(), as);
  }
}

template<class M1, class V1>
void bi::AncestryCache::writeState(const M1 X, const V1 as) {
  /* pre-conditions */
  BI_ASSERT(X.size1() == as.size());
  BI_ASSERT(!V1::on_device);

#ifdef ENABLE_DIAGNOSTICS
  synchronize();
  TicToc clock;
#endif

  const int P = X.size1();
  typename temp_host_vector<int>::type newAs(P);
  int p, /*q, */len, a;

  /* update ancestors and legacies */
  numOccupied = 0;
  if (current.size()) {
//#pragma omp parallel
    {
      int p, a;

//#pragma omp for
      for (int p = 0; p < P; ++p) {
        a = current(as(p));
        newAs(p) = a;
        while (a != -1 && legacies(a) < size()) {
          legacies(a) = size();
          a = ancestors(a);
          ++numOccupied;
        }
      }
    }
  } else {
    set_elements(newAs, -1);
  }

  /* enlarge cache if necessary */
  enlarge(X);

  /* write new particles */
  current.resize(P, false);
  p = 0;
  //q = 0; // for first-fit
  while (p < P) {
    /* starting index of writable range */
    if (q >= legacies.size()) {
      q = 0;
    }
    while (legacies(q) == size()) {
      ++q;
      if (q >= legacies.size()) {
        q = 0;
      }
    }

    /* length of writable range */
    len = 1;
    while (p + len < P && q + len < legacies.size()
        && legacies(q + len) < size()) {
      ++len;
    }

    /* write */
    rows(particles, q, len) = rows(X, p, len);
    subrange(ancestors, q, len) = subrange(newAs, p, len);
    set_elements(subrange(legacies, q, len), size());
    seq_elements(subrange(current, p, len), q);

    p += len;
    q += len;
  }
  ++size1;

#ifdef ENABLE_DIAGNOSTICS
  synchronize();
  usecs = clock.toc();
  report();
#endif
}

template<class Archive>
void bi::AncestryCache::save(Archive& ar, const unsigned version) const {
  save_resizable_matrix(ar, version, particles);
  save_resizable_vector(ar, version, ancestors);
  save_resizable_vector(ar, version, legacies);
  save_resizable_vector(ar, version, current);
  ar & size1;
  ar & usecs;
  ar & numOccupied;
  ar & q;
}

template<class Archive>
void bi::AncestryCache::load(Archive& ar, const unsigned version) {
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
