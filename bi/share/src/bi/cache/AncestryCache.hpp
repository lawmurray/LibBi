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
   *
   * @param size2 Number of variables to store.
   */
  AncestryCache(const Model& m);

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
   * @param s State.
   * @param as Ancestors.
   */
  template<class B, Location L, class V1>
  void writeState(const State<B,L>& s, const V1 as);

  /**
   * Size of the cache (number of time points represented).
   */
  int size() const;

  /**
   * Clear the cache.
   */
  void clear();

  /**
   * Empty the cache.
   */
  void empty();

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
};
}

#include "../math/view.hpp"
#include "../primitive/vector_primitive.hpp"

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
void bi::AncestryCache::writeState(const State<B,L>& s, const V1 as) {
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

  const int P = X.size1();
  typename temp_host_vector<int>::type newAs(P);
  int p, q, len, maxLen, a;

  /* update ancestors and legacies */
  if (current.size()) {
    #pragma omp parallel
    {
      int p, a;

      #pragma omp for
      for (int p = 0; p < P; ++p) {
        a = current(as(p));
        newAs(p) = a;
        while (a != -1 && legacies(a) < size()) {
          legacies(a) = size();
          a = ancestors(a);
        }
      }
    }
  } else {
    set_elements(newAs, -1);
  }

  /* write as many new particles as possible into existing entries in cache,
   * by overwriting particles that have no legacy at the current time */
  current.resize(P, false);
  p = 0;
  q = 0;
  while (p < P && q < particles.size1()) {
    /* starting index of writable range */
    while (q < particles.size1() && legacies(q) == size()) {
      ++q;
    }

    if (q < particles.size1()) {
      /* length of writable range */
      maxLen = bi::min(P - p, particles.size1() - q);
      len = 1;
      while (len < maxLen && legacies(q + len) < size()) {
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
  }

  /* write any remaining particles into new entries in the cache */
  len = P - p;

  particles.resize(particles.size1() + len, particles.size2(), true);
  ancestors.resize(ancestors.size() + len, true);
  legacies.resize(legacies.size() + len, true);

  rows(particles, q, len) = rows(X, p, len);
  subrange(ancestors, q, len) = subrange(newAs, p, len);
  set_elements(subrange(legacies, q, len), size());
  seq_elements(subrange(current, p, len), q);

  ++size1;

  /* post-conditions */
  BI_ASSERT(particles.size1() == ancestors.size());
  BI_ASSERT(particles.size1() == legacies.size());
}

inline int bi::AncestryCache::size() const {
  return size1;
}

#endif
