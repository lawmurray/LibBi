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
   * Initialise the ancestry tree with the first generation of particles.
   *
   * @tparam M1 Matrix type.
   *
   * @param X Particles.
   */
  template<class M1>
  void init(const M1 X);

  /**
   * Prune the ancestry tree.
   */
  void prune();

  /**
   * Insert a new generation of particles into the tree.
   *
   * @tparam M1 Matrix type.
   * @tparam V1 Integer vector type.
   *
   * @param X Particles.
   * @param as Ancestors.
   */
  template<class M1, class V1>
  void insert(const M1 X, const V1 as);

  /**
   * Enlarge the cache.
   *
   * @param N Number of new particles for which to make room.
   *
   * Enlarges the cache to accommodate at least @p N new particles.
   */
  void enlarge(const int N);

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
   * Particles. Rows index particles, columns index variables.
   */
  matrix_type Xs;

  /**
   * Ancestors. Each entry, corresponding to a row in @p Xs, gives
   * the index of the row in @p Xs that holds the ancestor of that
   * particle, or -1 if the particle is of the first generation, and so has
   * no ancestor.
   */
  host_int_vector_type as;

  /**
   * Offspring. Each entry, corresponding to a row in @p Xs, gives the
   * number of surviving children of that particle.
   */
  host_int_vector_type os;

  /**
   * Leaves. Each entry indicates a row in @p Xs that holds a particle of the
   * youngest generation.
   */
  host_int_vector_type ls;

  /**
   * Number of surviving nodes in the cache.
   */
  int m;

  /**
   * Current position in buffer for next-fit free slot search.
   */
  int q;

  /**
   * Time taken for last write, in microseconds.
   */
  long usecs;

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

#include "../resampler/Resampler.hpp"
#include "../math/temp_vector.hpp"
#include "../math/temp_matrix.hpp"
#include "../math/view.hpp"
#include "../math/serialization.hpp"
#include "../primitive/vector_primitive.hpp"
#include "../primitive/matrix_primitive.hpp"

#include <iomanip>

template<bi::Location CL>
bi::AncestryCache<CL>::AncestryCache() :
    m(0), q(0), usecs(0) {
  //
}

template<bi::Location CL>
bi::AncestryCache<CL>::AncestryCache(const AncestryCache<CL>& o) :
    Xs(o.Xs), as(o.as), os(o.os), ls(o.ls), m(o.m), q(o.q), usecs(o.usecs) {
  //
}

template<bi::Location CL>
bi::AncestryCache<CL>& bi::AncestryCache<CL>::operator=(
    const AncestryCache<CL>& o) {
  Xs.resize(o.Xs.size1(), o.Xs.size2(), false);
  as.resize(o.as.size(), false);
  os.resize(o.os.size(), false);
  ls.resize(o.ls.size(), false);

  Xs = o.Xs;
  as = o.as;
  os = o.os;
  ls = o.ls;
  m = o.m;
  q = o.q;
  usecs = o.usecs;

  return *this;
}

template<bi::Location CL>
void bi::AncestryCache<CL>::swap(AncestryCache<CL>& o) {
  Xs.swap(o.Xs);
  as.swap(o.as);
  os.swap(o.os);
  ls.swap(o.ls);
  std::swap(m, o.m);
  std::swap(q, o.q);
  std::swap(usecs, o.usecs);
}

template<bi::Location CL>
void bi::AncestryCache<CL>::clear() {
  os.clear();
  ls.resize(0, false);
  m = 0;
  q = 0;
  usecs = 0;
}

template<bi::Location CL>
void bi::AncestryCache<CL>::empty() {
  Xs.resize(0, 0, false);
  as.resize(0, false);
  os.resize(0, false);
  ls.resize(0, false);
  m = 0;
  q = 0;
  usecs = 0;
}

template<bi::Location CL>
template<class M1>
void bi::AncestryCache<CL>::readTrajectory(const int p, M1 X) const {
  /* pre-conditions */
  BI_ASSERT(X.size1() == Xs.size2());
  BI_ASSERT(p >= 0 && p < ls.size());

  ///@todo Implement this with scatter, so that one kernel call on device
  int a = ls(p);
  int t = X.size2() - 1;
  do {
    column(X, t) = row(Xs, a);
    a = as(a);
    --t;
  } while (a != -1);
}

template<bi::Location CL>
template<class B, bi::Location L, class V1>
void bi::AncestryCache<CL>::writeState(const int t, const State<B,L>& s,
    const V1 as, const bool r) {
  host_int_vector_type as1(as);
  synchronize(V1::on_device);
  writeState(s.getDyn(), as1, r);
}

template<bi::Location CL>
template<class M1>
void bi::AncestryCache<CL>::init(const M1 X) {
  const int N = X.size1();

  Xs.resize(Xs.size1(), X.size2(), false);
  ls.resize(N, false);

  if (Xs.size1() < N) {
    enlarge(N);
  }
  rows(Xs, 0, N) = X;

  set_elements(subrange(as, 0, N), -1);
  set_elements(subrange(os, 0, N), 0);
  seq_elements(subrange(ls, 0, N), 0);
  m = N;
}

template<bi::Location CL>
void bi::AncestryCache<CL>::prune() {
  int i, j;

  for (i = 0; i < ls.size(); ++i) {
    j = ls(i);
    while (this->os(j) == 0) {
      --m;
      j = as(j);
      if (j >= 0) {
        --this->os(j);
      } else {
        break;
      }
    }
  }
}

template<bi::Location CL>
template<class M1, class V1>
void bi::AncestryCache<CL>::insert(const M1 X, const V1 as) {
  /* pre-condition */
  BI_ASSERT(X.size1() == as.size());
  BI_ASSERT(Xs.size1() - m >= X.size1());

  const int N = X.size1();
  host_int_vector_type bs(N);
  int i;

  bi::gather(as, ls, bs);
  ls.resize(N, false);

  for (i = 0; i < N; ++i) {
    while (this->os(q) > 0) {
      ++q;
      if (q == Xs.size1()) {
        q = 0;
      }
    }
    ls(i) = q;
    ++q;
    if (q == Xs.size1()) {
      q = 0;
    }
  }

  int_vector_type ls1(ls);
  bi::scatter(ls, bs, this->as);
  bi::scatter_rows(ls1, X, this->Xs);

  m += N;
}

template<bi::Location CL>
void bi::AncestryCache<CL>::enlarge(const int N) {
  /*
   * There are two heuristics that have been tried here:
   *
   * 1. newSize = oldSize + N;
   *
   *      Conservative with memory, and makes some theoretical sense to
   *      increase the number of slots linearly, but means more allocations,
   *      and CUDA memory functions are slow.
   *
   * 2. newSize = 2*bi::max(oldSize, N);
   *
   *      Fewer calls to slow CUDA memory functions, but more generous with
   *      allocations, which may be problematic on GPUs, which typically
   *      have memory sizes much smaller than main memory.
   */
  int oldSize = Xs.size1();
  int newSize = oldSize + N;

  Xs.resize(newSize, Xs.size2(), true);
  as.resize(newSize, true);
  os.resize(newSize, true);
  subrange(os, oldSize, newSize - oldSize).clear();

  /* post-conditions */
  BI_ASSERT(Xs.size1() - m >= N);
  BI_ASSERT(Xs.size1() == as.size());
  BI_ASSERT(Xs.size1() == os.size());
}

template<bi::Location CL>
template<class M1, class V1>
void bi::AncestryCache<CL>::writeState(const M1 X, const V1 as,
    const bool r) {
  /* pre-conditions */
  BI_ASSERT(X.size1() == as.size());
  BI_ASSERT(!V1::on_device);

#ifdef ENABLE_DIAGNOSTICS
  synchronize();
  TicToc clock;
#endif

  if (m == 0) {
    init(X);
  } else {
    host_int_vector_type os(X.size1());
    Resampler::ancestorsToOffspring(as, os);

    bi::scatter(ls, os, this->os);
    if (r) {
      prune();
    }
    if (Xs.size1() - m < X.size1()) {
      enlarge(X.size1());
    }
    insert(X, as);
  }
#ifdef ENABLE_DIAGNOSTICS
  synchronize();
  usecs = clock.toc();
  report();
#endif

}

template<bi::Location CL>
void bi::AncestryCache<CL>::report() const {
  int freeBlocks = numFreeBlocks();
  double largest = largestFreeBlock();
  double frag = (Xs.size1() == m) ? 0.0 : 1.0 - (largest / (Xs.size1() - m));

  std::cerr << "AncestryCache: ";
  std::cerr << Xs.size1() << " slots, ";
  std::cerr << m << " nodes, ";
  std::cerr << freeBlocks << " free blocks, ";
  std::cerr << std::setprecision(4) << frag << " fragmentation, ";
  std::cerr << usecs << " us last write.";
  std::cerr << std::endl;
}

template<bi::Location CL>
int bi::AncestryCache<CL>::numFreeBlocks() const {
  ///@bug Incorrect, as doesn't discount leaf nodes with os(q) == 0
  int q, len = 0, blocks = 0;
  for (q = 0; q < os.size(); ++q) {
    if (os(q) == 0) {
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
  ///@bug Incorrect, as doesn't discount leaf nodes with os(q) == 0
  int q, len = 0, maxLen = 0;
  for (q = 0; q < os.size(); ++q) {
    if (os(q) == 0) {
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
  save_resizable_matrix(ar, version, Xs);
  save_resizable_vector(ar, version, as);
  save_resizable_vector(ar, version, os);
  save_resizable_vector(ar, version, ls);
  ar & m;
  ar & q;
  ar & usecs;
}

template<bi::Location CL>
template<class Archive>
void bi::AncestryCache<CL>::load(Archive& ar, const unsigned version) {
  load_resizable_matrix(ar, version, Xs);
  load_resizable_vector(ar, version, as);
  load_resizable_vector(ar, version, os);
  load_resizable_vector(ar, version, ls);
  ar & m;
  ar & q;
  ar & usecs;
}

#endif
