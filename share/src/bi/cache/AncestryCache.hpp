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
  typedef typename loc_vector<CL,int>::type int_vector_type;

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
   * Read single path from the cache.
   *
   * @tparam M1 Matrix type.
   *
   * @param p Index of particle at current time.
   * @param[out] X Path. Rows index variables, columns index times.
   */
  template<class M1>
  void readPath(const int p, M1 X) const;

  /**
   * Add particles at a new time to the cache.
   *
   * @tparam M1 Matrix type.
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param X State.
   * @param as Ancestors.
   * @param r Was resampling performed? This is for optimisation only, if
   * resampling is not performed the prune step is omitted internally.
   */
  template<class M1, class V1>
  void writeState(const int k, const M1 X, const V1 as, const bool r = true);

  /**
   * @name Diagnostics
   */
  //@{
  /**
   * Report to stderr.
   */
  void report() const;
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
   * Implementation of writeState().
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
  int_vector_type as;

  /**
   * Offspring. Each entry, corresponding to a row in @p Xs, gives the
   * number of surviving children of that particle.
   */
  int_vector_type os;

  /**
   * Leaves. Each entry indicates a row in @p Xs that holds a particle of the
   * youngest generation.
   */
  int_vector_type ls;

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

#include "../host/cache/AncestryCacheHost.hpp"
#ifdef __CUDACC__
#include "../cuda/cache/AncestryCacheGPU.cuh"
#endif

#include "../resampler/misc.hpp"
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
void bi::AncestryCache<CL>::readPath(const int p, M1 X) const {
  /* pre-conditions */
  BI_ASSERT(X.size1() == Xs.size2());
  BI_ASSERT(p >= 0 && p < ls.size());

  ///@todo Implement this with scatter, so that one kernel call on device

  typename temp_host_vector<int>::type as1(as);
  synchronize(as.on_device);

  int a = *(ls.begin() + p);
  int t = X.size2() - 1;
  do {
    column(X, t) = row(Xs, a);
    a = as1(a);
    --t;
  } while (a != -1);
}

template<bi::Location CL>
template<class M1, class V1>
void bi::AncestryCache<CL>::writeState(const int k, const M1 X, const V1 as,
    const bool r) {
  writeState(X, as, r);
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
  q = 0;
}

template<bi::Location CL>
void bi::AncestryCache<CL>::prune() {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<CL == ON_DEVICE,
  AncestryCacheGPU,
  AncestryCacheHost>::type impl;
#else
  typedef AncestryCacheHost impl;
#endif
  m -= impl::prune(this->as, this->os, this->ls);
}

template<bi::Location CL>
template<class M1, class V1>
void bi::AncestryCache<CL>::insert(const M1 X, const V1 as) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<CL == ON_DEVICE,
  AncestryCacheGPU,
  AncestryCacheHost>::type impl;
#else
  typedef AncestryCacheHost impl;
#endif
  q = impl::insert(this->Xs, this->as, this->os, this->ls, q, X, as);
  m += X.size1();
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
#ifdef ENABLE_CUDA
  int newSize = oldSize + N;
#else
  int newSize = 2 * bi::max(oldSize, N);
#endif

  Xs.resize(newSize, Xs.size2(), true);
  as.resize(newSize, true);
  os.resize(newSize, true);
  subrange(os, oldSize, newSize - oldSize).clear();
  q = oldSize;

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

#if ENABLE_DIAGNOSTICS == 1
  synchronize();
  TicToc clock;
#endif

  if (m == 0) {
    init(X);
  } else {
    int_vector_type os(ls.size());
    ancestorsToOffspring(as, os);

    bi::scatter(ls, os, this->os);
    if (r) {
      prune();
    }
    if (Xs.size1() - m < X.size1()) {
      enlarge(X.size1());
    }
    insert(X, as);
  }
#if ENABLE_DIAGNOSTICS == 1
  synchronize();
  usecs = clock.toc();
  report();
#endif
}

template<bi::Location CL>
void bi::AncestryCache<CL>::report() const {
  std::cerr << "AncestryCache: ";
  std::cerr << Xs.size1() << " slots, ";
  std::cerr << m << " nodes, ";
  std::cerr << usecs << " us last write.";
  std::cerr << std::endl;
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
