/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_ADAPTIVEPARTICLEFILTERCACHE_HPP
#define BI_CACHE_ADAPTIVEPARTICLEFILTERCACHE_HPP

#include "ParticleFilterCache.hpp"
#include "CacheObject.hpp"
#include "CacheCross.hpp"
#include "Cache1D.hpp"

namespace bi {
/**
 * Additional wrapper around ParticleFilterCache for use of
 * AdaptiveParticleFilter. Buffers output in memory until stopping criterion
 * is met, and then passes only surviving particles to ParticleFilterCache
 * for output.
 *
 * @ingroup io_cache
 *
 * @tparam IO1 Buffer type.
 * @tparam CL Location.
 */
template<class IO1 = ParticleFilterNetCDFBuffer, Location CL = ON_HOST>
class AdaptiveParticleFilterCache: public ParticleFilterCache<IO1,CL> {
public:
  /**
   * Constructor.
   *
   * @param out output buffer.
   */
  AdaptiveParticleFilterCache(IO1* out = NULL);

  /**
   * Shallow copy.
   */
  AdaptiveParticleFilterCache(const AdaptiveParticleFilterCache<IO1,CL>& o);

  /**
   * Destructor.
   */
  ~AdaptiveParticleFilterCache();

  /**
   * Deep assignment.
   */
  AdaptiveParticleFilterCache<IO1,CL>& operator=(
      const AdaptiveParticleFilterCache<IO1,CL>& o);

  /**
   * @copydoc SimulatorNetCDFBuffer::writeTime()
   */
  void writeTime(const int k, const real& t);

  /**
   * @copydoc ParticleFilterNetCDFBuffer::writeLogWeights()
   */
  template<class V1>
  void writeLogWeights(const int k, const V1 lws);

  /**
   * Write-through to the underlying buffer, as well as efficient caching
   * of the ancestry using AncestryCache.
   *
   * @tparam M1 Matrix type.
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param X State.
   * @param as Ancestors.
   * @param r Was resampling performed?
   */
  template<class M1, class V1>
  void writeState(const int k, const M1 X, const V1 as, const bool r);

  /**
   * Push down to ParticleFilterCache. This is a special method for
   * AdaptiveParticleFilterCache that pushes temporary storage of particles
   * into ParticleFilterCache once the stopping criterion is met.
   */
  void push();

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(AdaptiveParticleFilterCache<IO1,CL>& o);

  /**
   * Clear cache.
   */
  void clear();

  /**
   * Empty cache.
   */
  void empty();

  /**
   * Flush cache to output buffer.
   */
  void flush();

private:
  typedef typename loc_temp_matrix<CL,real>::type matrix_type;
  typedef typename loc_temp_vector<CL,real>::type vector_type;
  typedef typename loc_temp_vector<CL,int>::type int_vector_type;

  /**
   * Caches for times while adapting.
   */
  Cache1D<real,CL> timeCache;

  /**
   * Caches for particles while adapting, indexed by time.
   */
  CacheObject<matrix_type> particleCache;

  /**
   * Cache for log-weights while adapting.
   */
  CacheObject<vector_type> logWeightCache;

  /**
   * Cache for ancestry while adapting.
   */
  CacheObject<int_vector_type> ancestorCache;

  /**
   * Base time index.
   */
  int base;

  /**
   * Number of particles written while adapting.
   */
  int P;

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

/**
 * Factory for creating ParticleFilterCache objects.
 *
 * @ingroup io_cache
 *
 * @see Forcer
 */
template<Location CL = ON_HOST>
struct AdaptiveParticleFilterCacheFactory {
  /**
   * Create AdaptiveParticleFilterCache.
   *
   * @return AdaptiveParticleFilterCache object. Caller has ownership.
   *
   * @see AdaptiveParticleFilterCache::AdaptiveParticleFilterCache()
   */
  template<class IO1>
  static AdaptiveParticleFilterCache<IO1,CL>* create(IO1* out = NULL) {
    return new AdaptiveParticleFilterCache<IO1,CL>(out);
  }

  /**
   * Create AdaptiveParticleFilterCache.
   *
   * @return AdaptiveParticleFilterCache object. Caller has ownership.
   *
   * @see AdaptiveParticleFilterCache::ParticleFilterCache()
   */
  static AdaptiveParticleFilterCache<ParticleFilterNetCDFBuffer,CL>* create() {
    return new AdaptiveParticleFilterCache<ParticleFilterNetCDFBuffer,CL>();
  }
};
}

template<class IO1, bi::Location CL>
bi::AdaptiveParticleFilterCache<IO1,CL>::AdaptiveParticleFilterCache(IO1* out) :
    ParticleFilterCache<IO1,CL>(out), base(-1), P(0) {
  //
}

template<class IO1, bi::Location CL>
bi::AdaptiveParticleFilterCache<IO1,CL>::AdaptiveParticleFilterCache(
    const AdaptiveParticleFilterCache<IO1,CL>& o) :
    ParticleFilterCache<IO1,CL>(o), timeCache(o.timeCache), particleCache(
        o.particleCache), logWeightCache(o.logWeightCache), ancestorCache(
        o.ancestorCache), base(o.base), P(o.P) {
  //
}

template<class IO1, bi::Location CL>
bi::AdaptiveParticleFilterCache<IO1,CL>& bi::AdaptiveParticleFilterCache<IO1,
    CL>::operator=(const AdaptiveParticleFilterCache<IO1,CL>& o) {
  ParticleFilterCache<IO1,CL>::operator=(o);
  timeCache = o.timeCache;
  particleCache = o.particleCache;
  logWeightCache = o.logWeightCache;
  ancestorCache = o.ancestorCache;
  base = o.base;
  P = o.P;
  return *this;
}

template<class IO1, bi::Location CL>
bi::AdaptiveParticleFilterCache<IO1,CL>::~AdaptiveParticleFilterCache() {
  flush();
}

template<class IO1, bi::Location CL>
void bi::AdaptiveParticleFilterCache<IO1,CL>::writeTime(const int k,
    const real& t) {
  if (base < 0) {
    base = k;
  }
  timeCache.set(k - base, t);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::AdaptiveParticleFilterCache<IO1,CL>::writeLogWeights(const int k,
    const V1 lws) {
  int j = k - base;

  if (j >= logWeightCache.size()) {
    logWeightCache.resize(j + 1);
  }
  if (!logWeightCache.isValid(j)) {
    logWeightCache.setValid(j);
  }
  if (logWeightCache.get(j).size() < P) {
    logWeightCache.get(j).resize(P, true);
  }

  subrange(logWeightCache.get(j), P - lws.size(), lws.size()) = lws;
}

template<class IO1, bi::Location CL>
template<class M1, class V1>
void bi::AdaptiveParticleFilterCache<IO1,CL>::writeState(const int k,
    const M1 X, const V1 as, const bool r) {
  /* pre-condition */
  assert(X.size1() == as.size());

  int j = k - base;
  P += X.size1();

  if (j >= particleCache.size()) {
    particleCache.resize(j + 1);
  }
  if (!particleCache.isValid(j)) {
    particleCache.setValid(j);
  }
  if (particleCache.get(j).size1() < P) {
    particleCache.get(j).resize(P, X.size2(), true);
  }

  if (j >= ancestorCache.size()) {
    ancestorCache.resize(j + 1);
  }
  if (!ancestorCache.isValid(j)) {
    ancestorCache.setValid(j);
  }
  if (ancestorCache.get(j).size() < P) {
    ancestorCache.get(j).resize(P, true);
  }

  rows(particleCache.get(j), P - X.size1(), X.size1()) = X;
  subrange(ancestorCache.get(j), P - as.size(), as.size()) = as;
}

template<class IO1, bi::Location CL>
void bi::AdaptiveParticleFilterCache<IO1,CL>::push() {
  int k = 0;
  while (timeCache.isValid(k)) {
    ParticleFilterCache<IO1,CL>::writeTime(base + k, timeCache.get(k));
    ParticleFilterCache<IO1,CL>::writeState(base + k, particleCache.get(k),
        ancestorCache.get(k), true);
    ParticleFilterCache<IO1,CL>::writeLogWeights(base + k,
        logWeightCache.get(k));
    ++k;
  }

  timeCache.clear();
  particleCache.clear();
  logWeightCache.clear();
  ancestorCache.clear();
  base = -1;
  P = 0;
}

template<class IO1, bi::Location CL>
void bi::AdaptiveParticleFilterCache<IO1,CL>::swap(
    AdaptiveParticleFilterCache<IO1,CL>& o) {
  ParticleFilterCache<IO1,CL>::swap(o);
  timeCache.swap(o.timeCache);
  particleCache.swap(o.particleCache);
  logWeightCache.swap(o.logWeightCache);
  ancestorCache.swap(o.ancestorCache);
  std::swap(base, o.base);
  std::swap(P, o.P);
}

template<class IO1, bi::Location CL>
void bi::AdaptiveParticleFilterCache<IO1,CL>::clear() {
  ParticleFilterCache<IO1,CL>::clear();
  timeCache.clear();
  particleCache.clear();
  logWeightCache.clear();
  ancestorCache.clear();
  base = -1;
  P = 0;
}

template<class IO1, bi::Location CL>
void bi::AdaptiveParticleFilterCache<IO1,CL>::empty() {
  ParticleFilterCache<IO1,CL>::empty();
  timeCache.empty();
  particleCache.empty();
  logWeightCache.empty();
  ancestorCache.empty();
  base = -1;
  P = 0;
}

template<class IO1, bi::Location CL>
void bi::AdaptiveParticleFilterCache<IO1,CL>::flush() {
  push();
  ParticleFilterCache<IO1,CL>::flush();
  timeCache.flush();
  particleCache.flush();
  logWeightCache.flush();
  ancestorCache.flush();
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::AdaptiveParticleFilterCache<IO1,CL>::save(Archive& ar,
    const unsigned version) const {
  ar
      & boost::serialization::base_object < ParticleFilterCache<IO1,CL>
          > (*this);
  ar & particleCache;
  ar & logWeightCache;
  ar & ancestorCache;
  ar & base;
  ar & P;
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::AdaptiveParticleFilterCache<IO1,CL>::load(Archive& ar,
    const unsigned version) {
  ar
      & boost::serialization::base_object < ParticleFilterCache<IO1,CL>
          > (*this);
  ar & particleCache;
  ar & logWeightCache;
  ar & ancestorCache;
  ar & base;
  ar & P;
}

#endif
