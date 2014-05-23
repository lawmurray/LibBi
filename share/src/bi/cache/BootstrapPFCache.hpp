/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_BOOTSTRAPPFCACHE_HPP
#define BI_CACHE_BOOTSTRAPPFCACHE_HPP

#include "SimulatorCache.hpp"
#include "Cache1D.hpp"
#include "AncestryCache.hpp"
#include "../buffer/ParticleFilterNetCDFBuffer.hpp"

#include "boost/serialization/split_member.hpp"
#include "boost/serialization/base_object.hpp"

namespace bi {
/**
 * Cache for ParticleFilterNetCDFBuffer reads and writes, and efficiently
 * holding ancestry tree for drawing trajectories from the filter-smoother
 * distribution.
 *
 * @ingroup io_cache
 *
 * @tparam IO1 Buffer type.
 * @tparam CL Location.
 */
template<class IO1 = ParticleFilterNetCDFBuffer, Location CL = ON_HOST>
class BootstrapPFCache: public SimulatorCache<IO1,CL> {
public:
  using SimulatorCache<IO1,CL>::writeState;

  /**
   * Constructor.
   *
   * @param out output buffer.
   */
  BootstrapPFCache(IO1* out = NULL);

  /**
   * Shallow copy.
   */
  BootstrapPFCache(const BootstrapPFCache<IO1,CL>& o);

  /**
   * Destructor.
   */
  ~BootstrapPFCache();

  /**
   * Deep assignment.
   */
  BootstrapPFCache<IO1,CL>& operator=(
      const BootstrapPFCache<IO1,CL>& o);

  /**
   * Get the most recent log-weights vector.
   *
   * @return The most recent log-weights vector to be written to the cache.
   */
  const typename Cache1D<real,CL>::vector_reference_type getLogWeights() const;

  /**
   * @copydoc ParticleFilterNetCDFBuffer::readLogWeights()
   */
  template<class V1>
  void readLogWeights(const int k, V1 lws) const;

  /**
   * @copydoc ParticleFilterNetCDFBuffer::writeLogWeights()
   */
  template<class V1>
  void writeLogWeights(const int k, const V1 lws);

  /**
   * @copydoc ParticleFilterNetCDFBuffer::readAncestors()
   */
  template<class V1>
  void readAncestors(const int k, V1 a) const;

  /**
   * @copydoc ParticleFilterNetCDFBuffer::writeAncestors()
   */
  template<class V1>
  void writeAncestors(const int k, const V1 a);

  /**
   * @copydoc ParticleFilterNetCDFBuffer::writeLL()
   */
  void writeLL(const real ll);

  /**
   * @copydoc AncestryCache::readTrajectory()
   */
  template<class M1>
  void readTrajectory(const int p, M1 X) const;

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
   */
  template<class M1, class V1>
  void writeState(const int k, const M1 X, const V1 as);

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(BootstrapPFCache<IO1,CL>& o);

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
  /**
   * Ancestry cache.
   *
   * @todo Move to ParticleMCMCCache, as not needed in context of filter only.
   */
  #ifdef ENABLE_GPU_CACHE
  AncestryCache<CL> ancestryCache;
  #else
  AncestryCache<ON_HOST> ancestryCache;
  #endif

  /**
   * Most recent log-weights.
   */
  Cache1D<real,CL> logWeightsCache;

  /**
   * Output buffer.
   */
  IO1* out;

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
 * Factory for creating BootstrapPFCache objects.
 *
 * @ingroup io_cache
 *
 * @see Forcer
 */
template<Location CL = ON_HOST>
struct BootstrapPFCacheFactory {
  /**
   * Create BootstrapPFCache.
   *
   * @return BootstrapPFCache object. Caller has ownership.
   *
   * @see BootstrapPFCache::BootstrapPFCache()
   */
  template<class IO1>
  static BootstrapPFCache<IO1,CL>* create(IO1* out = NULL) {
    return new BootstrapPFCache<IO1,CL>(out);
  }

  /**
   * Create BootstrapPFCache.
   *
   * @return BootstrapPFCache object. Caller has ownership.
   *
   * @see BootstrapPFCache::BootstrapPFCache()
   */
  static BootstrapPFCache<ParticleFilterNetCDFBuffer,CL>* create() {
    return new BootstrapPFCache<ParticleFilterNetCDFBuffer,CL>();
  }
};
}

template<class IO1, bi::Location CL>
bi::BootstrapPFCache<IO1,CL>::BootstrapPFCache(IO1* out) :
    SimulatorCache<IO1,CL>(out), out(out) {
  //
}

template<class IO1, bi::Location CL>
bi::BootstrapPFCache<IO1,CL>::BootstrapPFCache(
    const BootstrapPFCache<IO1,CL>& o) :
    SimulatorCache<IO1,CL>(o), ancestryCache(o.ancestryCache), logWeightsCache(
        o.logWeightsCache), out(o.out) {
  //
}

template<class IO1, bi::Location CL>
bi::BootstrapPFCache<IO1,CL>& bi::BootstrapPFCache<IO1,CL>::operator=(
    const BootstrapPFCache<IO1,CL>& o) {
  SimulatorCache<IO1,CL>::operator=(o);

  ancestryCache = o.ancestryCache;
  logWeightsCache = o.logWeightsCache;
  out = o.out;

  return *this;
}

template<class IO1, bi::Location CL>
bi::BootstrapPFCache<IO1,CL>::~BootstrapPFCache() {
  flush();
}

template<class IO1, bi::Location CL>
const typename bi::Cache1D<real,CL>::vector_reference_type bi::BootstrapPFCache<
    IO1,CL>::getLogWeights() const {
  return logWeightsCache.get(0, logWeightsCache.size());
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::BootstrapPFCache<IO1,CL>::readLogWeights(const int k,
    V1 lws) const {
  BI_ASSERT(out != NULL);

  out->readLogWeights(k, lws);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::BootstrapPFCache<IO1,CL>::writeLogWeights(const int k,
    const V1 lws) {
  if (out != NULL) {
    out->writeLogWeights(k, lws);
  }
  logWeightsCache.resize(lws.size());
  logWeightsCache.set(0, lws.size(), lws);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::BootstrapPFCache<IO1,CL>::readAncestors(const int k,
    V1 as) const {
  BI_ASSERT(out != NULL);

  out->readAncestors(k, as);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::BootstrapPFCache<IO1,CL>::writeAncestors(const int k,
    const V1 as) {
  if (out != NULL) {
    out->writeAncestors(k, as);
  }
}

template<class IO1, bi::Location CL>
inline void bi::BootstrapPFCache<IO1,CL>::writeLL(const real ll) {
  if (out != NULL) {
    out->writeLL(ll);
  }
}

template<class IO1, bi::Location CL>
template<class M1>
void bi::BootstrapPFCache<IO1,CL>::readTrajectory(const int p,
    M1 X) const {
  ancestryCache.readTrajectory(p, X);
}

template<class IO1, bi::Location CL>
template<class M1, class V1>
void bi::BootstrapPFCache<IO1,CL>::writeState(const int k,
    const M1 X, const V1 as) {
  SimulatorCache<IO1,CL>::writeState(k, X);
  writeAncestors(k, as);

  #if defined(ENABLE_CUDA) and !defined(ENABLE_GPU_CACHE)
  typename temp_host_matrix<real>::type X1(X.size1(), X.size2());
  typename temp_host_vector<int>::type as1(as.size());
  X1 = X;
  as1 = as;
  synchronize();

  ancestryCache.writeState(k, X1, as1);
  #else
  ancestryCache.writeState(k, X, as);
  #endif
}

template<class IO1, bi::Location CL>
void bi::BootstrapPFCache<IO1,CL>::swap(BootstrapPFCache<IO1,CL>& o) {
  SimulatorCache<IO1,CL>::swap(o);
  ancestryCache.swap(o.ancestryCache);
  logWeightsCache.swap(o.logWeightsCache);
}

template<class IO1, bi::Location CL>
void bi::BootstrapPFCache<IO1,CL>::clear() {
  SimulatorCache<IO1,CL>::clear();
  ancestryCache.clear();
}

template<class IO1, bi::Location CL>
void bi::BootstrapPFCache<IO1,CL>::empty() {
  SimulatorCache<IO1,CL>::empty();
  ancestryCache.empty();
}

template<class IO1, bi::Location CL>
void bi::BootstrapPFCache<IO1,CL>::flush() {
  //ancestryCache.flush();
  SimulatorCache<IO1,CL>::flush();
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::BootstrapPFCache<IO1,CL>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < SimulatorCache<IO1,CL> > (*this);
  ar & ancestryCache;
  ar & logWeightsCache;
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::BootstrapPFCache<IO1,CL>::load(Archive& ar,
    const unsigned version) {
  ar & boost::serialization::base_object < SimulatorCache<IO1,CL> > (*this);
  ar & ancestryCache;
  ar & logWeightsCache;
}

#endif
