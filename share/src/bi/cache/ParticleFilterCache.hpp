/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_PARTICLEFILTERCACHE_HPP
#define BI_CACHE_PARTICLEFILTERCACHE_HPP

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
class ParticleFilterCache: public SimulatorCache<IO1,CL> {
public:
  using SimulatorCache<IO1,CL>::writeState;

  /**
   * Constructor.
   *
   * @param out output buffer.
   */
  ParticleFilterCache(IO1* out = NULL);

  /**
   * Shallow copy.
   */
  ParticleFilterCache(const ParticleFilterCache<IO1,CL>& o);

  /**
   * Destructor.
   */
  ~ParticleFilterCache();

  /**
   * Deep assignment.
   */
  ParticleFilterCache<IO1,CL>& operator=(
      const ParticleFilterCache<IO1,CL>& o);

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
   * @param r Was resampling performed?
   */
  template<class M1, class V1>
  void writeState(const int k, const M1 X, const V1 as, const bool r);

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(ParticleFilterCache<IO1,CL>& o);

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
   */
  AncestryCache<CL> ancestryCache;

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
 * Factory for creating ParticleFilterCache objects.
 *
 * @ingroup io_cache
 *
 * @see Forcer
 */
template<Location CL = ON_HOST>
struct ParticleFilterCacheFactory {
  /**
   * Create ParticleFilterCache.
   *
   * @return ParticleFilterCache object. Caller has ownership.
   *
   * @see ParticleFilterCache::ParticleFilterCache()
   */
  template<class IO1>
  static ParticleFilterCache<IO1,CL>* create(IO1* out = NULL) {
    return new ParticleFilterCache<IO1,CL>(out);
  }

  /**
   * Create ParticleFilterCache.
   *
   * @return ParticleFilterCache object. Caller has ownership.
   *
   * @see ParticleFilterCache::ParticleFilterCache()
   */
  static ParticleFilterCache<ParticleFilterNetCDFBuffer,CL>* create() {
    return new ParticleFilterCache<ParticleFilterNetCDFBuffer,CL>();
  }
};
}

template<class IO1, bi::Location CL>
bi::ParticleFilterCache<IO1,CL>::ParticleFilterCache(IO1* out) :
    SimulatorCache<IO1,CL>(out), out(out) {
  //
}

template<class IO1, bi::Location CL>
bi::ParticleFilterCache<IO1,CL>::ParticleFilterCache(
    const ParticleFilterCache<IO1,CL>& o) :
    SimulatorCache<IO1,CL>(o), ancestryCache(o.ancestryCache), logWeightsCache(
        o.logWeightsCache), out(o.out) {
  //
}

template<class IO1, bi::Location CL>
bi::ParticleFilterCache<IO1,CL>& bi::ParticleFilterCache<IO1,CL>::operator=(
    const ParticleFilterCache<IO1,CL>& o) {
  SimulatorCache<IO1,CL>::operator=(o);

  ancestryCache = o.ancestryCache;
  logWeightsCache = o.logWeightsCache;
  out = o.out;

  return *this;
}

template<class IO1, bi::Location CL>
bi::ParticleFilterCache<IO1,CL>::~ParticleFilterCache() {
  flush();
}

template<class IO1, bi::Location CL>
const typename bi::Cache1D<real,CL>::vector_reference_type bi::ParticleFilterCache<
    IO1,CL>::getLogWeights() const {
  return logWeightsCache.get(0, logWeightsCache.size());
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::ParticleFilterCache<IO1,CL>::readLogWeights(const int k,
    V1 lws) const {
  BI_ASSERT(out != NULL);

  out->readLogWeights(k, lws);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::ParticleFilterCache<IO1,CL>::writeLogWeights(const int k,
    const V1 lws) {
  if (out != NULL) {
    out->writeLogWeights(k, lws);
  }
  logWeightsCache.resize(lws.size());
  logWeightsCache.set(0, lws.size(), lws);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::ParticleFilterCache<IO1,CL>::readAncestors(const int k,
    V1 as) const {
  BI_ASSERT(out != NULL);

  out->readAncestors(k, as);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::ParticleFilterCache<IO1,CL>::writeAncestors(const int k,
    const V1 as) {
  if (out != NULL) {
    out->writeAncestors(k, as);
  }
}

template<class IO1, bi::Location CL>
inline void bi::ParticleFilterCache<IO1,CL>::writeLL(const real ll) {
  if (out != NULL) {
    out->writeLL(ll);
  }
}

template<class IO1, bi::Location CL>
template<class M1>
void bi::ParticleFilterCache<IO1,CL>::readTrajectory(const int p,
    M1 X) const {
  ancestryCache.readTrajectory(p, X);
}

template<class IO1, bi::Location CL>
template<class M1, class V1>
void bi::ParticleFilterCache<IO1,CL>::writeState(const int k,
    const M1 X, const V1 as, const bool r) {
  SimulatorCache<IO1,CL>::writeState(k, X);
  writeAncestors(k, as);
  ancestryCache.writeState(k, X, as, r);
}

template<class IO1, bi::Location CL>
void bi::ParticleFilterCache<IO1,CL>::swap(ParticleFilterCache<IO1,CL>& o) {
  SimulatorCache<IO1,CL>::swap(o);
  ancestryCache.swap(o.ancestryCache);
  logWeightsCache.swap(o.logWeightsCache);
}

template<class IO1, bi::Location CL>
void bi::ParticleFilterCache<IO1,CL>::clear() {
  SimulatorCache<IO1,CL>::clear();
  ancestryCache.clear();
}

template<class IO1, bi::Location CL>
void bi::ParticleFilterCache<IO1,CL>::empty() {
  SimulatorCache<IO1,CL>::empty();
  ancestryCache.empty();
}

template<class IO1, bi::Location CL>
void bi::ParticleFilterCache<IO1,CL>::flush() {
  //ancestryCache.flush();
  SimulatorCache<IO1,CL>::flush();
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::ParticleFilterCache<IO1,CL>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < SimulatorCache<IO1,CL> > (*this);
  ar & ancestryCache;
  ar & logWeightsCache;
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::ParticleFilterCache<IO1,CL>::load(Archive& ar,
    const unsigned version) {
  ar & boost::serialization::base_object < SimulatorCache<IO1,CL> > (*this);
  ar & ancestryCache;
  ar & logWeightsCache;
}

#endif
