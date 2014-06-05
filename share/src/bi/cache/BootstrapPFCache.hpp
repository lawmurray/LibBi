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
 * @tparam CL Location.
 * @tparam IO1 Buffer type.
 */
template<Location CL = ON_HOST, class IO1 = ParticleFilterNetCDFBuffer>
class BootstrapPFCache: public SimulatorCache<CL,IO1> {
public:
  using SimulatorCache<CL,IO1>::writeState;

  /**
   * Constructor.
   *
   * @param out output buffer.
   */
  BootstrapPFCache(IO1& out = NULL);

  /**
   * Shallow copy.
   */
  BootstrapPFCache(const BootstrapPFCache<CL,IO1>& o);

  /**
   * Destructor.
   */
  ~BootstrapPFCache();

  /**
   * Deep assignment.
   */
  BootstrapPFCache<CL,IO1>& operator=(const BootstrapPFCache<CL,IO1>& o);

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
   * @copydoc AncestryCache::readPath()
   */
  template<class M1>
  void readPath(const int p, M1 X) const;

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
  void swap(BootstrapPFCache<CL,IO1>& o);

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
  IO1& out;

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

template<bi::Location CL, class IO1>
bi::BootstrapPFCache<CL,IO1>::BootstrapPFCache(IO1& out) :
    SimulatorCache<CL,IO1>(out), out(out) {
  //
}

template<bi::Location CL, class IO1>
bi::BootstrapPFCache<CL,IO1>::BootstrapPFCache(
    const BootstrapPFCache<CL,IO1>& o) :
    SimulatorCache<CL,IO1>(o), ancestryCache(o.ancestryCache), logWeightsCache(
        o.logWeightsCache), out(o.out) {
  //
}

template<bi::Location CL, class IO1>
bi::BootstrapPFCache<CL,IO1>& bi::BootstrapPFCache<CL,IO1>::operator=(
    const BootstrapPFCache<CL,IO1>& o) {
  SimulatorCache<CL,IO1>::operator=(o);

  ancestryCache = o.ancestryCache;
  logWeightsCache = o.logWeightsCache;
  out = o.out;

  return *this;
}

template<bi::Location CL, class IO1>
bi::BootstrapPFCache<CL,IO1>::~BootstrapPFCache() {
  flush();
}

template<bi::Location CL, class IO1>
const typename bi::Cache1D<real,CL>::vector_reference_type bi::BootstrapPFCache<
    CL,IO1>::getLogWeights() const {
  return logWeightsCache.get(0, logWeightsCache.size());
}

template<bi::Location CL, class IO1>
template<class V1>
void bi::BootstrapPFCache<CL,IO1>::readLogWeights(const int k, V1 lws) const {
  BI_ASSERT(!(equals<IO1,OutputBuffer>::value));

  out.readLogWeights(k, lws);
}

template<bi::Location CL, class IO1>
template<class V1>
void bi::BootstrapPFCache<CL,IO1>::writeLogWeights(const int k,
    const V1 lws) {
  out.writeLogWeights(k, lws);
  logWeightsCache.resize(lws.size());
  logWeightsCache.set(0, lws.size(), lws);
}

template<bi::Location CL, class IO1>
template<class V1>
void bi::BootstrapPFCache<CL,IO1>::readAncestors(const int k, V1 as) const {
  BI_ASSERT(!(equals<IO1,OutputBuffer>::value));

  out.readAncestors(k, as);
}

template<bi::Location CL, class IO1>
template<class V1>
void bi::BootstrapPFCache<CL,IO1>::writeAncestors(const int k, const V1 as) {
  out.writeAncestors(k, as);
}

template<bi::Location CL, class IO1>
inline void bi::BootstrapPFCache<CL,IO1>::writeLL(const real ll) {
  out.writeLL(ll);
}

template<bi::Location CL, class IO1>
template<class M1>
void bi::BootstrapPFCache<CL,IO1>::readPath(const int p, M1 X) const {
  ancestryCache.readPath(p, X);
}

template<bi::Location CL, class IO1>
template<class M1, class V1>
void bi::BootstrapPFCache<CL,IO1>::writeState(const int k, const M1 X,
    const V1 as) {
  SimulatorCache<CL,IO1>::writeState(k, X);
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

template<bi::Location CL, class IO1>
void bi::BootstrapPFCache<CL,IO1>::swap(BootstrapPFCache<CL,IO1>& o) {
  SimulatorCache<CL,IO1>::swap(o);
  ancestryCache.swap(o.ancestryCache);
  logWeightsCache.swap(o.logWeightsCache);
}

template<bi::Location CL, class IO1>
void bi::BootstrapPFCache<CL,IO1>::clear() {
  SimulatorCache<CL,IO1>::clear();
  ancestryCache.clear();
}

template<bi::Location CL, class IO1>
void bi::BootstrapPFCache<CL,IO1>::empty() {
  SimulatorCache<CL,IO1>::empty();
  ancestryCache.empty();
}

template<bi::Location CL, class IO1>
void bi::BootstrapPFCache<CL,IO1>::flush() {
  //ancestryCache.flush();
  SimulatorCache<CL,IO1>::flush();
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::BootstrapPFCache<CL,IO1>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < SimulatorCache<CL,IO1> > (*this);
  ar & ancestryCache;
  ar & logWeightsCache;
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::BootstrapPFCache<CL,IO1>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < SimulatorCache<CL,IO1> > (*this);
  ar & ancestryCache;
  ar & logWeightsCache;
}

#endif
