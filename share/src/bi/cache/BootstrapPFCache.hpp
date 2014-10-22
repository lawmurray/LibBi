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
#include "../null/ParticleFilterNullBuffer.hpp"

#include "boost/serialization/split_member.hpp"
#include "boost/serialization/base_object.hpp"

namespace bi {
/**
 * Cache for particle filter.
 *
 * @ingroup io_cache
 *
 * @tparam CL Location.
 * @tparam IO1 Buffer type.
 */
template<Location CL = ON_HOST, class IO1 = ParticleFilterNullBuffer>
class BootstrapPFCache: public SimulatorCache<CL,IO1> {
public:
  typedef SimulatorCache<CL,IO1> parent_type;

  /**
   * @copydoc ParticleFilterBuffer::ParticleFilterBuffer()
   */
  BootstrapPFCache(const Model& m, const size_t P = 0, const size_t T = 0,
      const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = DEFAULT);

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
   * @copydoc ParticleFilterNetCDFBuffer::writeLogWeights()
   */
  template<class V1>
  void writeLogWeights(const int k, const V1 lws);

  /**
   * @copydoc AncestryCache::readPath()
   */
  template<class M1>
  void readPath(const int p, M1 X) const;

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
bi::BootstrapPFCache<CL,IO1>::BootstrapPFCache(const Model& m, const size_t P,
    const size_t T, const std::string& file, const FileMode mode,
    const SchemaMode schema) :
    parent_type(m, P, T, file, mode, schema) {
  //
}

template<bi::Location CL, class IO1>
bi::BootstrapPFCache<CL,IO1>::BootstrapPFCache(
    const BootstrapPFCache<CL,IO1>& o) :
    parent_type(o), ancestryCache(o.ancestryCache), logWeightsCache(
        o.logWeightsCache) {
  //
}

template<bi::Location CL, class IO1>
bi::BootstrapPFCache<CL,IO1>& bi::BootstrapPFCache<CL,IO1>::operator=(
    const BootstrapPFCache<CL,IO1>& o) {
  parent_type::operator=(o);

  ancestryCache = o.ancestryCache;
  logWeightsCache = o.logWeightsCache;

  return *this;
}

template<bi::Location CL, class IO1>
bi::BootstrapPFCache<CL,IO1>::~BootstrapPFCache() {
  //
}

template<bi::Location CL, class IO1>
const typename bi::Cache1D<real,CL>::vector_reference_type bi::BootstrapPFCache<
    CL,IO1>::getLogWeights() const {
  return logWeightsCache.get(0, logWeightsCache.size());
}

template<bi::Location CL, class IO1>
template<class M1, class V1>
void bi::BootstrapPFCache<CL,IO1>::writeState(const int k, const M1 X,
    const V1 as) {
  parent_type::writeState(k, X, as);

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
template<class V1>
void bi::BootstrapPFCache<CL,IO1>::writeLogWeights(const int k,
    const V1 lws) {
  parent_type::writeLogWeights(k, lws);
  logWeightsCache.resize(lws.size());
  logWeightsCache.set(0, lws.size(), lws);
}

template<bi::Location CL, class IO1>
template<class M1>
void bi::BootstrapPFCache<CL,IO1>::readPath(const int p, M1 X) const {
  ancestryCache.readPath(p, X);
}

template<bi::Location CL, class IO1>
void bi::BootstrapPFCache<CL,IO1>::swap(BootstrapPFCache<CL,IO1>& o) {
  parent_type::swap(o);
  ancestryCache.swap(o.ancestryCache);
  logWeightsCache.swap(o.logWeightsCache);
}

template<bi::Location CL, class IO1>
void bi::BootstrapPFCache<CL,IO1>::clear() {
  parent_type::clear();
  ancestryCache.clear();
}

template<bi::Location CL, class IO1>
void bi::BootstrapPFCache<CL,IO1>::empty() {
  parent_type::empty();
  ancestryCache.empty();
}

template<bi::Location CL, class IO1>
void bi::BootstrapPFCache<CL,IO1>::flush() {
  //ancestryCache.flush();
  parent_type::flush();
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::BootstrapPFCache<CL,IO1>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < parent_type > (*this);
  ar & ancestryCache;
  ar & logWeightsCache;
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::BootstrapPFCache<CL,IO1>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < parent_type > (*this);
  ar & ancestryCache;
  ar & logWeightsCache;
}

#endif
