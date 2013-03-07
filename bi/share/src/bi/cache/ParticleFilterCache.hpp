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
   * Vector type.
   */
  typedef typename temp_host_vector<real>::type vector_type;

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
  const vector_type getLogWeights() const;

  /**
   * @copydoc ParticleFilterNetCDFBuffer::readLogWeights()
   */
  template<class V1>
  void readLogWeights(const int t, V1 lws) const;

  /**
   * @copydoc ParticleFilterNetCDFBuffer::writeLogWeights()
   */
  template<class V1>
  void writeLogWeights(const int t, const V1 lws);

  /**
   * @copydoc ParticleFilterNetCDFBuffer::readAncestors()
   */
  template<class V1>
  void readAncestors(const int t, V1 a) const;

  /**
   * @copydoc ParticleFilterNetCDFBuffer::writeAncestors()
   */
  template<class V1>
  void writeAncestors(const int t, const V1 a);

  /**
   * @copydoc ParticleFilterNetCDFBuffer::readResample()
   */
  int readResample(const int t) const;

  /**
   * @copydoc ParticleFilterNetCDFBuffer::writeResample()
   */
  void writeResample(const int t, const int& r);

  /**
   * @copydoc ParticleFilterNetCDFBuffer::readResamples()
   */
  template<class V1>
  void readResamples(const int t, V1 r) const;

  /**
   * @copydoc ParticleFilterNetCDFBuffer::writeResamples()
   */
  template<class V1>
  void writeResamples(const int t, const V1 r);

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
  AncestryCache ancestryCache;

  /**
   * Resampling cache.
   */
  Cache1D<real,CL> resampleCache;

  /**
   * Most recent log-weights.
   */
  vector_type logWeightsCache;

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
    SimulatorCache<IO1,CL>(o),
    ancestryCache(o.ancestryCache),
    resampleCache(o.resampleCache),
    logWeightsCache(o.logWeightsCache),
    out(o.out) {
  //
}

template<class IO1, bi::Location CL>
bi::ParticleFilterCache<IO1,CL>& bi::ParticleFilterCache<IO1,CL>::operator=(
    const ParticleFilterCache<IO1,CL>& o) {
  SimulatorCache<IO1,CL>::operator=(o);

  logWeightsCache.resize(o.logWeightsCache.size(), false);

  ancestryCache = o.ancestryCache;
  resampleCache = o.resampleCache;
  logWeightsCache = o.logWeightsCache;
  out = o.out;

  return *this;
}

template<class IO1, bi::Location CL>
bi::ParticleFilterCache<IO1,CL>::~ParticleFilterCache() {
  flush();
}

template<class IO1, bi::Location CL>
const typename bi::ParticleFilterCache<IO1,CL>::vector_type bi::ParticleFilterCache<
    IO1,CL>::getLogWeights() const {
  return logWeightsCache;
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::ParticleFilterCache<IO1,CL>::readLogWeights(const int t,
    V1 lws) const {
  BI_ASSERT(out != NULL);

  out->readLogWeights(t, lws);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::ParticleFilterCache<IO1,CL>::writeLogWeights(const int t,
    const V1 lws) {
  if (out != NULL) {
    out->writeLogWeights(t, lws);
  }
  logWeightsCache.resize(lws.size(), false);
  logWeightsCache = lws;
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::ParticleFilterCache<IO1,CL>::readAncestors(const int t,
    V1 as) const {
  BI_ASSERT(out != NULL);

  out->readAncestors(t, as);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::ParticleFilterCache<IO1,CL>::writeAncestors(const int t,
    const V1 as) {
  if (out != NULL) {
    out->writeAncestors(t, as);
  }
}

template<class IO1, bi::Location CL>
int bi::ParticleFilterCache<IO1,CL>::readResample(const int t) const {
  return resampleCache.get(t);
}

template<class IO1, bi::Location CL>
void bi::ParticleFilterCache<IO1,CL>::writeResample(const int t,
    const int& x) {
  resampleCache.set(t, x);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::ParticleFilterCache<IO1,CL>::readResamples(const int t, V1 x) const {
  x = resampleCache.get(t, x.size());
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::ParticleFilterCache<IO1,CL>::writeResamples(const int t,
    const V1 x) {
  resampleCache.set(t, x.size(), x);
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
template<class B, bi::Location L, class V1>
void bi::ParticleFilterCache<IO1,CL>::writeState(const int t,
    const State<B,L>& s, const V1 as) {
  SimulatorCache<IO1,CL>::writeState(t, s);
  ancestryCache.writeState(t, s, as);
}

template<class IO1, bi::Location CL>
void bi::ParticleFilterCache<IO1,CL>::swap(ParticleFilterCache<IO1,CL>& o) {
  SimulatorCache<IO1,CL>::swap(o);
  ancestryCache.swap(o.ancestryCache);
  resampleCache.swap(o.resampleCache);
  logWeightsCache.swap(o.logWeightsCache);
}

template<class IO1, bi::Location CL>
void bi::ParticleFilterCache<IO1,CL>::clear() {
  SimulatorCache<IO1,CL>::clear();
  ancestryCache.clear();
  resampleCache.clear();
}

template<class IO1, bi::Location CL>
void bi::ParticleFilterCache<IO1,CL>::empty() {
  SimulatorCache<IO1,CL>::empty();
  ancestryCache.empty();
  resampleCache.empty();
}

template<class IO1, bi::Location CL>
void bi::ParticleFilterCache<IO1,CL>::flush() {
  if (out != NULL) {
    out->writeResamples(0, resampleCache.get(0, resampleCache.size()));
  }
  //ancestryCache.flush();
  resampleCache.flush();
  SimulatorCache<IO1,CL>::flush();
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::ParticleFilterCache<IO1,CL>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object<SimulatorCache<IO1,CL> >(*this);
  ar & ancestryCache;
  ar & resampleCache;
  save_resizable_vector(ar, version, logWeightsCache);
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::ParticleFilterCache<IO1,CL>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object<SimulatorCache<IO1,CL> >(*this);
  ar & ancestryCache;
  ar & resampleCache;
  load_resizable_vector(ar, version, logWeightsCache);
}

#endif
