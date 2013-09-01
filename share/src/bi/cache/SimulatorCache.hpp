/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_SIMULATORCACHE_HPP
#define BI_CACHE_SIMULATORCACHE_HPP

#include "Cache1D.hpp"
#include "../buffer/SimulatorNetCDFBuffer.hpp"

namespace bi {
/**
 * Cache for SimulatorNetCDFBuffer reads and writes. This caches reads and
 * writes of times. Reads and writes of variables are assumed to be large
 * and contiguous, such that NetCDF/HDF5's own buffering mechanisms, or even
 * direct reads/write on disk, are efficient enough.
 *
 * @ingroup io_cache
 *
 * @tparam IO1 Buffer type.
 * @tparam CL Location.
 */
template<class IO1 = SimulatorNetCDFBuffer, Location CL = ON_HOST>
class SimulatorCache {
public:
  /**
   * Constructor.
   *
   * @param out Output buffer.
   */
  SimulatorCache(IO1* out = NULL);

  /**
   * Shallow copy constructor.
   */
  SimulatorCache(const SimulatorCache<IO1,CL>& o);

  /**
   * Destructor.
   */
  ~SimulatorCache();

  /**
   * Deep assignment operator.
   */
  SimulatorCache<IO1,CL>& operator=(const SimulatorCache<IO1,CL>& o);

  /**
   * Get the vector of all times.
   *
   * @return The vector of all times.
   */
  const typename Cache1D<real,ON_HOST>::vector_reference_type getTimes() const;

  /**
   * @copydoc SimulatorNetCDFBuffer::readTime()
   */
  void readTime(const int k, real& t) const;

  /**
   * @copydoc SimulatorNetCDFBuffer::writeTime()
   */
  void writeTime(const int k, const real& t);

  /**
   * @copydoc SimulatorNetCDFBuffer::readTimes()
   */
  template<class V1>
  void readTimes(const int k, V1 ts) const;

  /**
   * @copydoc SimulatorNetCDFBuffer::writeTimes()
   */
  template<class V1>
  void writeTimes(const int k, const V1 ts);

  /**
   * @copydoc SimulatorNetCDFBuffer::readParameters()
   */
  template<class M1>
  void readParameters(M1 X) const;

  /**
   * @copydoc SimulatorNetCDFBuffer::writeParameters()
   */
  template<class M1>
  void writeParameters(const M1 X);

  /**
   * @copydoc SimulatorNetCDFBuffer::readState()
   */
  template<class M1>
  void readState(const int k, M1 X) const;

  /**
   * @copydoc SimulatorNetCDFBuffer::writeState()
   */
  template<class M1>
  void writeState(const int k, const M1 X);

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(SimulatorCache<IO1,CL>& o);

  /**
   * Size of the cache.
   */
  int size() const;

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
   * Time cache.
   */
  Cache1D<real,ON_HOST> timeCache;

  /**
   * Number of times in cache.
   */
  int len;

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
 * Factory for creating SimulatorCache objects.
 *
 * @ingroup io_cache
 *
 * @see Forcer
 */
template<Location CL = ON_HOST>
struct SimulatorCacheFactory {
  /**
   * Create SimulatorCache.
   *
   * @return SimulatorCache object. Caller has ownership.
   *
   * @see SimulatorCache::SimulatorCache()
   */
  template<class IO1>
  static SimulatorCache<IO1,CL>* create(IO1* out) {
    return new SimulatorCache<IO1,CL>(out);
  }

  /**
   * Create SimulatorCache.
   *
   * @return SimulatorCache object. Caller has ownership.
   *
   * @see SimulatorCache::SimulatorCache()
   */
  static SimulatorCache<SimulatorNetCDFBuffer,CL>* create() {
    return new SimulatorCache<SimulatorNetCDFBuffer,CL>();
  }
};
}

template<class IO1, bi::Location CL>
bi::SimulatorCache<IO1,CL>::SimulatorCache(IO1* out) :
    len(0), out(out) {
  //
}

template<class IO1, bi::Location CL>
bi::SimulatorCache<IO1,CL>::SimulatorCache(const SimulatorCache<IO1,CL>& o) :
    timeCache(o.timeCache), len(o.len), out(o.out) {
  //
}

template<class IO1, bi::Location CL>
bi::SimulatorCache<IO1,CL>::~SimulatorCache() {
  flush();
}

template<class IO1, bi::Location CL>
bi::SimulatorCache<IO1,CL>& bi::SimulatorCache<IO1,CL>::operator=(
    const SimulatorCache<IO1,CL>& o) {
  timeCache = o.timeCache;
  len = o.len;
  out = o.out;

  return *this;
}

template<class IO1, bi::Location CL>
inline const typename bi::Cache1D<real,bi::ON_HOST>::vector_reference_type bi::SimulatorCache<
    IO1,CL>::getTimes() const {
  return timeCache.get(0, len);
}

template<class IO1, bi::Location CL>
inline void bi::SimulatorCache<IO1,CL>::readTime(const int k, real& t) const {
  /* pre-condition */
  BI_ASSERT(k >= 0 && k < len);

  t = timeCache.get(k);
}

template<class IO1, bi::Location CL>
inline void bi::SimulatorCache<IO1,CL>::writeTime(const int k, const real& t) {
  /* pre-condition */
  BI_ASSERT(k >= 0 && k <= len);

  if (k == len) {
    ++len;
  }
  timeCache.set(k, t);
}

template<class IO1, bi::Location CL>
template<class V1>
inline void bi::SimulatorCache<IO1,CL>::readTimes(const int k, V1 ts) const {
  /* pre-condition */
  BI_ASSERT(k >= 0 && k + ts.size() <= len);

  ts = timeCache.get(k, ts.size());
}

template<class IO1, bi::Location CL>
template<class V1>
inline void bi::SimulatorCache<IO1,CL>::writeTimes(const int k, const V1 ts) {
  /* pre-condition */
  BI_ASSERT(k >= 0 && k <= len);

  if (k + ts.size() > len) {
    len = k + ts.size();
  }
  timeCache.set(k, ts.size(), ts);
}

template<class IO1, bi::Location CL>
template<class M1>
inline void bi::SimulatorCache<IO1,CL>::readParameters(M1 X) const {
  /* pre-conditions */
  BI_ASSERT(out != NULL);

  out->readParameters(X);
}

template<class IO1, bi::Location CL>
template<class M1>
inline void bi::SimulatorCache<IO1,CL>::writeParameters(const M1 X) {
  if (out != NULL) {
    out->writeParameters(X);
  }
}

template<class IO1, bi::Location CL>
template<class M1>
inline void bi::SimulatorCache<IO1,CL>::readState(const int t, M1 X) const {
  /* pre-conditions */
  BI_ASSERT(out != NULL);

  out->readState(t, X);
}

template<class IO1, bi::Location CL>
template<class M1>
inline void bi::SimulatorCache<IO1,CL>::writeState(const int t,
    const M1 X) {
  if (out != NULL) {
    out->writeState(t, X);
  }
}

template<class IO1, bi::Location CL>
inline void bi::SimulatorCache<IO1,CL>::swap(SimulatorCache<IO1,CL>& o) {
  timeCache.swap(o.timeCache);
  std::swap(len, o.len);
}

template<class IO1, bi::Location CL>
inline int bi::SimulatorCache<IO1,CL>::size() const {
  return len;
}

template<class IO1, bi::Location CL>
inline void bi::SimulatorCache<IO1,CL>::clear() {
  timeCache.clear();
  len = 0;
}

template<class IO1, bi::Location CL>
inline void bi::SimulatorCache<IO1,CL>::empty() {
  timeCache.empty();
  len = 0;
}

template<class IO1, bi::Location CL>
inline void bi::SimulatorCache<IO1,CL>::flush() {
  if (out != NULL) {
    out->writeTimes(0, getTimes());
  }
  timeCache.flush();
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::SimulatorCache<IO1,CL>::save(Archive& ar,
    const unsigned version) const {
  ar & timeCache;
  ar & len;
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::SimulatorCache<IO1,CL>::load(Archive& ar, const unsigned version) {
  ar & timeCache;
  ar & len;
}

#endif
