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
  void readTime(const int t, real& x) const;

  /**
   * @copydoc SimulatorNetCDFBuffer::writeTime()
   */
  void writeTime(const int t, const real& x);

  /**
   * @copydoc SimulatorNetCDFBuffer::readTimes()
   */
  template<class V1>
  void readTimes(const int t, V1 x) const;

  /**
   * @copydoc SimulatorNetCDFBuffer::writeTimes()
   */
  template<class V1>
  void writeTimes(const int t, const V1 x);

  /**
   * @copydoc SimulatorNetCDFBuffer::readParameters()
   */
  template<class B, Location L>
  void readParameters(State<B,L>& s) const;

  /**
   * @copydoc SimulatorNetCDFBuffer::writeParameters()
   */
  template<class B, Location L>
  void writeParameters(const State<B,L>& s);

  /**
   * @copydoc SimulatorNetCDFBuffer::readState()
   */
  template<class B, Location L>
  void readState(const int t, State<B,L>& s) const;

  /**
   * @copydoc SimulatorNetCDFBuffer::writeState()
   */
  template<class B, Location L>
  void writeState(const int t, const State<B,L>& s);

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(SimulatorCache<IO1,CL>& o);

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
const typename bi::Cache1D<real,bi::ON_HOST>::vector_reference_type bi::SimulatorCache<
    IO1,CL>::getTimes() const {
  return timeCache.get(0, len);
}

template<class IO1, bi::Location CL>
void bi::SimulatorCache<IO1,CL>::readTime(const int t, real& x) const {
  /* pre-condition */
  BI_ASSERT(t >= 0 && t < len);

  x = timeCache.get(t);
}

template<class IO1, bi::Location CL>
void bi::SimulatorCache<IO1,CL>::writeTime(const int t, const real& x) {
  /* pre-condition */
  BI_ASSERT(t >= 0 && t <= len);

  if (t == len) {
    ++len;
  }
  timeCache.set(t, x);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::SimulatorCache<IO1,CL>::readTimes(const int t, V1 x) const {
  /* pre-condition */
  BI_ASSERT(t >= 0 && t + x.size() <= len);

  x = timeCache.get(t, x.size());
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::SimulatorCache<IO1,CL>::writeTimes(const int t, const V1 x) {
  /* pre-condition */
  BI_ASSERT(t >= 0 && t <= len);

  if (t + x.size() > len) {
    len = t + x.size();
  }
  timeCache.set(t, x.size(), x);
}

template<class IO1, bi::Location CL>
template<class B, bi::Location L>
void bi::SimulatorCache<IO1,CL>::readParameters(State<B,L>& s) const {
  /* pre-conditions */
  BI_ASSERT(out != NULL);

  out->readParameters(s);
}

template<class IO1, bi::Location CL>
template<class B, bi::Location L>
void bi::SimulatorCache<IO1,CL>::writeParameters(const State<B,L>& s) {
  if (out != NULL) {
    out->writeParameters(s);
  }
}

template<class IO1, bi::Location CL>
template<class B, bi::Location L>
void bi::SimulatorCache<IO1,CL>::readState(const int t, State<B,L>& s) const {
  /* pre-conditions */
  BI_ASSERT(out != NULL);

  out->readState(t, s);
}

template<class IO1, bi::Location CL>
template<class B, bi::Location L>
void bi::SimulatorCache<IO1,CL>::writeState(const int t,
    const State<B,L>& s) {
  if (out != NULL) {
    out->writeState(t, s);
  }
}

template<class IO1, bi::Location CL>
void bi::SimulatorCache<IO1,CL>::swap(SimulatorCache<IO1,CL>& o) {
  timeCache.swap(o.timeCache);
  std::swap(len, o.len);
}

template<class IO1, bi::Location CL>
void bi::SimulatorCache<IO1,CL>::clear() {
  timeCache.clear();
  len = 0;
}

template<class IO1, bi::Location CL>
void bi::SimulatorCache<IO1,CL>::empty() {
  timeCache.empty();
  len = 0;
}

template<class IO1, bi::Location CL>
void bi::SimulatorCache<IO1,CL>::flush() {
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
