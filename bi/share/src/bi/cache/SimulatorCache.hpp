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
  const typename Cache1D<real,CL>::vector_reference_type getTimes() const;

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
  Cache1D<real,CL> timeCache;

  /**
   * Output buffer.
   */
  IO1* out;
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
bi::SimulatorCache<IO1,CL>::SimulatorCache(IO1* out) : out(out) {
  //
}

template<class IO1, bi::Location CL>
bi::SimulatorCache<IO1,CL>::SimulatorCache(const SimulatorCache<IO1,CL>& o) :
    timeCache(o.timeCache), out(o.out) {
  //
}

template<class IO1, bi::Location CL>
bi::SimulatorCache<IO1,CL>::~SimulatorCache() {
  flush();
}

template<class IO1, bi::Location CL>
bi::SimulatorCache<IO1,CL>& bi::SimulatorCache<IO1,CL>::operator=(const SimulatorCache<IO1,CL>& o) {
  timeCache = o.timeCache;
  out = o.out;

  return *this;
}

template<class IO1, bi::Location CL>
const typename bi::Cache1D<real,CL>::vector_reference_type bi::SimulatorCache<
    IO1,CL>::getTimes() const {
  return timeCache.get(0, timeCache.size());
}

template<class IO1, bi::Location CL>
void bi::SimulatorCache<IO1,CL>::readTime(const int t, real& x) const {
  x = timeCache.get(t);
}

template<class IO1, bi::Location CL>
void bi::SimulatorCache<IO1,CL>::writeTime(const int t, const real& x) {
  timeCache.set(t, x);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::SimulatorCache<IO1,CL>::readTimes(const int t, V1 x) const {
  x = timeCache.get(t, x.size());
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::SimulatorCache<IO1,CL>::writeTimes(const int t, const V1 x) {
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
void bi::SimulatorCache<IO1,CL>::clear() {
  timeCache.clear();
}

template<class IO1, bi::Location CL>
void bi::SimulatorCache<IO1,CL>::empty() {
  timeCache.empty();
}

template<class IO1, bi::Location CL>
void bi::SimulatorCache<IO1,CL>::flush() {
  if (out != NULL) {
    out->writeTimes(0, timeCache.get(0, timeCache.size()));
  }
  timeCache.flush();
}

#endif
