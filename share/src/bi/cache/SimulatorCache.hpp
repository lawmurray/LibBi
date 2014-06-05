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
 * @tparam CL Location.
 * @tparam IO1 Buffer type.
 */
template<Location CL = ON_HOST, class IO1 = SimulatorNetCDFBuffer>
class SimulatorCache {
public:
  /**
   * Constructor.
   *
   * @param out Output buffer.
   */
  SimulatorCache(IO1& out = NULL);

  /**
   * Shallow copy constructor.
   */
  SimulatorCache(const SimulatorCache<CL,IO1>& o);

  /**
   * Destructor.
   */
  ~SimulatorCache();

  /**
   * Deep assignment operator.
   */
  SimulatorCache<CL,IO1>& operator=(const SimulatorCache<CL,IO1>& o);

  /**
   * @copydoc OutputBuffer::write()
   */
  template<class S1>
  void write(const size_t k, const real t, const S1& s);

  /**
   * @copydoc OutputBuffer::write0()
   */
  template<class S1>
  void write0(const S1& s);

protected:
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
  void swap(SimulatorCache<CL,IO1>& o);

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
bi::SimulatorCache<CL,IO1>::SimulatorCache(IO1& out) :
    len(0), out(out) {
  //
}

template<bi::Location CL, class IO1>
bi::SimulatorCache<CL,IO1>::SimulatorCache(const SimulatorCache<CL,IO1>& o) :
    timeCache(o.timeCache), len(o.len), out(o.out) {
  //
}

template<bi::Location CL, class IO1>
bi::SimulatorCache<CL,IO1>::~SimulatorCache() {
  flush();
}

template<bi::Location CL, class IO1>
bi::SimulatorCache<CL,IO1>& bi::SimulatorCache<CL,IO1>::operator=(
    const SimulatorCache<CL,IO1>& o) {
  timeCache = o.timeCache;
  len = o.len;
  out = o.out;

  return *this;
}

template<bi::Location CL, class IO1>
template<class S1>
void bi::SimulatorCache<CL,IO1>::write(const size_t k, const real t,
    const S1& s) {
  writeTime(k, t);
  writeState(k, s.getDyn());
}

template<bi::Location CL, class IO1>
template<class S1>
void bi::SimulatorCache<CL,IO1>::write0(const S1& s) {
  writeParameters(s.get(P_VAR));
}

template<bi::Location CL, class IO1>
inline const typename bi::Cache1D<real,bi::ON_HOST>::vector_reference_type bi::SimulatorCache<
    CL,IO1>::getTimes() const {
  return timeCache.get(0, len);
}

template<bi::Location CL, class IO1>
inline void bi::SimulatorCache<CL,IO1>::readTime(const int k, real& t) const {
  /* pre-condition */
  BI_ASSERT(k >= 0 && k < len);

  t = timeCache.get(k);
}

template<bi::Location CL, class IO1>
inline void bi::SimulatorCache<CL,IO1>::writeTime(const int k,
    const real& t) {
  /* pre-condition */
  BI_ASSERT(k >= 0 && k <= len);

  if (k == len) {
    ++len;
  }
  timeCache.set(k, t);
}

template<bi::Location CL, class IO1>
template<class V1>
inline void bi::SimulatorCache<CL,IO1>::readTimes(const int k, V1 ts) const {
  /* pre-condition */
  BI_ASSERT(k >= 0 && k + ts.size() <= len);

  ts = timeCache.get(k, ts.size());
}

template<bi::Location CL, class IO1>
template<class V1>
inline void bi::SimulatorCache<CL,IO1>::writeTimes(const int k, const V1 ts) {
  /* pre-condition */
  BI_ASSERT(k >= 0 && k <= len);

  if (k + ts.size() > len) {
    len = k + ts.size();
  }
  timeCache.set(k, ts.size(), ts);
}

template<bi::Location CL, class IO1>
template<class M1>
inline void bi::SimulatorCache<CL,IO1>::readParameters(M1 X) const {
  /* pre-conditions */
  BI_ASSERT(!(equals<IO1,OutputBuffer>::value));

  out.readParameters(X);
}

template<bi::Location CL, class IO1>
template<class M1>
inline void bi::SimulatorCache<CL,IO1>::writeParameters(const M1 X) {
  out.writeParameters(X);
}

template<bi::Location CL, class IO1>
template<class M1>
inline void bi::SimulatorCache<CL,IO1>::readState(const int k, M1 X) const {
  /* pre-conditions */
  BI_ASSERT(!(equals<IO1,OutputBuffer>::value));

  out.readState(k, X);
}

template<bi::Location CL, class IO1>
template<class M1>
inline void bi::SimulatorCache<CL,IO1>::writeState(const int k, const M1 X) {
  out.writeState(k, X);
}

template<bi::Location CL, class IO1>
inline void bi::SimulatorCache<CL,IO1>::swap(SimulatorCache<CL,IO1>& o) {
  timeCache.swap(o.timeCache);
  std::swap(len, o.len);
}

template<bi::Location CL, class IO1>
inline int bi::SimulatorCache<CL,IO1>::size() const {
  return len;
}

template<bi::Location CL, class IO1>
inline void bi::SimulatorCache<CL,IO1>::clear() {
  timeCache.clear();
  len = 0;
}

template<bi::Location CL, class IO1>
inline void bi::SimulatorCache<CL,IO1>::empty() {
  timeCache.empty();
  len = 0;
}

template<bi::Location CL, class IO1>
inline void bi::SimulatorCache<CL,IO1>::flush() {
  out.writeTimes(0, getTimes());
  timeCache.flush();
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::SimulatorCache<CL,IO1>::save(Archive& ar,
    const unsigned version) const {
  ar & timeCache;
  ar & len;
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::SimulatorCache<CL,IO1>::load(Archive& ar, const unsigned version) {
  ar & timeCache;
  ar & len;
}

#endif
