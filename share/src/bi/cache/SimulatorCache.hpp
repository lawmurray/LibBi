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
#include "../null/SimulatorNullBuffer.hpp"

namespace bi {
/**
 * Cache for simulation.
 *
 * @ingroup io_cache
 *
 * @tparam CL Location.
 * @tparam IO1 Buffer type.
 */
template<Location CL = ON_HOST, class IO1 = SimulatorNullBuffer>
class SimulatorCache: public IO1 {
public:
  /**
   * @copydoc SimulatorBuffer::SimulatorBuffer()
   */
  SimulatorCache(const Model& m, const size_t P = 0, const size_t T = 0,
      const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = DEFAULT);

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

public:
  // hack that these two are public... used directly in MCMCBuffer::write()
  /**
   * Time cache.
   */
  Cache1D<real,ON_HOST> timeCache;

  /**
   * Number of times in cache.
   */
  int len;

private:
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
bi::SimulatorCache<CL,IO1>::SimulatorCache(const Model& m, const size_t P,
    const size_t T, const std::string& file, const FileMode mode,
    const SchemaMode schema) :
    IO1(m, P, T, file, mode, schema), len(0) {
  //
}

template<bi::Location CL, class IO1>
bi::SimulatorCache<CL,IO1>::SimulatorCache(const SimulatorCache<CL,IO1>& o) :
    IO1(o), timeCache(o.timeCache), len(o.len) {
  //
}

template<bi::Location CL, class IO1>
bi::SimulatorCache<CL,IO1>::~SimulatorCache() {
  //
}

template<bi::Location CL, class IO1>
bi::SimulatorCache<CL,IO1>& bi::SimulatorCache<CL,IO1>::operator=(
    const SimulatorCache<CL,IO1>& o) {
  timeCache = o.timeCache;
  len = o.len;

  return *this;
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
  IO1::writeTimes(0, timeCache.get(0, len));
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
