/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_KALMANFILTERCACHE_HPP
#define BI_CACHE_KALMANFILTERCACHE_HPP

#include "SimulatorCache.hpp"
#include "../buffer/KalmanFilterNetCDFBuffer.hpp"

namespace bi {
/**
 * Cache for KalmanFilterNetCDFBuffer reads and writes.
 *
 * @ingroup io_cache
 *
 * @tparam IO1 Buffer type.
 * @tparam CL Location.
 */
template<class IO1 = KalmanFilterNetCDFBuffer, Location CL = ON_HOST>
class KalmanFilterCache: public SimulatorCache<IO1,CL> {
public:
  /**
   * Vector type.
   */
  typedef typename temp_host_vector<real>::type vector_type;

  /**
   * Constructor.
   *
   * @param out output buffer.
   */
  KalmanFilterCache(IO1* out = NULL);

  /**
   * Shallow copy.
   */
  KalmanFilterCache(const KalmanFilterCache<IO1,CL>& o);

  /**
   * Destructor.
   */
  ~KalmanFilterCache();

  /**
   * Deep assignment.
   */
  KalmanFilterCache<IO1,CL>& operator=(const KalmanFilterCache<IO1,CL>& o);

  /**
   * @copydoc KalmanFilterNetCDFBuffer::readStd()
   */
  template<class M1>
  void readStd(const int t, M1 S);

  /**
   * @copydoc KalmanFilterNetCDFBuffer::writeStd()
   */
  template<class M1>
  void writeStd(const int t, const M1 S);

  /**
   * @copydoc KalmanFilterNetCDFBuffer::writeLL()
   */
  void writeLL(const real ll);

  /**
   * @copydoc AncestryCache::readTrajectory()
   */
  template<class M1>
  void readTrajectory(const int p, M1 X) const;

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(KalmanFilterCache<IO1,CL>& o);

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
 * Factory for creating KalmanFilterCache objects.
 *
 * @ingroup io_cache
 *
 * @see Forcer
 */
template<Location CL = ON_HOST>
struct KalmanFilterCacheFactory {
  /**
   * Create KalmanFilterCache.
   *
   * @return KalmanFilterCache object. Caller has ownership.
   *
   * @see KalmanFilterCache::KalmanFilterCache()
   */
  template<class IO1>
  static KalmanFilterCache<IO1,CL>* create(IO1* out = NULL) {
    return new KalmanFilterCache<IO1,CL>(out);
  }

  /**
   * Create KalmanFilterCache.
   *
   * @return KalmanFilterCache object. Caller has ownership.
   *
   * @see KalmanFilterCache::KalmanFilterCache()
   */
  static KalmanFilterCache<KalmanFilterNetCDFBuffer,CL>* create() {
    return new KalmanFilterCache<KalmanFilterNetCDFBuffer,CL>();
  }
};
}

template<class IO1, bi::Location CL>
bi::KalmanFilterCache<IO1,CL>::KalmanFilterCache(IO1* out) :
    SimulatorCache<IO1,CL>(out), out(out) {
  //
}

template<class IO1, bi::Location CL>
bi::KalmanFilterCache<IO1,CL>::KalmanFilterCache(
    const KalmanFilterCache<IO1,CL>& o) :
    SimulatorCache<IO1,CL>(o), out(o.out) {
  //
}

template<class IO1, bi::Location CL>
bi::KalmanFilterCache<IO1,CL>& bi::KalmanFilterCache<IO1,CL>::operator=(
    const KalmanFilterCache<IO1,CL>& o) {
  SimulatorCache<IO1,CL>::operator=(o);
  out = o.out;

  return *this;
}

template<class IO1, bi::Location CL>
bi::KalmanFilterCache<IO1,CL>::~KalmanFilterCache() {
  flush();
}

template<class IO1, bi::Location CL>
template<class M1>
void bi::KalmanFilterCache<IO1,CL>::readStd(const int k, M1 S) {
  if (out != NULL) {
    out->readStd(k, S);
  }
}

template<class IO1, bi::Location CL>
template<class M1>
void bi::KalmanFilterCache<IO1,CL>::writeStd(const int k, const M1 S) {
  if (out != NULL) {
    out->writeStd(k, S);
  }
}

template<class IO1, bi::Location CL>
inline void bi::KalmanFilterCache<IO1,CL>::writeLL(const double ll) {
  if (out != NULL) {
    out->writeLL(ll);
  }
}

template<class IO1, bi::Location CL>
template<class M1>
void bi::KalmanFilterCache<IO1,CL>::readTrajectory(const int p, M1 X) const {
  //
}

template<class IO1, bi::Location CL>
void bi::KalmanFilterCache<IO1,CL>::swap(KalmanFilterCache<IO1,CL>& o) {
  SimulatorCache<IO1,CL>::swap(o);
}

template<class IO1, bi::Location CL>
void bi::KalmanFilterCache<IO1,CL>::clear() {
  SimulatorCache<IO1,CL>::clear();
}

template<class IO1, bi::Location CL>
void bi::KalmanFilterCache<IO1,CL>::empty() {
  SimulatorCache<IO1,CL>::empty();
}

template<class IO1, bi::Location CL>
void bi::KalmanFilterCache<IO1,CL>::flush() {
  SimulatorCache<IO1,CL>::flush();
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::KalmanFilterCache<IO1,CL>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object<SimulatorCache<IO1,CL> >(*this);

  //
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::KalmanFilterCache<IO1,CL>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object<SimulatorCache<IO1,CL> >(*this);

  //
}

#endif
