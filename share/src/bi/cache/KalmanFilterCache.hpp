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
   * Read predicted mean.
   *
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param[out] mu1 Mean.
   */
  template<class V1>
  void readPredictedMean(const int k, V1 mu1) const;

  /**
   * Write predicted mean.
   *
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param[out] mu1 Mean.
   */
  template<class V1>
  void writePredictedMean(const int k, V1 mu1);

  /**
   * @copydoc KalmanFilterNetCDFBuffer::readPredictedStd()
   */
  template<class M1>
  void readPredictedStd(const int k, M1 U1);

  /**
   * @copydoc KalmanFilterNetCDFBuffer::writePredictedStd()
   */
  template<class M1>
  void writePredictedStd(const int k, const M1 U1);

  /**
   * Read corrected mean.
   *
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param[out] mu2 Mean.
   */
  template<class V1>
  void readCorrectedMean(const int k, V1 mu2) const;

  /**
   * Write corrected mean.
   *
   * @tparam V1 Vector type.
   *
   * @param k Time index.
   * @param[out] mu2 Mean.
   */
  template<class V1>
  void writeCorrectedMean(const int k, V1 mu2);

  /**
   * @copydoc KalmanFilterNetCDFBuffer::readCorrectedStd()
   */
  template<class M1>
  void readCorrectedStd(const int k, M1 U2);

  /**
   * @copydoc KalmanFilterNetCDFBuffer::writeCorrectedStd()
   */
  template<class M1>
  void writeCorrectedStd(const int k, const M1 U2);

  /**
   * @copydoc KalmanFilterNetCDFBuffer::readCross()
   */
  template<class M1>
  void readCross(const int k, M1 C);

  /**
   * @copydoc KalmanFilterNetCDFBuffer::writeCross()
   */
  template<class M1>
  void writeCross(const int k, const M1 C);

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
   * Vector type for caches.
   */
  typedef host_vector<real> vector_type;

  /**
   * Matrix type for caches.
   */
  typedef host_matrix<real> matrix_type;

  /**
   * Predicted mean cache.
   */
  CacheObject<vector_type> mu1Cache;

  /**
   * Cholesky factor of predicted covariance matrix cache.
   */
  CacheObject<matrix_type> U1Cache;

  /**
   * Corrected mean cache.
   */
  CacheObject<vector_type> mu2Cache;

  /**
   * Cholesky factor of corrected covariance matrix cache.
   */
  CacheObject<matrix_type> U2Cache;

  /**
   * Across-time covariance cache.
   */
  CacheObject<matrix_type> CCache;

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
    SimulatorCache<IO1,CL>(o), mu1Cache(o.mu1Cache), U1Cache(o.U1Cache), mu2Cache(
        o.mu2Cache), U2Cache(o.U2Cache), CCache(o.CCache), out(o.out) {
  //
}

template<class IO1, bi::Location CL>
bi::KalmanFilterCache<IO1,CL>& bi::KalmanFilterCache<IO1,CL>::operator=(
    const KalmanFilterCache<IO1,CL>& o) {
  SimulatorCache<IO1,CL>::operator=(o);
  out = o.out;
  mu1Cache = o.mu1Cache;
  U1Cache = o.U1Cache;
  mu2Cache = o.mu2Cache;
  U2Cache = o.U2Cache;
  CCache = o.CCache;

  return *this;
}

template<class IO1, bi::Location CL>
bi::KalmanFilterCache<IO1,CL>::~KalmanFilterCache() {
  flush();
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::KalmanFilterCache<IO1,CL>::readPredictedMean(const int k,
    V1 mu1) const {
  mu1 = mu1Cache.get(k);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::KalmanFilterCache<IO1,CL>::writePredictedMean(const int k,
    const V1 mu1) {
  if (!mu1Cache.isValid(k)) {
    vector_type tmp;
    mu1Cache.set(k, tmp);
    mu1Cache.get(k).resize(mu1.size(), false);
  }
  mu1Cache.set(k, mu1);

  if (out != NULL) {
    out->writePredictedMean(k, mu1);
  }
}

template<class IO1, bi::Location CL>
template<class M1>
void bi::KalmanFilterCache<IO1,CL>::readPredictedStd(const int k, M1 U1) {
  U1 = U1Cache.get(k);
}

template<class IO1, bi::Location CL>
template<class M1>
void bi::KalmanFilterCache<IO1,CL>::writePredictedStd(const int k,
    const M1 U1) {
  if (!U1Cache.isValid(k)) {
    matrix_type tmp;
    U1Cache.set(k, tmp);
    U1Cache.get(k).resize(U1.size1(), U1.size2(), false);
  }
  U1Cache.set(k, U1);

  if (out != NULL) {
    out->writePredictedStd(k, U1);
  }
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::KalmanFilterCache<IO1,CL>::readCorrectedMean(const int k,
    V1 mu2) const {
  mu2 = mu2Cache.get(k);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::KalmanFilterCache<IO1,CL>::writeCorrectedMean(const int k,
    const V1 mu2) {
  if (!mu2Cache.isValid(k)) {
    vector_type tmp;
    mu2Cache.set(k, tmp);
    mu2Cache.get(k).resize(mu2.size(), false);
  }
  mu2Cache.set(k, mu2);

  if (out != NULL) {
    out->writeCorrectedMean(k, mu2);
  }
}

template<class IO1, bi::Location CL>
template<class M1>
void bi::KalmanFilterCache<IO1,CL>::readCorrectedStd(const int k, M1 U2) {
  U2 = U2Cache.get(k);
}

template<class IO1, bi::Location CL>
template<class M1>
void bi::KalmanFilterCache<IO1,CL>::writeCorrectedStd(const int k,
    const M1 U2) {
  if (!U2Cache.isValid(k)) {
    matrix_type tmp;
    U2Cache.set(k, tmp);
    U2Cache.get(k).resize(U2.size1(), U2.size2(), false);
  }
  U2Cache.set(k, U2);

  if (out != NULL) {
    out->writeCorrectedStd(k, U2);
  }
}

template<class IO1, bi::Location CL>
template<class M1>
void bi::KalmanFilterCache<IO1,CL>::readCross(const int k, M1 C) {
  C = CCache.get(k);
}

template<class IO1, bi::Location CL>
template<class M1>
void bi::KalmanFilterCache<IO1,CL>::writeCross(const int k, const M1 C) {
  if (!CCache.isValid(k)) {
    matrix_type tmp;
    CCache.set(k, tmp);
    CCache.get(k).resize(C.size1(), C.size2(), false);
  }
  CCache.set(k, C);

  if (out != NULL) {
    out->writeCross(k, C);
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
  mu1Cache.flush();
  U1Cache.flush();
  mu2Cache.flush();
  U2Cache.flush();
  CCache.flush();
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::KalmanFilterCache<IO1,CL>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < SimulatorCache<IO1,CL> > (*this);
  ar & mu1Cache;
  ar & U1Cache;
  ar & mu2Cache;
  ar & U2Cache;
  ar & CCache;
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::KalmanFilterCache<IO1,CL>::load(Archive& ar,
    const unsigned version) {
  ar & boost::serialization::base_object < SimulatorCache<IO1,CL> > (*this);
  ar & mu1Cache;
  ar & U1Cache;
  ar & mu2Cache;
  ar & U2Cache;
  ar & CCache;
}

#endif
