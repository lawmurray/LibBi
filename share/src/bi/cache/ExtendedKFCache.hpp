/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_EXTENDEDKFCACHE_HPP
#define BI_CACHE_EXTENDEDKFCACHE_HPP

#include "SimulatorCache.hpp"
#include "../null/KalmanFilterNullBuffer.hpp"

namespace bi {
/**
 * Cache for Kalman filter.
 *
 * @ingroup io_cache
 *
 * @tparam CL Location.
 * @tparam IO1 Buffer type.
 */
template<Location CL = ON_HOST, class IO1 = KalmanFilterNullBuffer>
class ExtendedKFCache: public SimulatorCache<CL,IO1> {
public:
  typedef SimulatorCache<CL,IO1> parent_type;

  /**
   * @copydoc KalmanFilterBuffer::KalmanFilterBuffer()
   */
  ExtendedKFCache(const Model& m, const size_t P = 0, const size_t T = 0,
      const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = DEFAULT);

  /**
   * Shallow copy.
   */
  ExtendedKFCache(const ExtendedKFCache<CL,IO1>& o);

  /**
   * Destructor.
   */
  ~ExtendedKFCache();

  /**
   * Deep assignment.
   */
  ExtendedKFCache<CL,IO1>& operator=(const ExtendedKFCache<CL,IO1>& o);

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
   * @copydoc AncestryCache::readPath()
   */
  template<class M1>
  void readPath(const int p, M1 X) const;

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(ExtendedKFCache<CL,IO1>& o);

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
bi::ExtendedKFCache<CL,IO1>::ExtendedKFCache(const Model& m, const size_t P,
    const size_t T, const std::string& file, const FileMode mode,
    const SchemaMode schema) :
    parent_type(m, P, T, file, mode, schema) {
  //
}

template<bi::Location CL, class IO1>
bi::ExtendedKFCache<CL,IO1>::ExtendedKFCache(const ExtendedKFCache<CL,IO1>& o) :
    parent_type(o), mu1Cache(o.mu1Cache), U1Cache(o.U1Cache), mu2Cache(
        o.mu2Cache), U2Cache(o.U2Cache), CCache(o.CCache) {
  //
}

template<bi::Location CL, class IO1>
bi::ExtendedKFCache<CL,IO1>& bi::ExtendedKFCache<CL,IO1>::operator=(
    const ExtendedKFCache<CL,IO1>& o) {
  parent_type::operator=(o);
  mu1Cache = o.mu1Cache;
  U1Cache = o.U1Cache;
  mu2Cache = o.mu2Cache;
  U2Cache = o.U2Cache;
  CCache = o.CCache;

  return *this;
}

template<bi::Location CL, class IO1>
bi::ExtendedKFCache<CL,IO1>::~ExtendedKFCache() {
  //
}

template<bi::Location CL, class IO1>
template<class V1>
void bi::ExtendedKFCache<CL,IO1>::readPredictedMean(const int k,
    V1 mu1) const {
  mu1 = mu1Cache.get(k);
}

template<bi::Location CL, class IO1>
template<class V1>
void bi::ExtendedKFCache<CL,IO1>::writePredictedMean(const int k,
    const V1 mu1) {
  if (!mu1Cache.isValid(k)) {
    vector_type tmp;
    mu1Cache.set(k, tmp);
    mu1Cache.get(k).resize(mu1.size(), false);
  }
  mu1Cache.set(k, mu1);
  parent_type::writePredictedMean(k, mu1);
}

template<bi::Location CL, class IO1>
template<class M1>
void bi::ExtendedKFCache<CL,IO1>::readPredictedStd(const int k, M1 U1) {
  U1 = U1Cache.get(k);
}

template<bi::Location CL, class IO1>
template<class M1>
void bi::ExtendedKFCache<CL,IO1>::writePredictedStd(const int k,
    const M1 U1) {
  if (!U1Cache.isValid(k)) {
    matrix_type tmp;
    U1Cache.set(k, tmp);
    U1Cache.get(k).resize(U1.size1(), U1.size2(), false);
  }
  U1Cache.set(k, U1);
  parent_type::writePredictedStd(k, U1);
}

template<bi::Location CL, class IO1>
template<class V1>
void bi::ExtendedKFCache<CL,IO1>::readCorrectedMean(const int k,
    V1 mu2) const {
  mu2 = mu2Cache.get(k);
}

template<bi::Location CL, class IO1>
template<class V1>
void bi::ExtendedKFCache<CL,IO1>::writeCorrectedMean(const int k,
    const V1 mu2) {
  if (!mu2Cache.isValid(k)) {
    vector_type tmp;
    mu2Cache.set(k, tmp);
    mu2Cache.get(k).resize(mu2.size(), false);
  }
  mu2Cache.set(k, mu2);
  parent_type::writeCorrectedMean(k, mu2);
}

template<bi::Location CL, class IO1>
template<class M1>
void bi::ExtendedKFCache<CL,IO1>::readCorrectedStd(const int k, M1 U2) {
  U2 = U2Cache.get(k);
}

template<bi::Location CL, class IO1>
template<class M1>
void bi::ExtendedKFCache<CL,IO1>::writeCorrectedStd(const int k,
    const M1 U2) {
  if (!U2Cache.isValid(k)) {
    matrix_type tmp;
    U2Cache.set(k, tmp);
    U2Cache.get(k).resize(U2.size1(), U2.size2(), false);
  }
  U2Cache.set(k, U2);
  parent_type::writeCorrectedStd(k, U2);
}

template<bi::Location CL, class IO1>
template<class M1>
void bi::ExtendedKFCache<CL,IO1>::readCross(const int k, M1 C) {
  C = CCache.get(k);
}

template<bi::Location CL, class IO1>
template<class M1>
void bi::ExtendedKFCache<CL,IO1>::writeCross(const int k, const M1 C) {
  if (!CCache.isValid(k)) {
    matrix_type tmp;
    CCache.set(k, tmp);
    CCache.get(k).resize(C.size1(), C.size2(), false);
  }
  CCache.set(k, C);
  parent_type::writeCross(k, C);
}

template<bi::Location CL, class IO1>
template<class M1>
void bi::ExtendedKFCache<CL,IO1>::readPath(const int p, M1 X) const {
  //
}

template<bi::Location CL, class IO1>
void bi::ExtendedKFCache<CL,IO1>::swap(ExtendedKFCache<CL,IO1>& o) {
  parent_type::swap(o);
  mu1Cache.swap(o.mu1Cache);
  U1Cache.swap(o.U1Cache);
  mu2Cache.swap(o.mu2Cache);
  U2Cache.swap(o.U2Cache);
  CCache.swap(o.CCache);
}

template<bi::Location CL, class IO1>
void bi::ExtendedKFCache<CL,IO1>::clear() {
  parent_type::clear();
  mu1Cache.clear();
  U1Cache.clear();
  mu2Cache.clear();
  U2Cache.clear();
  CCache.clear();
}

template<bi::Location CL, class IO1>
void bi::ExtendedKFCache<CL,IO1>::empty() {
  parent_type::empty();
  mu1Cache.empty();
  U1Cache.empty();
  mu2Cache.empty();
  U2Cache.empty();
  CCache.empty();
}

template<bi::Location CL, class IO1>
void bi::ExtendedKFCache<CL,IO1>::flush() {
  parent_type::flush();
  mu1Cache.flush();
  U1Cache.flush();
  mu2Cache.flush();
  U2Cache.flush();
  CCache.flush();
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::ExtendedKFCache<CL,IO1>::save(Archive& ar,
    const unsigned version) const {
  ar & boost::serialization::base_object < parent_type > (*this);
  ar & mu1Cache;
  ar & U1Cache;
  ar & mu2Cache;
  ar & U2Cache;
  ar & CCache;
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::ExtendedKFCache<CL,IO1>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < parent_type > (*this);
  ar & mu1Cache;
  ar & U1Cache;
  ar & mu2Cache;
  ar & U2Cache;
  ar & CCache;
}

#endif
