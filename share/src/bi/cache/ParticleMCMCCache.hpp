/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_PARTICLEMCMCCACHE_HPP
#define BI_CACHE_PARTICLEMCMCCACHE_HPP

#include "Cache1D.hpp"
#include "CacheCross.hpp"
#include "../buffer/ParticleMCMCNetCDFBuffer.hpp"

namespace bi {
/**
 * Cache for ParticleMCMCNetCDFBuffer reads and writes.
 *
 * @ingroup io_cache
 *
 * @tparam IO1 Output type.
 * @tparam CL Location.
 */
template<class IO1 = ParticleMCMCNetCDFBuffer, Location CL = ON_HOST>
class ParticleMCMCCache {
public:
  /**
   * Constructor.
   *
   * @tparam B Model type.
   *
   * @param m Model.
   * @param out Output buffer.
   */
  template<class B>
  ParticleMCMCCache(B& m, IO1* out = NULL);

  /**
   * Shallow copy constructor.
   */
  ParticleMCMCCache(const ParticleMCMCCache<IO1,CL>& o);

  /**
   * Destructor.
   */
  ~ParticleMCMCCache();

  /**
   * Deep assignment operator.
   */
  ParticleMCMCCache<IO1,CL>& operator=(const ParticleMCMCCache<IO1,CL>& o);

  /**
   * @copydoc ParticleMCMCNetCDFBuffer::readTimes()
   */
  template<class V1>
  void readTimes(const int t, V1 x) const;

  /**
   * @copydoc ParticleMCMCNetCDFBuffer::writeTimes()
   */
  template<class V1>
  void writeTimes(const int t, const V1 x);

  /**
   * Read log-likelihood.
   *
   * @param p Sample index.
   *
   * @return Log-likelihood.
   */
  real readLogLikelihood(const int p);

  /**
   * Write log-likelihood.
   *
   * @param p Sample index.
   * @param ll Log-likelihood.
   */
  void writeLogLikelihood(const int p, const real ll);

  /**
   * Read log-prior density.
   *
   * @param p Sample index.
   *
   * @return Log-prior density.
   */
  real readLogPrior(const int p);

  /**
   * Write log-prior density.
   *
   * @param p Sample index.
   * @param lp Log-prior density.
   */
  void writeLogPrior(const int p, const real lp);

  /**
   * Read parameter sample.
   *
   * @tparam V1 Vector type.
   *
   * @param p Sample index.
   * @param[out] theta Sample.
   */
  template<class V1>
  void readParameter(const int p, V1 theta);

  /**
   * Write parameter sample.
   *
   * @tparam V1 Vector type.
   *
   * @param p Sample index.
   * @param theta Sample.
   */
  template<class V1>
  void writeParameter(const int p, const V1 theta);

  /**
   * Read state trajectory sample.
   *
   * @tparam M1 Matrix type.
   *
   * @param p Sample index.
   * @param[out] X Trajectory. Rows index variables, columns index times.
   */
  template<class M1>
  void readTrajectory(const int p, M1 X);

  /**
   * Write state trajectory samples.
   *
   * @tparam M1 Matrix type.
   *
   * @param p Sample index.
   * @param X Trajectories. Rows index variables, columns index times.
   */
  template<class M1>
  void writeTrajectory(const int p, const M1 X);

  /**
   * Is cache full?
   */
  bool isFull() const;

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(ParticleMCMCCache<IO1,CL>& o);

  /**
   * Clear cache.
   */
  void clear();

  /**
   * Empty cache.
   */
  void empty();

  /**
   * Flush to output buffer.
   */
  void flush();

private:
  /**
   * Model.
   */
  Model& m;

  /**
   * Log-likelihoods cache.
   */
  Cache1D<real,CL> llCache;

  /**
   * Log-prior densities cache.
   */
  Cache1D<real,CL> lpCache;

  /**
   * Parameters cache.
   */
  CacheCross<real,CL> parameterCache;

  /**
   * Trajectories cache.
   */
  std::vector<CacheCross<real,CL>*> trajectoryCache;

  /**
   * Id of first sample in cache.
   */
  int first;

  /**
   * Number of samples in cache.
   */
  int len;

  /**
   * Output buffer.
   */
  IO1* out;

  /**
   * Maximum number of samples to store in cache.
   */
  static const int NUM_SAMPLES = 4096 / sizeof(real);

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
 * Factory for creating ParticleMCMCCache objects.
 *
 * @ingroup io_cache
 *
 * @see Forcer
 */
template<Location CL = ON_HOST>
struct ParticleMCMCCacheFactory {
  /**
   * Create ParticleMCMCCache.
   *
   * @return ParticleMCMCCache object. Caller has ownership.
   *
   * @see ParticleMCMCCache::ParticleMCMCCache()
   */
  template<class B, class IO1>
  static ParticleMCMCCache<IO1,CL>* create(B& m, IO1* out = NULL) {
    return new ParticleMCMCCache<IO1,CL>(m, out);
  }

  /**
   * Create ParticleMCMCCache.
   *
   * @return ParticleMCMCCache object. Caller has ownership.
   *
   * @see ParticleMCMCCache::ParticleMCMCCache()
   */
  template<class B>
  static ParticleMCMCCache<ParticleMCMCNetCDFBuffer,CL>* create(B& m) {
    return new ParticleMCMCCache<ParticleMCMCNetCDFBuffer,CL>(m);
  }
};
}

template<class IO1, bi::Location CL>
template<class B>
bi::ParticleMCMCCache<IO1,CL>::ParticleMCMCCache(B& m, IO1* out) : m(m),
    llCache(NUM_SAMPLES), lpCache(NUM_SAMPLES), parameterCache(NUM_SAMPLES,
    m.getNetSize(P_VAR)), first(0), len(0), out(out) {
  //
}

template<class IO1, bi::Location CL>
bi::ParticleMCMCCache<IO1,CL>::ParticleMCMCCache(const ParticleMCMCCache<IO1,CL>& o) :
    m(o.m),
    llCache(o.llCache),
    lpCache(o.lpCache),
    parameterCache(o.parameterCache),
    first(o.first),
    len(o.len),
    out(o.out) {
  trajectoryCache.resize(o.trajectoryCache.size());
  for (int i = 0; i < trajectoryCache.size(); ++i) {
    trajectoryCache[i] = new CacheCross<real,CL>(*o.trajectoryCache[i]);
  }
}

template<class IO1, bi::Location CL>
bi::ParticleMCMCCache<IO1,CL>::~ParticleMCMCCache() {
  flush();
  for (int t = 0; t < int(trajectoryCache.size()); ++t) {
    delete trajectoryCache[t];
  }
}

template<class IO1, bi::Location CL>
bi::ParticleMCMCCache<IO1,CL>& bi::ParticleMCMCCache<IO1,CL>::operator=(
    const ParticleMCMCCache<IO1,CL>& o) {
  m = o.m;

  empty();
  llCache = o.llCache;
  lpCache = o.lpCache;
  parameterCache = o.parameterCache;
  first = o.first;
  len = o.len;
  out = o.out;

  trajectoryCache.resize(o.trajectoryCache.size());
  for (int i = 0; i < trajectoryCache.size(); ++i) {
    trajectoryCache[i] = new CacheCross<real,CL>();
    trajectoryCache[i] = *o.trajectoryCache[i];
  }

  return *this;
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::ParticleMCMCCache<IO1,CL>::readTimes(const int t, V1 x) const {
  /* pre-condition */
  BI_ASSERT(out != NULL);

  out->readTimes(t, x);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::ParticleMCMCCache<IO1,CL>::writeTimes(const int t, const V1 x) {
  if (out != NULL) {
    out->writeTimes(t, x);
  }
}

template<class IO1, bi::Location CL>
real bi::ParticleMCMCCache<IO1,CL>::readLogLikelihood(const int p) {
  /* pre-condition */
  BI_ASSERT(p >= first && p < first + len);

  return llCache.get(p - first);
}

template<class IO1, bi::Location CL>
void bi::ParticleMCMCCache<IO1,CL>::writeLogLikelihood(const int p,
    const real ll) {
  /* pre-condition */
  BI_ASSERT(len == 0 || (p >= first && p <= first + len));

  if (len == 0) {
    first = p;
  }
  if (p - first == len) {
    len = p - first + 1;
  }
  llCache.set(p - first, ll);
}

template<class IO1, bi::Location CL>
real bi::ParticleMCMCCache<IO1,CL>::readLogPrior(const int p) {
  /* pre-condition */
  BI_ASSERT(p >= first && p < first + len);

  return lpCache.get(p - first);
}

template<class IO1, bi::Location CL>
void bi::ParticleMCMCCache<IO1,CL>::writeLogPrior(const int p, const real lp) {
  /* pre-condition */
  BI_ASSERT(len == 0 || (p >= first && p <= first + len));

  if (len == 0) {
    first = p;
  }
  if (p - first == len) {
    len = p - first + 1;
  }
  lpCache.set(p - first, lp);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::ParticleMCMCCache<IO1,CL>::readParameter(const int p, V1 theta) {
  /* pre-condition */
  BI_ASSERT(p >= first && p < first + len);

  theta = parameterCache.get(p - first);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::ParticleMCMCCache<IO1,CL>::writeParameter(const int p,
    const V1 theta) {
  /* pre-condition */
  BI_ASSERT(len == 0 || (p >= first && p <= first + len));

  if (len == 0) {
    first = p;
  }
  if (p - first == len) {
    len = p - first + 1;
  }
  parameterCache.set(p - first, theta);
}

template<class IO1, bi::Location CL>
template<class M1>
void bi::ParticleMCMCCache<IO1,CL>::readTrajectory(const int p, M1 X) {
  /* pre-condition */
  BI_ASSERT(len == 0 || (p >= first && p <= first + len));
  BI_ASSERT(X.size2() == trajectoryCache.size());

  for (int t = 0; t < trajectoryCache.size(); ++t) {
    column(X, t) = trajectoryCache[t]->get(p - first);
  }
}

template<class IO1, bi::Location CL>
template<class M1>
void bi::ParticleMCMCCache<IO1,CL>::writeTrajectory(const int p, const M1 X) {
  /* pre-condition */
  BI_ASSERT(len == 0 || (p >= first && p <= first + len));

  if (len == 0) {
    first = p;
  }
  if (p - first == len) {
    len = p - first + 1;
  }

  if (int(trajectoryCache.size()) < X.size2()) {
    trajectoryCache.resize(X.size2(), NULL);
  }
  for (int t = 0; t < X.size2(); ++t) {
    if (trajectoryCache[t] == NULL) {
      trajectoryCache[t] = new CacheCross<real,CL>(NUM_SAMPLES, m.getDynSize());
    }
    trajectoryCache[t]->set(p - first, column(X, t));
  }
}

template<class IO1, bi::Location CL>
bool bi::ParticleMCMCCache<IO1,CL>::isFull() const {
  return len == NUM_SAMPLES;
}

template<class IO1, bi::Location CL>
void bi::ParticleMCMCCache<IO1,CL>::swap(ParticleMCMCCache<IO1,CL>& o) {
  llCache.swap(o.llCache);
  lpCache.swap(o.lpCache);
  parameterCache.swap(o.parameterCache);
  trajectoryCache.swap(o.trajectoryCache);
  std::swap(first, o.first);
  std::swap(len, o.len);
}

template<class IO1, bi::Location CL>
void bi::ParticleMCMCCache<IO1,CL>::clear() {
  llCache.clear();
  lpCache.clear();
  parameterCache.clear();
  for (int t = 0; t < int(trajectoryCache.size()); ++t) {
    trajectoryCache[t]->clear();
  }
  first = 0;
  len = 0;
}

template<class IO1, bi::Location CL>
void bi::ParticleMCMCCache<IO1,CL>::empty() {
  llCache.empty();
  lpCache.empty();
  parameterCache.empty();
  for (int t = 0; t < trajectoryCache.size(); ++t) {
    trajectoryCache[t]->empty();
    delete trajectoryCache[t];
  }
  trajectoryCache.resize(0);
  first = 0;
  len = 0;
}

template<class IO1, bi::Location CL>
void bi::ParticleMCMCCache<IO1,CL>::flush() {
  if (out != NULL) {
    out->writeLogLikelihoods(first, llCache.get(0, len));
    out->writeLogPriors(first, lpCache.get(0, len));
    out->writeParameters(first, parameterCache.get(0, len));

    llCache.flush();
    lpCache.flush();
    parameterCache.flush();

    for (int t = 0; t < int(trajectoryCache.size()); ++t) {
      out->writeState(t, first, trajectoryCache[t]->get(0, len));
      trajectoryCache[t]->flush();
    }
  }
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::ParticleMCMCCache<IO1,CL>::save(Archive& ar, const unsigned version) const {
  ar & llCache;
  ar & lpCache;
  ar & parameterCache;
  ar & trajectoryCache;
  ar & first;
  ar & len;
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::ParticleMCMCCache<IO1,CL>::load(Archive& ar, const unsigned version) {
  ar & llCache;
  ar & lpCache;
  ar & parameterCache;
  ar & trajectoryCache;
  ar & first;
  ar & len;
}

#endif
