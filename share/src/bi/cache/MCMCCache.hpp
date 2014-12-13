/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_MCMCCACHE_HPP
#define BI_CACHE_MCMCCACHE_HPP

#include "SimulatorCache.hpp"
#include "Cache1D.hpp"
#include "CacheCross.hpp"
#include "../model/Model.hpp"
#include "../null/MCMCNullBuffer.hpp"

namespace bi {
/**
 * Cache for MCMC.
 *
 * @ingroup io_cache
 *
 * @tparam IO1 Output type.
 * @tparam CL Location.
 */
template<Location CL = ON_HOST, class IO1 = MCMCNullBuffer>
class MCMCCache: public SimulatorCache<CL,IO1> {
public:
  typedef SimulatorCache<CL,IO1> parent_type;

  /**
   * @copydoc MCMCBuffer::MCMCBuffer()
   */
  MCMCCache(const Model& m, const size_t P = 0, const size_t T = 0,
      const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = DEFAULT);

  /**
   * Shallow copy constructor.
   */
  MCMCCache(const MCMCCache<CL,IO1>& o);

  /**
   * Destructor.
   */
  ~MCMCCache();

  /**
   * Deep assignment operator.
   */
  MCMCCache<CL,IO1>& operator=(const MCMCCache<CL,IO1>& o);

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
   * Read state path sample.
   *
   * @tparam M1 Matrix type.
   *
   * @param p Sample index.
   * @param[out] X Path. Rows index variables, columns index times.
   */
  template<class M1>
  void readPath(const int p, M1 X);

  /**
   * Write state path sample.
   *
   * @tparam M1 Matrix type.
   *
   * @param p Sample index.
   * @param X Trajectories. Rows index variables, columns index times.
   */
  template<class M1>
  void writePath(const int p, const M1 X);

  /**
   * Is cache full?
   */
  bool isFull() const;

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(MCMCCache<CL,IO1>& o);

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

protected:
  /**
   * Flush state trajectories to disk.
   *
   * @param type Variable type.
   */
  void flushPaths(const VarType type);

  /**
   * Model.
   */
  const Model& m;

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
  std::vector<CacheCross<real,CL>*> pathCache;

  /**
   * Id of first sample in cache.
   */
  int first;

  /**
   * Number of samples in cache.
   */
  int len;

  /**
   * Maximum number of samples to store in cache.
   */
  static const int NUM_SAMPLES = ((CL == ON_HOST) ? 16384 : 4096)
      / sizeof(real);

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
bi::MCMCCache<CL,IO1>::MCMCCache(const Model& m, const size_t P,
    const size_t T, const std::string& file, const FileMode mode,
    const SchemaMode schema) :
    parent_type(m, P, T, file, mode, schema), m(m), llCache(NUM_SAMPLES), lpCache(
        NUM_SAMPLES), parameterCache(NUM_SAMPLES, m.getNetSize(P_VAR)), first(
        0), len(0) {
  const int N = m.getNetSize(R_VAR) + m.getNetSize(D_VAR);
  pathCache.resize(T);
  for (int i = 0; i < pathCache.size(); ++i) {
    pathCache[i] = new CacheCross<real,CL>(NUM_SAMPLES, N);
  }
}

template<bi::Location CL, class IO1>
bi::MCMCCache<CL,IO1>::MCMCCache(const MCMCCache<CL,IO1>& o) :
    parent_type(o), m(o.m), llCache(o.llCache), lpCache(o.lpCache), parameterCache(
        o.parameterCache), first(o.first), len(o.len) {
  pathCache.resize(o.pathCache.size());
  for (int i = 0; i < pathCache.size(); ++i) {
    pathCache[i] = new CacheCross<real,CL>(*o.pathCache[i]);
  }
}

template<bi::Location CL, class IO1>
bi::MCMCCache<CL,IO1>::~MCMCCache() {
  for (int i = 0; i < int(pathCache.size()); ++i) {
    delete pathCache[i];
  }
}

template<bi::Location CL, class IO1>
bi::MCMCCache<CL,IO1>& bi::MCMCCache<CL,IO1>::operator=(
    const MCMCCache<CL,IO1>& o) {
  parent_type::operator=(o);

  llCache = o.llCache;
  lpCache = o.lpCache;
  parameterCache = o.parameterCache;
  first = o.first;
  len = o.len;

  pathCache.resize(o.pathCache.size());
  for (int i = 0; i < pathCache.size(); ++i) {
    pathCache[i] = new CacheCross<real,CL>();
    pathCache[i] = *o.pathCache[i];
  }

  return *this;
}

template<bi::Location CL, class IO1>
real bi::MCMCCache<CL,IO1>::readLogLikelihood(const int p) {
  /* pre-condition */
  BI_ASSERT(p >= first && p < first + len);

  return llCache.get(p - first);
}

template<bi::Location CL, class IO1>
void bi::MCMCCache<CL,IO1>::writeLogLikelihood(const int p, const real ll) {
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

template<bi::Location CL, class IO1>
real bi::MCMCCache<CL,IO1>::readLogPrior(const int p) {
  /* pre-condition */
  BI_ASSERT(p >= first && p < first + len);

  return lpCache.get(p - first);
}

template<bi::Location CL, class IO1>
void bi::MCMCCache<CL,IO1>::writeLogPrior(const int p, const real lp) {
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

template<bi::Location CL, class IO1>
template<class V1>
void bi::MCMCCache<CL,IO1>::readParameter(const int p, V1 theta) {
  /* pre-condition */
  BI_ASSERT(p >= first && p < first + len);

  theta = parameterCache.get(p - first);
}

template<bi::Location CL, class IO1>
template<class V1>
void bi::MCMCCache<CL,IO1>::writeParameter(const int p, const V1 theta) {
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

template<bi::Location CL, class IO1>
template<class M1>
void bi::MCMCCache<CL,IO1>::readPath(const int p, M1 X) {
  /* pre-condition */
  BI_ASSERT(len == 0 || (p >= first && p <= first + len));
  BI_ASSERT(X.size2() == pathCache.size());

  for (int t = 0; t < pathCache.size(); ++t) {
    column(X, t) = pathCache[t]->get(p - first);
  }
}

template<bi::Location CL, class IO1>
template<class M1>
void bi::MCMCCache<CL,IO1>::writePath(const int p, const M1 X) {
  /* pre-condition */
  BI_ASSERT(len == 0 || (p >= first && p <= first + len));

  if (len == 0) {
    first = p;
  }
  if (p - first == len) {
    len = p - first + 1;
  }
  for (int t = 0; t < pathCache.size(); ++t) {
    pathCache[t]->set(p - first, column(X, t));
  }
}

template<bi::Location CL, class IO1>
bool bi::MCMCCache<CL,IO1>::isFull() const {
  return len == NUM_SAMPLES;
}

template<bi::Location CL, class IO1>
void bi::MCMCCache<CL,IO1>::swap(MCMCCache<CL,IO1>& o) {
  parent_type::swap(o);
  llCache.swap(o.llCache);
  lpCache.swap(o.lpCache);
  parameterCache.swap(o.parameterCache);
  pathCache.swap(o.pathCache);
  std::swap(first, o.first);
  std::swap(len, o.len);
}

template<bi::Location CL, class IO1>
void bi::MCMCCache<CL,IO1>::clear() {
  llCache.clear();
  lpCache.clear();
  parameterCache.clear();
  for (int t = 0; t < int(pathCache.size()); ++t) {
    pathCache[t]->clear();
  }
  first = 0;
  len = 0;
  parent_type::clear();
}

template<bi::Location CL, class IO1>
void bi::MCMCCache<CL,IO1>::empty() {
  llCache.empty();
  lpCache.empty();
  parameterCache.empty();
  for (int k = 0; k < pathCache.size(); ++k) {
    pathCache[k]->empty();
    delete pathCache[k];
  }
  pathCache.resize(0);
  first = 0;
  len = 0;
  parent_type::empty();
}

template<bi::Location CL, class IO1>
void bi::MCMCCache<CL,IO1>::flush() {
  parent_type::writeLogLikelihoods(first, llCache.get(0, len));
  parent_type::writeLogPriors(first, lpCache.get(0, len));
  parent_type::writeParameters(first, parameterCache.get(0, len));

  llCache.flush();
  lpCache.flush();
  parameterCache.flush();

  flushPaths(R_VAR);
  flushPaths(D_VAR);
  parent_type::flush();
}

template<bi::Location CL, class IO1>
void bi::MCMCCache<CL,IO1>::flushPaths(const VarType type) {
  /* don't do it time-by-time, too much seeking in looping over variables
   * several times... */
  //for (int k = 0; k < int(pathCache.size()); ++k) {
  //  IO1::writeState(k, first, pathCache[k]->get(0, len));
  //  pathCache[k]->flush();
  //}
  /* ...do it variable-by-variable instead, and loop over times several
   * times */
  Var* var;
  int id, i, k, start, size;

  for (id = 0; id < m.getNumVars(type); ++id) {
    var = m.getVar(type, id);
    start = var->getStart() + ((type == D_VAR) ? m.getNetSize(R_VAR) : 0);
    size = var->getSize();

    for (k = 0; k < int(pathCache.size()); ++k) {
      IO1::writeStateVar(type, id, k, first,
          columns(pathCache[k]->get(0, len), start, size));
    }
  }
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::MCMCCache<CL,IO1>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object < parent_type > (*this);
  ar & llCache;
  ar & lpCache;
  ar & parameterCache;
  ar & pathCache;
  ar & first;
  ar & len;
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::MCMCCache<CL,IO1>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < parent_type > (*this);
  ar & llCache;
  ar & lpCache;
  ar & parameterCache;
  ar & pathCache;
  ar & first;
  ar & len;
}

#endif
