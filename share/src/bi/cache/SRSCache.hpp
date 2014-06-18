/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_SRSCACHE_HPP
#define BI_CACHE_SRSCACHE_HPP

#include "SimulatorCache.hpp"
#include "Cache1D.hpp"
#include "CacheCross.hpp"
#include "../null/SMCNullBuffer.hpp"

namespace bi {
/**
 * Cache for marginal SRS.
 *
 * @ingroup io_cache
 *
 * @tparam IO1 Output type.
 * @tparam CL Location.
 */
template<Location CL = ON_HOST, class IO1 = SMCNullBuffer>
class SRSCache: public MCMCCache<CL,IO1> {
public:
  typedef MCMCCache<CL,IO1> parent_type;

  /**
   * @copydoc MCMCBuffer::MCMCBuffer()
   */
  SRSCache(const Model& m, const size_t P = 0, const size_t T = 0,
      const std::string& file = "", const FileMode mode = READ_ONLY,
      const SchemaMode schema = DEFAULT);

  /**
   * Shallow copy constructor.
   */
  SRSCache(const SRSCache<CL,IO1>& o);

  /**
   * Destructor.
   */
  ~SRSCache();

  /**
   * Deep assignment operator.
   */
  SRSCache<CL,IO1>& operator=(const SRSCache<CL,IO1>& o);

  /**
   * Read log-weight.
   *
   * @param p Sample index.
   *
   * @return Log-weight.
   */
  real readLogWeight(const int p);

  /**
   * Write log-likelihood.
   *
   * @param p Sample index.
   * @param ll Log-likelihood.
   */
  void writeLogWeight(const int p, const real ll);

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(SRSCache<CL,IO1>& o);

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
   * Log-weights cache.
   */
  Cache1D<real,CL> lwCache;

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
bi::SRSCache<CL,IO1>::SRSCache(const Model& m, const size_t P, const size_t T,
    const std::string& file, const FileMode mode, const SchemaMode schema) :
    parent_type(m, P, T, file, mode, schema), lwCache(
        parent_type::NUM_SAMPLES) {
  //
}

template<bi::Location CL, class IO1>
bi::SRSCache<CL,IO1>::SRSCache(const SRSCache<CL,IO1>& o) :
    parent_type(o), lwCache(o.llCache) {
  //
}

template<bi::Location CL, class IO1>
bi::SRSCache<CL,IO1>::~SRSCache() {
  //
}

template<bi::Location CL, class IO1>
bi::SRSCache<CL,IO1>& bi::SRSCache<CL,IO1>::operator=(
    const SRSCache<CL,IO1>& o) {
  parent_type::operator=(o);
  lwCache = o.lwCache;

  return *this;
}

template<bi::Location CL, class IO1>
real bi::SRSCache<CL,IO1>::readLogWeight(const int p) {
  /* pre-condition */
  BI_ASSERT(p >= this->first && p < this->first + this->len);

  return lwCache.get(p - this->first);
}

template<bi::Location CL, class IO1>
void bi::SRSCache<CL,IO1>::writeLogWeight(const int p, const real lw) {
  /* pre-condition */
  BI_ASSERT(
      this->len == 0 || (p >= this->first && p <= this->first + this->len));

  if (this->len == 0) {
    this->first = p;
  }
  if (p - this->first == this->len) {
    this->len = p - this->first + 1;
  }
  lwCache.set(p - this->first, lw);
}

template<bi::Location CL, class IO1>
void bi::SRSCache<CL,IO1>::swap(SRSCache<CL,IO1>& o) {
  parent_type::swap(o);
  lwCache.swap(o.lwCache);
}

template<bi::Location CL, class IO1>
void bi::SRSCache<CL,IO1>::clear() {
  lwCache.clear();
  parent_type::clear();
}

template<bi::Location CL, class IO1>
void bi::SRSCache<CL,IO1>::empty() {
  lwCache.empty();
  parent_type::empty();
}

template<bi::Location CL, class IO1>
void bi::SRSCache<CL,IO1>::flush() {
  parent_type::writeLogWeights(this->first, lwCache.get(0, this->len));
  parent_type::flush();
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::SRSCache<CL,IO1>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object < parent_type > (*this);
  ar & lwCache;
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::SRSCache<CL,IO1>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < parent_type > (*this);
  ar & lwCache;
}

#endif
