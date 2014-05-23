/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_MarginalSIRCACHE_HPP
#define BI_CACHE_MarginalSIRCACHE_HPP

#include "ParticleMCMCCache.hpp"
#include "../buffer/SMCNetCDFBuffer.hpp"

#include "boost/serialization/split_member.hpp"
#include "boost/serialization/base_object.hpp"

namespace bi {
/**
 * Cache for SMCNetCDFBuffer reads and writes.
 *
 * @ingroup io_cache
 *
 * @tparam IO1 Buffer type.
 * @tparam CL Location.
 */
template<class IO1 = SMCNetCDFBuffer, Location CL = ON_HOST>
class MarginalSIRCache: public ParticleMCMCCache<IO1,CL> {
public:
  /**
   * Constructor.
   *
   * @tparam B Model type.
   *
   * @param m Model.
   * @param out output buffer.
   */
  template<class B>
  MarginalSIRCache(B& m, IO1* out = NULL);

  /**
   * Shallow copy.
   */
  MarginalSIRCache(const MarginalSIRCache<IO1,CL>& o);

  /**
   * Destructor.
   */
  ~MarginalSIRCache();

  /**
   * Deep assignment.
   */
  MarginalSIRCache<IO1,CL>& operator=(const MarginalSIRCache<IO1,CL>& o);

  /**
   * @copydoc SMCNetCDFBuffer::readLogWeights()
   */
  template<class V1>
  void readLogWeights(V1 lws) const;

  /**
   * @copydoc SMCNetCDFBuffer::writeLogWeights()
   */
  template<class V1>
  void writeLogWeights(const V1 lws);

  /**
   * @copydoc SMCNetCDFBuffer::readLogEvidences()
   */
  template<class V1>
  void readLogEvidences(V1 les) const;

  /**
   * @copydoc SMCNetCDFBuffer::writeLogEvidences()
   */
  template<class V1>
  void writeLogEvidences(const V1 les);

  /**
   * Swap the contents of the cache with that of another.
   */
  void swap(MarginalSIRCache<IO1,CL>& o);

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
 * Factory for creating MarginalSIRCache objects.
 *
 * @ingroup io_cache
 *
 * @see Forcer
 */
template<Location CL = ON_HOST>
struct MarginalSIRCacheFactory {
  /**
   * Create MarginalSIRCache.
   *
   * @return MarginalSIRCache object. Caller has ownership.
   *
   * @see MarginalSIRCache::MarginalSIRCache()
   */
  template<class B, class IO1>
  static MarginalSIRCache<IO1,CL>* create(B& m, IO1* out = NULL) {
    return new MarginalSIRCache<IO1,CL>(m, out);
  }

  /**
   * Create MarginalSIRCache.
   *
   * @return MarginalSIRCache object. Caller has ownership.
   *
   * @see MarginalSIRCache::MarginalSIRCache()
   */
  template<class B>
  static MarginalSIRCache<SMCNetCDFBuffer,CL>* create(B& m) {
    return new MarginalSIRCache<SMCNetCDFBuffer,CL>(m);
  }
};
}

template<class IO1, bi::Location CL>
template<class B>
bi::MarginalSIRCache<IO1,CL>::MarginalSIRCache(B& m, IO1* out) :
    ParticleMCMCCache<IO1,CL>(m, out), out(out) {
  //
}

template<class IO1, bi::Location CL>
bi::MarginalSIRCache<IO1,CL>::MarginalSIRCache(const MarginalSIRCache<IO1,CL>& o) :
    ParticleMCMCCache<IO1,CL>(o), out(o.out) {
  //
}

template<class IO1, bi::Location CL>
bi::MarginalSIRCache<IO1,CL>& bi::MarginalSIRCache<IO1,CL>::operator=(
    const MarginalSIRCache<IO1,CL>& o) {
  ParticleMCMCCache<IO1,CL>::operator=(o);
  out = o.out;

  return *this;
}

template<class IO1, bi::Location CL>
bi::MarginalSIRCache<IO1,CL>::~MarginalSIRCache() {
  //flush();
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::MarginalSIRCache<IO1,CL>::readLogWeights(V1 lws) const {
  /* pre-condition */
  BI_ASSERT(out != NULL);

  out->readLogWeights(lws);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::MarginalSIRCache<IO1,CL>::writeLogWeights(const V1 lws) {
  if (out != NULL) {
    out->writeLogWeights(lws);
  }
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::MarginalSIRCache<IO1,CL>::readLogEvidences(V1 les) const {
  /* pre-condition */
  BI_ASSERT(out != NULL);

  out->readLogEvidences(les);
}

template<class IO1, bi::Location CL>
template<class V1>
void bi::MarginalSIRCache<IO1,CL>::writeLogEvidences(const V1 les) {
  if (out != NULL) {
    out->writeLogEvidences(les);
  }
}

template<class IO1, bi::Location CL>
void bi::MarginalSIRCache<IO1,CL>::swap(MarginalSIRCache<IO1,CL>& o) {
  ParticleMCMCCache<IO1,CL>::swap(o);
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::MarginalSIRCache<IO1,CL>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object<ParticleMCMCCache<IO1,CL> >(*this);
}

template<class IO1, bi::Location CL>
template<class Archive>
void bi::MarginalSIRCache<IO1,CL>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object<ParticleMCMCCache<IO1,CL> >(*this);
}

#endif
