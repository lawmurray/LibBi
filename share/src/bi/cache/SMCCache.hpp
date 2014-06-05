/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CACHE_SMCCACHE_HPP
#define BI_CACHE_SMCCACHE_HPP

#include "MCMCCache.hpp"
#include "../buffer/SMCNetCDFBuffer.hpp"

#include "boost/serialization/split_member.hpp"
#include "boost/serialization/base_object.hpp"

namespace bi {
/**
 * Cache for SMCNetCDFBuffer reads and writes.
 *
 * @ingroup io_cache
 *
 * @tparam CL Location.
 * @tparam IO1 Buffer type.
 */
template<Location CL = ON_HOST, class IO1 = SMCNetCDFBuffer>
class SMCCache: public MCMCCache<CL,IO1> {
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
  SMCCache(B& m, IO1& out = NULL);

  /**
   * Shallow copy.
   */
  SMCCache(const SMCCache<CL,IO1>& o);

  /**
   * Destructor.
   */
  ~SMCCache();

  /**
   * Deep assignment.
   */
  SMCCache<CL,IO1>& operator=(const SMCCache<CL,IO1>& o);

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
  void swap(SMCCache<CL,IO1>& o);

private:
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
template<class B>
bi::SMCCache<CL,IO1>::SMCCache(B& m, IO1& out) :
    MCMCCache<CL,IO1>(m, out), out(out) {
  //
}

template<bi::Location CL, class IO1>
bi::SMCCache<CL,IO1>::SMCCache(const SMCCache<CL,IO1>& o) :
    MCMCCache<CL,IO1>(o), out(o.out) {
  //
}

template<bi::Location CL, class IO1>
bi::SMCCache<CL,IO1>& bi::SMCCache<CL,IO1>::operator=(
    const SMCCache<CL,IO1>& o) {
  MCMCCache<CL,IO1>::operator=(o);
  out = o.out;

  return *this;
}

template<bi::Location CL, class IO1>
bi::SMCCache<CL,IO1>::~SMCCache() {
  //flush();
}

template<bi::Location CL, class IO1>
template<class V1>
void bi::SMCCache<CL,IO1>::readLogWeights(V1 lws) const {
  /* pre-condition */
  BI_ASSERT(!(equals<IO1,OutputBuffer>::value));

  out.readLogWeights(lws);
}

template<bi::Location CL, class IO1>
template<class V1>
void bi::SMCCache<CL,IO1>::writeLogWeights(const V1 lws) {
  out.writeLogWeights(lws);
}

template<bi::Location CL, class IO1>
template<class V1>
void bi::SMCCache<CL,IO1>::readLogEvidences(V1 les) const {
  /* pre-condition */
  BI_ASSERT(!(equals<IO1,OutputBuffer>::value));

  out.readLogEvidences(les);
}

template<bi::Location CL, class IO1>
template<class V1>
void bi::SMCCache<CL,IO1>::writeLogEvidences(const V1 les) {
  out.writeLogEvidences(les);
}

template<bi::Location CL, class IO1>
void bi::SMCCache<CL,IO1>::swap(SMCCache<CL,IO1>& o) {
  MCMCCache<CL,IO1>::swap(o);
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::SMCCache<CL,IO1>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object < MCMCCache<CL,IO1> > (*this);
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::SMCCache<CL,IO1>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < MCMCCache<CL,IO1> > (*this);
}

#endif
