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
#include "../netcdf/SMCNetCDFBuffer.hpp"

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
   * Pass-through constructor.
   */
  SMCCache();

  /**
   * Pass-through constructor.
   */
  template<class T1>
  SMCCache(T1& o1);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2>
  SMCCache(T1& o1, T2& o2);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3>
  SMCCache(T1& o1, T2& o2, T3& o3);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3, class T4>
  SMCCache(T1& o1, T2& o2, T3& o3, T4& o4);

  /**
   * Pass-through constructor.
   */
  template<class T1, class T2, class T3, class T4, class T5>
  SMCCache(T1& o1, T2& o2, T3& o3, T4& o4, T5& o5);

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
bi::SMCCache<CL,IO1>::SMCCache() {
  //
}

template<bi::Location CL, class IO1>
template<class T1>
bi::SMCCache<CL,IO1>::SMCCache(T1& o1) :
    MCMCCache<CL,IO1>(o1) {
  //
}

template<bi::Location CL, class IO1>
template<class T1, class T2>
bi::SMCCache<CL,IO1>::SMCCache(T1& o1, T2& o2) :
    MCMCCache<CL,IO1>(o1, o2) {
  //
}

template<bi::Location CL, class IO1>
template<class T1, class T2, class T3>
bi::SMCCache<CL,IO1>::SMCCache(T1& o1, T2& o2, T3& o3) :
    MCMCCache<CL,IO1>(o1, o2, o3) {
  //
}

template<bi::Location CL, class IO1>
template<class T1, class T2, class T3, class T4>
bi::SMCCache<CL,IO1>::SMCCache(T1& o1, T2& o2, T3& o3, T4& o4) :
    MCMCCache<CL,IO1>(o1, o2, o3, o4) {
  //
}

template<bi::Location CL, class IO1>
template<class T1, class T2, class T3, class T4, class T5>
bi::SMCCache<CL,IO1>::SMCCache(T1& o1, T2& o2, T3& o3, T4& o4, T5& o5) :
    MCMCCache<CL,IO1>(o1, o2, o3, o4, o5) {
  //
}

template<bi::Location CL, class IO1>
bi::SMCCache<CL,IO1>::SMCCache(const SMCCache<CL,IO1>& o) :
    MCMCCache<CL,IO1>(o) {
  //
}

template<bi::Location CL, class IO1>
bi::SMCCache<CL,IO1>& bi::SMCCache<CL,IO1>::operator=(
    const SMCCache<CL,IO1>& o) {
  MCMCCache<CL,IO1>::operator=(o);

  return *this;
}

template<bi::Location CL, class IO1>
bi::SMCCache<CL,IO1>::~SMCCache() {
  //flush();
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
