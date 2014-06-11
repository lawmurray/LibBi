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
  typedef MCMCCache<CL,IO1> parent_type;

  /**
   * @copydoc SMCBuffer::SMCBuffer()
   */
  SMCCache(const Model& m, const std::string& file = "", const FileMode mode =
      READ_ONLY, const SchemaMode schema = DEFAULT, const size_t P = 0,
      const size_t T = 0);

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
bi::SMCCache<CL,IO1>::SMCCache(const Model& m, const std::string& file,
    const FileMode mode, const SchemaMode schema, const size_t P,
    const size_t T) :
    parent_type(m, file, mode, schema, P, T) {
  //
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::SMCCache<CL,IO1>::save(Archive& ar, const unsigned version) const {
  ar & boost::serialization::base_object < parent_type > (*this);
}

template<bi::Location CL, class IO1>
template<class Archive>
void bi::SMCCache<CL,IO1>::load(Archive& ar, const unsigned version) {
  ar & boost::serialization::base_object < parent_type > (*this);
}

#endif
