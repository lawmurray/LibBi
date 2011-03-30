/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MATH_TEMP_VECTOR_HPP
#define BI_MATH_TEMP_VECTOR_HPP

#include "host_vector.hpp"
#include "../misc/pooled_allocator.hpp"
#include "../misc/pinned_allocator.hpp"

#include "boost/typeof/typeof.hpp"
#include "boost/mpl/if.hpp"

namespace bi {
/**
 * @internal
 *
 * @group math_host
 *
 * @tparam V1 Vector type.
 */
template<class V1>
struct host_vector_map_type {
  /**
   * Allocator type.
   */
  typedef pooled_allocator<pinned_allocator<typename V1::value_type> > allocator_type;

  /**
   * Equivalent host type.
   */
  typedef host_vector<typename V1::value_type, allocator_type> host_type;

  /**
   * Type map. Host types map to same, device types map to equivalent %host
   * type.
   */
  typedef typename boost::mpl::if_c<V1::on_device,host_type,V1>::type type;

};

/**
 * @internal
 *
 * @group math_host
 */
template<class V1, bool on_device>
struct host_vector_map {
  static typename host_vector_map_type<V1>::type* map(const V1 x);
};

/**
 * @internal
 *
 * @group math_host
 */
template<class V1>
struct host_vector_map<V1,true> {
  static typename host_vector_map_type<V1>::type* map(const V1 x) {
    BOOST_AUTO(result, new typename host_vector_map_type<V1>::type(x.size()));
    *result = x;
    return result;
  }
};

/**
 * @internal
 *
 * @group math_host
 */
template<class V1>
struct host_vector_map<V1,false> {
  static typename host_vector_map_type<V1>::type* map(const V1 x) {
    return new typename host_vector_map_type<V1>::type(x);
  }
};

/**
 * @internal
 *
 * @group math_host
 */
template<class T1>
struct host_vector_temp_type {
  /**
   * Allocator type.
   */
  typedef pooled_allocator<pinned_allocator<T1> > allocator_type;

  /**
   * Temp type.
   */
  typedef host_vector<T1,allocator_type> type;
};

/**
 * Construct temporary vector on %host.
 *
 * @ingroup math_host
 *
 * @tparam T1 Scalar type.
 */
template<class T1>
typename host_vector_temp_type<T1>::type* host_temp_vector(const int size);

/**
 * Construct temporary vector on %host as copy of arbitrary vector.
 *
 * @ingroup math_host
 *
 * @tparam V1 Vector type.
 *
 * @return If argument is a %host vector, returns a reference to the same.
 * If argument is a device vector, constructs a suitable %host vector using
 * pooled memory, asynchronously copies contents to this, and returns a
 * reference to it.
 */
template<class V1>
typename host_vector_map_type<V1>::type* host_map_vector(const V1 x);

}

template<class T1>
inline typename bi::host_vector_temp_type<T1>::type* bi::host_temp_vector(
    const int size) {
  return new typename bi::host_vector_temp_type<T1>::type(size);
}

template<class V1>
inline typename bi::host_vector_map_type<V1>::type* bi::host_map_vector(
    const V1 x) {
  return host_vector_map<V1,V1::on_device>::map(x);
}

#endif
