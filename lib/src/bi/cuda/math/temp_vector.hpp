/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_MATH_TEMP_VECTOR_HPP
#define BI_CUDA_MATH_TEMP_VECTOR_HPP

#include "vector.hpp"
#include "../../math/temp_vector.hpp"
#include "../../misc/device_allocator.hpp"
#include "../../misc/pooled_allocator.hpp"

#include "boost/typeof/typeof.hpp"
#include "boost/mpl/if.hpp"

namespace bi {
/**
 * @internal
 *
 * @group math_gpu
 */
template<class V1>
struct gpu_vector_map_type {
  /**
   * Allocator type.
   */
  typedef pooled_allocator<device_allocator<typename V1::value_type> > allocator_type;

  /**
   * Equivalent device type.
   */
  typedef gpu_vector<typename V1::value_type, allocator_type> gpu_type;

  /**
   * Type map. Device types map to same, host types map to equivalent device
   * type.
   */
  typedef typename boost::mpl::if_c<V1::on_device,V1,gpu_type>::type type;

};

/**
 * @internal
 *
 * @group math_gpu
 */
template<class V1, bool on_device>
struct gpu_vector_map {
  static typename gpu_vector_map_type<V1>::type* map(const V1 x);
};

/**
 * @internal
 *
 * @group math_gpu
 */
template<class V1>
struct gpu_vector_map<V1,true> {
  static typename gpu_vector_map_type<V1>::type* map(const V1 x) {
    return new typename gpu_vector_map_type<V1>::type(x);
  }
};

/**
 * @internal
 *
 * @group math_gpu
 */
template<class V1>
struct gpu_vector_map<V1,false> {
  static typename gpu_vector_map_type<V1>::type* map(const V1 x) {
    BOOST_AUTO(result, new typename gpu_vector_map_type<V1>::type(x.size()));
    *result = x;
    return result;
  }
};

/**
 * @internal
 *
 * @group math_gpu
 */
template<class T1>
struct gpu_vector_temp_type {
  /**
   * Allocator type.
   */
  typedef pooled_allocator<device_allocator<T1> > allocator_type;

  /**
   * Temp type.
   */
  typedef gpu_vector<T1,allocator_type> type;
};

/**
 * Construct temporary vector on device.
 *
 * @ingroup math_gpu
 *
 * @tparam T1 Scalar type.
 */
template<class T1>
typename gpu_vector_temp_type<T1>::type* gpu_temp_vector(const int size);

/**
 * Construct temporary vector on device as copy of arbitrary vector.
 *
 * @ingroup math_gpu
 *
 * @tparam V1 Vector type.
 *
 * @return If argument is a device vector, returns a reference to the same.
 * If argument is a host vector, constructs a suitable device vector using
 * pooled memory, asynchronously copies contents to this, and returns a
 * reference to it.
 */
template<class V1>
typename gpu_vector_map_type<V1>::type* gpu_map_vector(const V1 x);

/**
 * Construct temporary vector, as copy of arbitrary vector, on device.
 *
 * @ingroup math
 *
 * @tparam V1 Vector type.
 *
 * @param x Vector.
 *
 * @return A temporary device vector that is a copy of @p x. This is
 * guaranteed to be contiguous in memory (i.e. <tt>inc() == 1</tt>).
 */
template<class V1>
typename gpu_vector_temp_type<typename V1::value_type>::type* gpu_duplicate_vector(const V1 x);

/**
 * @internal
 *
 * @group math
 */
template<class V1, class V2>
struct vector_map_type {
  typedef typename boost::mpl::if_c<V1::on_device,
      gpu_vector_map_type<V2>,
      host_vector_map_type<V2>
  >::type::type type;
};

/**
 * @internal
 *
 * @group math
 */
template<class V1, class V2, bool on_device>
struct vector_map {
  static typename vector_map_type<V1,V2>::type* map(const V2 x);
};

/**
 * @internal
 *
 * @group math
 */
template<class V1, class V2>
struct vector_map<V1,V2,true> {
  static typename vector_map_type<V1,V2>::type* map(const V2 x) {
    return gpu_map_vector(x);
  }
};

/**
 * @internal
 *
 * @group math
 */
template<class V1, class V2>
struct vector_map<V1,V2,false> {
  static typename vector_map_type<V1,V2>::type* map(const V2 x) {
    return host_map_vector(x);
  }
};

/**
 * @internal
 *
 * @group math
 */
template<class V1>
struct vector_temp_type {
  typedef typename boost::mpl::if_c<V1::on_device,
      gpu_vector_temp_type<typename V1::value_type>,
      host_vector_temp_type<typename V1::value_type>
  >::type::type type;
};

/**
 * Construct temporary vector, on %host or device according to example vector
 * type.
 *
 * @ingroup math
 *
 * @tparam V1 Vector type.
 *
 * @return If @p V1 is a device type, returns a temporary device vector. If
 * @p V1 is a host type, returns a temporary host vector.
 */
template<class V1>
typename vector_temp_type<V1>::type* temp_vector(const int size);

/**
 * Construct temporary vector, as copy of arbitrary vector, on %host or device
 * according to example vector.
 *
 * @ingroup math
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param y Example vector.
 * @param x Vector.
 *
 * @return If @p V1 and @p V2 both device, or both host, types, returns a
 * reference to @p x. If one of @p V1 and @p V2 is on device, and the other
 * a host, type, constructs a suitable vector on the same location (host
 * or device) as @p V1, asynchronously copies @p x contents to it, and
 * returns a reference to it.
 *
 * Note that @p y is not used, its presence is only for conveniently calling
 * the function without explicit template arguments.
 */
template<class V1, class V2>
typename vector_map_type<V1,V2>::type* map_vector(const V1 y, const V2 x);

/**
 * Construct temporary vector, as copy of arbitrary vector, on %host or device
 * according to that vector.
 *
 * @ingroup math
 *
 * @tparam V1 Vector type.
 *
 * @param x Vector.
 *
 * @return A temporary vector at the same location (host or device), and with
 * the same contents, as @p x.
 */
template<class V1>
typename vector_temp_type<V1>::type* duplicate_vector(const V1 x);

}

template<class T1>
inline typename bi::gpu_vector_temp_type<T1>::type* bi::gpu_temp_vector(
    const int size) {
  return new typename bi::gpu_vector_temp_type<T1>::type(size);
}

template<class V1>
inline typename bi::gpu_vector_map_type<V1>::type* bi::gpu_map_vector(
    const V1 x) {
  return gpu_vector_map<V1,V1::on_device>::map(x);
}

template<class V1>
inline typename bi::gpu_vector_temp_type<typename V1::value_type>::type* bi::gpu_duplicate_vector(
    const V1 x) {
  typedef typename V1::value_type T1;
  typedef typename gpu_vector_temp_type<T1>::type V2;

  V2 *result = new V2(x.size());
  *result = x;

  return result;
}

template<class V1>
inline typename bi::vector_temp_type<V1>::type* bi::temp_vector(
    const int size) {
  return new typename bi::vector_temp_type<V1>::type(size);
}

template<class V1, class V2>
inline typename bi::vector_map_type<V1,V2>::type* bi::map_vector(const V1 y,
    const V2 x) {
  return vector_map<V1,V2,V1::on_device>::map(x);
}

template<class V1>
inline typename bi::vector_temp_type<V1>::type* bi::duplicate_vector(
    const V1 x) {
  typename bi::vector_temp_type<V1>::type* y =
      new typename bi::vector_temp_type<V1>::type(x.size());
  *y = x;
  return y;
}

#endif
