/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_MATH_TEMP_HPP
#define BI_CUDA_MATH_TEMP_HPP

#include "matrix.hpp"
#include "../../math/temp_matrix.hpp"
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
template<class M1>
struct gpu_matrix_map_type {
  /**
   * Allocator type.
   */
  typedef pooled_allocator<device_allocator<typename M1::value_type> >
      allocator_type;

  /**
   * Equivalent device type.
   */
  typedef gpu_matrix<typename M1::value_type, allocator_type> gpu_type;

  /**
   * Type map. Device types map to same, host types map to equivalent device
   * type.
   */
  typedef typename boost::mpl::if_c<M1::on_device,M1,gpu_type>::type type;

};

/**
 * @internal
 *
 * @group math_gpu
 */
template<class M1, bool on_device>
struct gpu_matrix_map {
  static typename gpu_matrix_map_type<M1>::type* map(const M1 X);
};

/**
 * @internal
 *
 * @group math_gpu
 */
template<class M1>
struct gpu_matrix_map<M1,true> {
  static typename gpu_matrix_map_type<M1>::type* map(const M1 X) {
    return new typename gpu_matrix_map_type<M1>::type(X);
  }
};

/**
 * @internal
 *
 * @group math_gpu
 */
template<class M1>
struct gpu_matrix_map<M1,false> {
  static typename gpu_matrix_map_type<M1>::type* map(const M1 X) {
    BOOST_AUTO(result, new typename gpu_matrix_map_type<M1>::type(X.size1(),
        X.size2()));
    *result = X;
    return result;
  }
};

/**
 * @internal
 *
 * @group math_gpu
 */
template<class T1>
struct gpu_matrix_temp_type {
  /**
   * Allocator type.
   */
  typedef pooled_allocator<device_allocator<T1> > allocator_type;

  /**
   * Temp type.
   */
  typedef gpu_matrix<T1,allocator_type> type;
};

/**
 * Construct temporary matrix on device.
 *
 * @ingroup math_gpu
 *
 * @tparam T1 Scalar type.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 */
template<class T1>
typename gpu_matrix_temp_type<T1>::type* gpu_temp_matrix(const int rows,
    const int cols);

/**
 * Construct temporary matrix on device as copy of arbitrary matrix.
 *
 * @ingroup math_gpu
 *
 * @tparam M1 Matrix type.
 *
 * @return If argument is a device matrix, returns a reference to the same.
 * If argument is a host matrix, constructs a suitable device matrix using
 * pooled memory, asynchronously copies contents to this, and returns a
 * reference to it.
 */
template<class M1>
typename gpu_matrix_map_type<M1>::type* gpu_map_matrix(const M1 X);

/**
 * @internal
 *
 * @group math
 */
template<class M1, class M2>
struct matrix_map_type {
  typedef typename boost::mpl::if_c<M1::on_device,
      gpu_matrix_map_type<M2>,
      host_matrix_map_type<M2>
  >::type::type type;
};

/**
 * @internal
 *
 * @group math
 */
template<class M1, class M2, bool on_device>
struct matrix_map {
  static typename matrix_map_type<M1,M2>::type* map(const M2 X);
};

/**
 * @internal
 *
 * @group math
 */
template<class M1, class M2>
struct matrix_map<M1,M2,true> {
  static typename matrix_map_type<M1,M2>::type* map(const M2 X) {
    return gpu_map_matrix(X);
  }
};

/**
 * @internal
 *
 * @group math
 */
template<class M1, class M2>
struct matrix_map<M1,M2,false> {
  static typename matrix_map_type<M1,M2>::type* map(const M2 X) {
    return host_map_matrix(X);
  }
};

/**
 * @internal
 *
 * @group math
 */
template<class M1>
struct matrix_temp_type {
  typedef typename boost::mpl::if_c<M1::on_device,
      gpu_matrix_temp_type<typename M1::value_type>,
      host_matrix_temp_type<typename M1::value_type>
  >::type::type type;
};

/**
 * Construct temporary matrix, on %host or device according to example matrix
 * type.
 *
 * @ingroup math
 *
 * @tparam M1 Matrix type.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 *
 * @return If @p M1 is a device type, returns a temporary device matrix. If
 * @p M1 is a host type, returns a temporary host matrix.
 */
template<class M1>
typename matrix_temp_type<M1>::type* temp_matrix(
    const int rows, const int cols);

/**
 * Construct temporary matrix, on %host or device according to example matrix
 * type, as copy of arbitrary matrix.
 *
 * @ingroup math
 *
 * @tparam M1 Matrix type.
 * @tparam M2 Matrix type.
 *
 * @param Y
 * @param X Matrix.
 *
 * @return If @p M1 and @p M2 both device, or both host, types, returns a
 * reference to @p X. If one of @p M1 and @p M2 is a device, and the other
 * a host, type, constructs a suitable matrix on the same location (host
 * or device) as @p M1, asynchronously copies @p X contents to it, and
 * returns a reference to it.
 *
 * Note that @p Y is not used, its presence is only for conveniently calling
 * the function without explicit types.
 */
template<class M1, class M2>
typename matrix_map_type<M1,M2>::type* map_matrix(const M1 Y, const M2 X);

/**
 * Construct temporary matrix, as copy of arbitrary matrix, on %host or device
 * according to example matrix.
 *
 * @ingroup math
 *
 * @tparam M1 Matrix type.
 * @tparam M2 Matrix type.
 *
 * @param Y
 * @param X Matrix.
 *
 * @return If @p M1 and @p M2 both device, or both host, types, returns a
 * reference to @p X. If one of @p M1 and @p M2 is a device, and the other
 * a host, type, constructs a suitable matrix on the same location (host
 * or device) as @p M1, asynchronously copies @p X contents to it, and
 * returns a reference to it.
 *
 * Note that @p Y is not used, its presence is only for conveniently calling
 * the function without explicit types.
 */
template<class M1, class M2>
typename matrix_map_type<M1,M2>::type* map_matrix(const M1 Y, const M2 X);

/**
 * Construct temporary matrix, as copy of arbitrary matrix, on %host or device
 * according to that matrix.
 *
 * @ingroup math
 *
 * @tparam M1 Matrix type.
 *
 * @param X Matrix.
 *
 * @return A temporary matrix at the same location (host or device), and with
 * the same contents, as @p x.
 */
template<class M1>
typename matrix_temp_type<M1>::type* duplicate_matrix(const M1 X);

}

template<class T1>
inline typename bi::gpu_matrix_temp_type<T1>::type* bi::gpu_temp_matrix(
    const int rows, const int cols) {
  return new typename bi::gpu_matrix_temp_type<T1>::type(rows, cols);
}

template<class M1>
inline typename bi::gpu_matrix_map_type<M1>::type* bi::gpu_map_matrix(
    const M1 X) {
  return gpu_matrix_map<M1,M1::on_device>::map(X);
}

template<class M1>
inline typename bi::matrix_temp_type<M1>::type* bi::temp_matrix(
    const int rows, const int cols) {
  return new typename bi::matrix_temp_type<M1>::type(rows, cols);
}

template<class M1, class M2>
inline typename bi::matrix_map_type<M1,M2>::type* bi::map_matrix(const M1 Y,
    const M2 X) {
  return matrix_map<M1,M2,M1::on_device>::map(X);
}

template<class M1>
inline typename bi::matrix_temp_type<M1>::type* bi::duplicate_matrix(
    const M1 X) {
  typename bi::matrix_temp_type<M1>::type* Y =
      new typename bi::matrix_temp_type<M1>::type(X.size1(), X.size2());
  *Y = X;
  return Y;
}

#endif
