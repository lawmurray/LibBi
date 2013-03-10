/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_PRIMITIVE_MATRIXPRIMITIVEKERNEL_CUH
#define BI_CUDA_PRIMITIVE_MATRIXPRIMITIVEKERNEL_CUH

namespace bi {
/**
 * @copydoc gather_rows
 */
template<class V1, class M1, class M2>
CUDA_FUNC_GLOBAL void kernel_gather_rows(const V1 as, const M1 X, M2 Y);

/**
 * @copydoc scatter_rows
 */
template<class V1, class M1, class M2>
CUDA_FUNC_GLOBAL void kernel_scatter_rows(const V1 map, const M1 X, M2 Y);
}

template<class V1, class M1, class M2>
CUDA_FUNC_GLOBAL void bi::kernel_gather_rows(const V1 map, const M1 X, M2 Y) {
  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int id = blockIdx.y*blockDim.y + threadIdx.y;

  if (p < map.size()/* && map(p) != p*/) {
    // ^ the extra condition above destroys coalesced reads/writes
    Y(p, id) = X(map(p), id);
  }
}

template<class V1, class M1, class M2>
CUDA_FUNC_GLOBAL void bi::kernel_scatter_rows(const V1 map, const M1 X, M2 Y) {
  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int id = blockIdx.y*blockDim.y + threadIdx.y;

  if (p < map.size()/* && map(p) != p*/) {
    // ^ the extra condition above destroys coalesced reads/writes
    Y(map(p), id) = X(p, id);
  }
}

#endif
