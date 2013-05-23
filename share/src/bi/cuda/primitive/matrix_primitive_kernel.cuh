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
CUDA_FUNC_GLOBAL void kernel_gather_rows(const V1 map, const M1 X, M2 Y);

/**
 * @copydoc gather_columns
 */
template<class V1, class M1, class M2>
CUDA_FUNC_GLOBAL void kernel_gather_columns(const V1 map, const M1 X, M2 Y);

/**
 * @copydoc gather_matrix
 */
template<class V1, class V2, class M1, class M2>
CUDA_FUNC_GLOBAL void kernel_gather_matrix(const V1 map1, const V2 map2,
    const M1 X, M2 Y);

/**
 * @copydoc scatter_rows
 */
template<class V1, class M1, class M2>
CUDA_FUNC_GLOBAL void kernel_scatter_rows(const V1 map, const M1 X, M2 Y);

/**
 * @copydoc scatter_columns
 */
template<class V1, class M1, class M2>
CUDA_FUNC_GLOBAL void kernel_scatter_columns(const V1 map, const M1 X, M2 Y);

/**
 * @copydoc scatter_matrix
 */
template<class V1, class V2, class M1, class M2>
CUDA_FUNC_GLOBAL void kernel_scatter_matrix(const V1 map1, const V2 map2,
    const M1 X, M2 Y);
}

template<class V1, class M1, class M2>
CUDA_FUNC_GLOBAL void bi::kernel_gather_rows(const V1 map, const M1 X, M2 Y) {
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  if (i < map.size()) {
    Y(i, j) = X(map(i), j);
  }
}

template<class V1, class M1, class M2>
CUDA_FUNC_GLOBAL void bi::kernel_gather_columns(const V1 map, const M1 X,
    M2 Y) {
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  if (i < Y.size1()) {
    Y(i, j) = X(i, map(j));
  }
}

template<class V1, class V2, class M1, class M2>
CUDA_FUNC_GLOBAL void bi::kernel_gather_matrix(const V1 map1, const V2 map2,
    const M1 X, M2 Y) {
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  if (i < map1.size() && j < map2.size()) {
    Y(i, j) = X(map1(i), map2(j));
  }
}

template<class V1, class M1, class M2>
CUDA_FUNC_GLOBAL void bi::kernel_scatter_rows(const V1 map, const M1 X, M2 Y) {
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  if (i < map.size()) {
    Y(map(i), j) = X(i, j);
  }
}

template<class V1, class M1, class M2>
CUDA_FUNC_GLOBAL void bi::kernel_scatter_columns(const V1 map, const M1 X, M2 Y) {
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  if (i < Y.size1()) {
    Y(i, map(j)) = X(i, j);
  }
}

template<class V1, class V2, class M1, class M2>
CUDA_FUNC_GLOBAL void bi::kernel_scatter_matrix(const V1 map1, const V2 map2,
    const M1 X, M2 Y) {
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  if (i < map1.size() && j < map2.size()) {
    Y(map1(i), map2(j)) = X(i, j);
  }
}

#endif
