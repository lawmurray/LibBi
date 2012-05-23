/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_SPARSESTATICUPDATERKERNEL_CUH
#define BI_CUDA_UPDATER_SPARSESTATICUPDATERKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for sparse static updates.
 *
 * @tparam B Model type.
 * @tparam B1 Mask block type.
 *
 * @param ids Ids of variables to update.
 * @param start Offset into output.
 * @param P Number of trajectories to update.
 */
template<class B, class B1>
CUDA_FUNC_GLOBAL void kernelSparseStaticUpdater(const B1 mask, const int start,
    const int P);

}

template<class B, class B1>
void bi::kernelSparseStaticUpdater(const B1 mask, const int start, const int P) {

}

#endif
