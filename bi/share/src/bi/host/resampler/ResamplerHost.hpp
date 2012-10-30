/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_RESAMPLER_RESAMPLERHOST_HPP
#define BI_HOST_RESAMPLER_RESAMPLERHOST_HPP

#include "../../primitive/vector_primitive.hpp"

#include "thrust/sequence.h"
#include "thrust/sort.h"
#include "thrust/fill.h"

template<class V1>
void bi::ResamplerHost::permute(V1 as) {
  /* pre-condition */
  BI_ASSERT(!V1::on_device);

  typename V1::size_type i;
  typename V1::value_type j;
  for (i = 0; i < as.size(); ++i) {
    if (as[i] != i && as[i] < as.size() && as[as[i]] != as[i]) {
      /* swap */
      j = as[as[i]];
      as[as[i]] = as[i];
      as[i] = j;
      --i; // repeat for new value
    }
  }
}

template<class V1, class M1>
void bi::ResamplerHost::copy(const V1 as, M1 X) {
  /* pre-condition */
  BI_ASSERT(!V1::on_device);
  BI_ASSERT(!M1::on_device);
  BI_ASSERT(as.size() <= X.size1());

  int p;
  for (p = 0; p < as.size(); ++p) {
    if (as[p] != p) {
      row(X, p) = row(X, as[p]);
    }
  }
}

template<class V1, class V2>
void bi::ResamplerHost::ancestorsToOffspring(const V1 as, V2 os) {
  const int P = as.size();
  os.clear();

  #pragma omp parallel for
  for (int i = 0; i < P; ++i) {
    #pragma omp atomic
    ++os(as(i));
  }

  /* post-condition */
  BI_ASSERT(thrust::reduce(os.begin(), os.end()) == P);
}

#endif
