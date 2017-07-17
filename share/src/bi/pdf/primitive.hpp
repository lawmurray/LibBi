/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PDF_PRIMITIVE_HPP
#define BI_PDF_PRIMITIVE_HPP

namespace bi {
/**
 * Compute Gaussian densities of standardised variates.
 *
 * @ingroup math_pdf
 *
 * @tparam M1 Matrix type.
 * @tparam T1 Scalar type.
 * @tparam V1 Vector type.
 *
 * @param Z Standardised Gaussian variates.
 * @param logZ Log of normalisation constant.
 * @param[in,out] p Densities.
 * @param clear Clear @p p? If false, the density is multiplied into @p p.
 */
template<class M1, class T1, class V1>
void gaussian_densities(const M1 Z, const T1 logZ, V1 p,
    const bool clear = false);

/**
 * Compute Gaussian log-densities of standardised variates.
 *
 * @ingroup math_pdf
 *
 * @tparam M1 Matrix type.
 * @tparam T1 Scalar type.
 * @tparam V1 Vector type.
 *
 * @param Z Standardised Gaussian variates.
 * @param logZ Log of normalisation constant.
 * @param[in,out] lp Log-densities.
 * @param clear Clear @p p? If false, the density is multiplied into @p p.
 */
template<class M1, class T1, class V1>
void gaussian_log_densities(const M1 Z, const T1 logZ, V1 lp,
    const bool clear = false);

/**
 * Compute Gamma densities.
 *
 * @ingroup math_pdf
 *
 * @tparam M1 Matrix type.
 * @tparam T1 Scalar type.
 * @tparam V1 Vector type.
 *
 * @param Z Gamma variates.
 * @param alpha Shape parameter.
 * @param beta Scale parameter.
 * @param[in,out] p Densities.
 * @param clear Clear @p p? If false, the density is multiplied into @p p.
 */
template<class M1, class T1, class V1>
void gamma_densities(const M1 Z, const T1 alpha, const T1 beta,
    V1 p, const bool clear = true);

/**
 * Compute Gamma log-densities.
 *
 * @ingroup math_pdf
 *
 * @tparam M1 Matrix type.
 * @tparam T1 Scalar type.
 * @tparam V1 Vector type.
 *
 * @param Z Gamma variates.
 * @param alpha Shape parameter.
 * @param beta Scale parameter.
 * @param[in,out] p Log-densities.
 * @param clear Clear @p p? If false, the log-density is added to @p p.
 */
template<class M1, class T1, class V1>
void gamma_log_densities(const M1 Z, const T1 alpha, const T1 beta,
    V1 p, const bool clear = true);

/**
 * Compute inverse-Gamma densities.
 *
 * @ingroup math_pdf
 *
 * @tparam M1 Matrix type.
 * @tparam T1 Scalar type.
 * @tparam V1 Vector type.
 *
 * @param Z Inverse-Gamma variates.
 * @param alpha Shape parameter.
 * @param beta Scale parameter.
 * @param[in,out] p Densities.
 * @param clear Clear @p p? If false, the density is multiplied into @p p.
 */
template<class M1, class T1, class V1>
void inverse_gamma_densities(const M1 Z, const T1 alpha, const T1 beta,
    V1 p, const bool clear = true);

/**
 * Compute inverse-Gamma log-densities.
 *
 * @ingroup math_pdf
 *
 * @tparam M1 Matrix type.
 * @tparam T1 Scalar type.
 * @tparam V1 Vector type.
 *
 * @param Z Inverse-gamma variates.
 * @param alpha Shape parameter.
 * @param beta Scale parameter.
 * @param[in,out] p Log-densities.
 * @param clear Clear @p p? If false, the log-density is added to @p p.
 */
template<class M1, class T1, class V1>
void inverse_gamma_log_densities(const M1 Z, const T1 alpha, const T1 beta,
    V1 p, const bool clear = true);

}

#include "functor.hpp"

template<class M1, class T1, class V1>
void bi::gaussian_densities(const M1 Z, const T1 logZ, V1 p,
    const bool clear) {
  /* pre-condition */
  BI_ASSERT(Z.size1() == p.size());

  typedef typename V1::value_type T2;

  if (clear) {
    dot_rows(Z, p);
    op_elements(p, gaussian_density_functor<T2>(logZ), p);
  } else {
    typename sim_temp_vector<V1>::type p1(p.size());
    dot_rows(Z, p1);
    op_elements(p1, p, p, gaussian_density_update_functor<T2>(logZ));
  }
}

template<class M1, class T1, class V1>
void bi::gaussian_log_densities(const M1 Z, const T1 logZ, V1 p,
    const bool clear) {
  /* pre-condition */
  BI_ASSERT(Z.size1() == p.size());

  typedef typename V1::value_type T2;

  if (clear) {
    dot_rows(Z, p);
    op_elements(p, p, gaussian_log_density_functor<T2>(logZ));
  } else {
    typename sim_temp_vector<V1>::type p1(p.size());
    dot_rows(Z, p1);
    op_elements(p1, p, p, gaussian_log_density_update_functor<T2>(logZ));
  }
}

template<class M1, class T1, class V1>
void bi::gamma_densities(const M1 Z, const T1 alpha, const T1 beta,
    V1 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(Z.size1() == p.size());

  op_elements(vec(Z), vec(Z), gamma_density_functor<T1>(alpha, beta));
  if (clear) {
    prod_columns(Z, p);
  } else {
    typename sim_temp_vector<V1>::type p1(p.size());
    prod_columns(Z, p1);
    mul_elements(p, p1, p);
  }
}

template<class M1, class T1, class V1>
void bi::gamma_log_densities(const M1 Z, const T1 alpha, const T1 beta,
    V1 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(Z.size1() == p.size());

  op_elements(vec(Z), vec(Z), gamma_log_density_functor<T1>(alpha, beta));
  if (clear) {
    sum_columns(Z, p);
  } else {
    typename sim_temp_vector<V1>::type p1(p.size());
    sum_columns(Z, p1);
    add_elements(p, p1, p);
  }
}

template<class M1, class T1, class V1>
void bi::inverse_gamma_densities(const M1 Z, const T1 alpha, const T1 beta,
    V1 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(Z.size1() == p.size());

  op_elements(vec(Z), vec(Z), inverse_gamma_density_functor<T1>(alpha, beta));
  if (clear) {
    prod_columns(Z, p);
  } else {
    typename sim_temp_vector<V1>::type p1(p.size());
    prod_columns(Z, p1);
    mul_elements(p, p1, p);
  }
}

template<class M1, class T1, class V1>
void bi::inverse_gamma_log_densities(const M1 Z, const T1 alpha, const T1 beta,
    V1 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(Z.size1() == p.size());

  op_elements(vec(Z), vec(Z), inverse_gamma_log_density_functor<T1>(alpha, beta));
  if (clear) {
    sum_columns(Z, p);
  } else {
    typename sim_temp_vector<V1>::type p1(p.size());
    sum_columns(Z, p1);
    add_elements(p, p1, p);
  }
}

template<class M1, class T1, class V1>
void bi::poisson_densities(const M1 Z, const T1 lambda,
                                 V1 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(Z.size1() == p.size());

  op_elements(vec(Z), vec(Z), poisson_density_functor<T1>(lambda));
  if (clear) {
    prod_columns(Z, p);
  } else {
    typename sim_temp_vector<V1>::type p1(p.size());
    prod_columns(Z, p1);
    mul_elements(p, p1, p);
  }
}

template<class M1, class T1, class V1>
void bi::poisson_log_densities(const M1 Z, const T1 lambda,
                                     V1 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(Z.size1() == p.size());

  op_elements(vec(Z), vec(Z), poisson_log_density_functor<T1>(lambda));
  if (clear) {
    sum_columns(Z, p);
  } else {
    typename sim_temp_vector<V1>::type p1(p.size());
    sum_columns(Z, p1);
    add_elements(p, p1, p);
  }
}

template<class M1, class T1, class V1>
void bi::negbin_densities(const M1 Z, const T1 mean, const T1 shape,
                                 V1 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(Z.size1() == p.size());

  op_elements(vec(Z), vec(Z), negbin_density_functor<T1>(mean, shape));
  if (clear) {
    prod_columns(Z, p);
  } else {
    typename sim_temp_vector<V1>::type p1(p.size());
    prod_columns(Z, p1);
    mul_elements(p, p1, p);
  }
}

template<class M1, class T1, class V1>
void bi::negbin_log_densities(const M1 Z, const T1 mean, const T1 shape,
                                     V1 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(Z.size1() == p.size());

  op_elements(vec(Z), vec(Z), negbin_log_density_functor<T1>(mean, shape));
  if (clear) {
    sum_columns(Z, p);
  } else {
    typename sim_temp_vector<V1>::type p1(p.size());
    sum_columns(Z, p1);
    add_elements(p, p1, p);
  }
}

template<class M1, class T1, class T2, class V1>
void bi::binomial_densities(const M1 Z, const T1 size, const T2 prob,
                                 V1 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(Z.size1() == p.size());

  op_elements(vec(Z), vec(Z), binomial_density_functor<T1, T2>(size, prob));
  if (clear) {
    prod_columns(Z, p);
  } else {
    typename sim_temp_vector<V1>::type p1(p.size());
    prod_columns(Z, p1);
    mul_elements(p, p1, p);
  }
}

template<class M1, class T1, class T2, class V1>
void bi::binomial_log_densities(const M1 Z, const T1 size, const T2 prob,
                                     V1 p, const bool clear) {
  /* pre-condition */
  BI_ASSERT(Z.size1() == p.size());

  op_elements(vec(Z), vec(Z), binomial_log_density_functor<T1, T2>(size, prob));
  if (clear) {
    sum_columns(Z, p);
  } else {
    typename sim_temp_vector<V1>::type p1(p.size());
    sum_columns(Z, p1);
    add_elements(p, p1, p);
  }
}

#endif
