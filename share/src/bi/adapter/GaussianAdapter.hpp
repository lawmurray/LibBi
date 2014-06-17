/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ADAPTER_GAUSSIANADAPTER_HPP
#define BI_ADAPTER_GAUSSIANADAPTER_HPP

#include "SampleAdapter.hpp"

namespace bi {
/**
 * Adapter for Gaussian proposal.
 *
 * @ingroup method_adapter
 */
template<class B, Location L>
class GaussianAdapter: public SampleAdapter<B,L> {
public:
  /**
   * @copydoc Adapter::Adapter()
   */
  GaussianAdapter(const int initialSize = SampleAdapter<B,L>::DEFAULT_INITIAL_SIZE);

  /**
   * @copydoc Adapter::adapt()
   */
  template<class Q>
  void adapt(Q& q) const;
};
}

#include "../pdf/misc.hpp"

template<class B, bi::Location L>
bi::GaussianAdapter<B,L>::GaussianAdapter(const int initialSize) :
    SampleAdapter<B,L>(initialSize) {
  //
}

template<class B, bi::Location L>
template<class Q>
void bi::GaussianAdapter<B,L>::adapt(Q& q) const {
  typedef typename loc_temp_vector<L,real>::type temp_vector_type;
  typedef typename loc_temp_matrix<L,real>::type temp_matrix_type;

  const int N = q.size();
  temp_vector_type mu(N), ws(this->P);
  temp_matrix_type Sigma(N, N);

  expu_elements(this->lws, ws);
  mean(this->X, ws, mu);
  cov(this->X, ws, mu, Sigma);

  q.mean() = mu;
  chol(Sigma, q.std());
  q.init();
}

#endif
