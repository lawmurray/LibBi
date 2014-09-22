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
  GaussianAdapter(const int initialSize =
      SampleAdapter<B,L>::DEFAULT_INITIAL_SIZE);

  /**
   * @copydoc Adapter::adapt()
   */
  template<class Q1>
  void adapt(Q1& q) const throw (CholeskyException);

  /**
   * @copydoc Adapter::add()
   */
  //template<class V1, class V2>
  //void add(const V1 x, const V2 lws);

};
}

#include "../pdf/misc.hpp"

template<class B, bi::Location L>
bi::GaussianAdapter<B,L>::GaussianAdapter(const int initialSize) :
    SampleAdapter<B,L>(initialSize) {
  //
}

template<class B, bi::Location L>
template<class Q1>
void bi::GaussianAdapter<B,L>::adapt(Q1& q) const throw (CholeskyException) {
  typedef typename loc_temp_vector<L,real>::type temp_vector_type;
  typedef typename loc_temp_matrix<L,real>::type temp_matrix_type;

  const int N = q.size();
  temp_vector_type mu(N), ws(this->P);
  temp_matrix_type Sigma(N, N);

  expu_elements(subrange(this->lws, 0, this->P), ws);
  mean(rows(this->X, 0, this->P), ws, mu);
  cov(rows(this->X, 0, this->P), ws, mu, Sigma);

  chol(Sigma, q.std()); // do first, in case throws, so q unmodified
  q.mean() = mu;
  q.init();
}

#endif
