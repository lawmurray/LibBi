/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ADAPTER_GAUSSIANADAPTER_HPP
#define BI_ADAPTER_GAUSSIANADAPTER_HPP

#include "../pdf/GaussianPdf.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * Adapter for Gaussian proposal.
 *
 * @ingroup method_adapter
 */
template<class B, Location L>
class GaussianAdapter {
public:
  typedef GaussianPdf<> proposal_type;

  /**
   * @copydoc Adapter::Adapter()
   */
  GaussianAdapter();

  /**
   * @copydoc Adapter::add()
   */
  template<class V1, class V2>
  void add(const V1 x, const V2 lws);

  /**
   * @copydoc Adapter::stop()
   */
  bool stop(const int k);

  /**
   * @copydoc Adapter::adapt()
   */
  void adapt(const int k) throw (CholeskyException);

  /**
   * @copydoc Adapter::get()
   */
  proposal_type& get(const int k);

//private:
  /**
   * Proposal distribution.
   */
  proposal_type q;
};
}

#include "../pdf/misc.hpp"

template<class B, bi::Location L>
bi::GaussianAdapter<B,L>::GaussianAdapter() :
    q(B::NP) {
  //
}

template<class B, bi::Location L>
template<class V1, class V2>
void bi::GaussianAdapter<B,L>::add(const V1 x, const V2 lws) {

}

/**
 * @copydoc Adapter::stop()
 */
template<class B, bi::Location L>
bool bi::GaussianAdapter<B,L>::stop(const int k) {

}

template<class B, bi::Location L>
void bi::GaussianAdapter<B,L>::adapt(const int k)
    throw (CholeskyException) {
//  typedef typename loc_temp_vector<L,real>::type temp_vector_type;
//  typedef typename loc_temp_matrix<L,real>::type temp_matrix_type;
//
//  static const int N = B::N;
//  temp_vector_type mu(N), ws(this->P);
//  temp_matrix_type Sigma(N, N);
//
//  expu_elements(subrange(this->lws, 0, this->P), ws);
//  mean(rows(this->X, 0, this->P), ws, mu);
//  cov(rows(this->X, 0, this->P), ws, mu, Sigma);
//
//  chol(Sigma, q.std());  // do first, in case throws, so q unmodified
//  q.mean() = mu;
//  q.init();
}

template<class B, bi::Location L>
typename bi::GaussianAdapter<B,L>::proposal_type& bi::GaussianAdapter<B,L>::get(
    const int k) {
  return q;
}

#endif
