/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/MixturePdf.hpp
 */
#ifndef BI_PDF_GAUSSIANMIXTUREPDF_HPP
#define BI_PDF_GAUSSIANMIXTUREPDF_HPP

#include "MixturePdf.hpp"
#include "GaussianPdf.hpp"

namespace bi {
/**
 * Gaussian mixture probability density.
 *
 * @ingroup math_pdf
 *
 * @tparam V1 Vector type.
 * @tparam M1 Matrix type.
 *
 * @see MixturePdf, GaussianPdf
 */
template<class V1 = host_vector<>, class M1 = host_matrix<> >
class GaussianMixturePdf : public MixturePdf<GaussianPdf<V1,M1> > {
public:
  /**
   * @copydoc MixturePdf::MixturePdf()
   */
  GaussianMixturePdf();

  /**
   * @copydoc MixturePdf::MixturePdf(int)
   */
  GaussianMixturePdf(const int N);

  /**
   * Destructor.
   */
  ~GaussianMixturePdf();

  /**
   * Fit Gaussian mixture using EM.
   *
   * @tparam M2 Matrix type.
   *
   * @param rng Random number generator.
   * @param K Number of components.
   * @param X Samples.
   * @param eps \f$\epsilon\f$ Threshold on relative likelihood change between
   * consecutive iterations to be considered converged.
   *
   * A @p K component Gaussian mixture is fit to @p X using Expectation-
   * Maximisation (EM) with random initialisation.
   *
   * @return True if the refit was successful, false otherwise (such as for
   * lack of convergence).
   */
  template<class M2>
  bool refit(Random& rng, const int K, const M2& X, const real eps = 1e-2);

  /**
   * Fit Gaussian mixture using EM.
   *
   * @tparam M2 Matrix type.
   * @tparam V2 Vector type.
   *
   * @param rng Random number generator.
   * @param K Number of components.
   * @param X Samples.
   * @param y Weights.
   * @param eps \f$\epsilon\f$ Threshold on relative likelihood change between
   * consecutive iterations to be considered converged.
   *
   * A @p K component Gaussian mixture is fit to @p X, weighted by @p ws,
   * using Expectation-Maximisation (EM) with random initialisation.
   *
   * @return True if the refit was successful, false otherwise (such as for
   * lack of convergence).
   */
  template<class M2, class V2>
  bool refit(Random& rng, const int K, const M2& X, const V2& y,
      const real eps = 1e-2);

private:
  #ifndef __CUDACC__
  /**
   * Serialize or restore from serialization.
   */
  template<class Archive>
  void serialize(Archive& ar, const unsigned version);

  /*
   * Boost.Serialization requirements.
   */
  friend class boost::serialization::access;
  #endif
};

}

#ifndef __CUDACC__
#include "boost/serialization/base_object.hpp"
#endif

template<class V1, class M1>
bi::GaussianMixturePdf<V1,M1>::GaussianMixturePdf() {
  //
}

template<class V1, class M1>
bi::GaussianMixturePdf<V1,M1>::GaussianMixturePdf(const int N) :
    MixturePdf<GaussianPdf<V1,M1> >(N) {
  //
}

template<class V1, class M1>
template<class M2>
bool bi::GaussianMixturePdf<V1,M1>::refit(Random& rng, const int K,
    const M2& X, const real eps) {
  return false;
}

template<class V1, class M1>
template<class M2, class V2>
bool bi::GaussianMixturePdf<V1,M1>::refit(Random& rng, const int K,
    const M2& X, const V2& y, const real eps) {
  /* pre-condition */
  assert (y.size() == X.size1());
  assert (this->size() == X.size2());

  const int P = X.size1();
  real W = this->weight();
  real Y = bi::sum(y.begin(), y.end(), 0.0);
  int k, p, j;

  /* allocation */
  this->clear();
  for (k = 0; k < K; ++k) {
    add(GaussianPdf<V1,M1>(this->N));
  }

  /* random initialisation */
  std::vector<real> weights(K, 0);
  for (k = 0; k < K; ++k) {
    this->get(k).mean().clear();
    this->get(k).cov().clear();
  }
  for (p = 0; p < P; ++p) {
    k = rng.uniformInt(0, K - 1);
    axpy(y(p), row(X,p), this->get(k).mean());
    syr(y(p), row(X,p), this->get(k).cov());
    weights[k] += y(p);
  }
  for (k = 0; k < K; ++k) {
    scal(1.0/weights[k], this->get(k).mean());
    for (j = 0; j < this->N; ++j) {
      scal(1.0/weights[k], column(this->get(k).cov(), j));
    }
    syr(-1.0, this->get(k).mean(), this->get(k).cov());
  }
  for (k = 0; k < K; ++k) {
    this->get(k).init();
  }

  /**
   * Expectation-maximisation.
   *
   * Let \f$X: \Re^P \times \Re^N\f$ be the matrix of samples and
   * \f$\mathbf{c}: \Re^K\f$ the vector of mixture components. Let
   * \f$\mathbf{w}: \Re^K\f$ be component weights, and
   * \f$\mathbf{y}: \Re^P\f$ sample weights. Let
   * \f$T: \Re^P \times \Re^K\f$ be a matrix of probabilities
   * \f$T_{p,k}\f$. Let
   * \f$\mathbf{a}: \Re^K\f$, \f$\mathbf{b}: \Re^K\f$,
   * \f$\mathbf{c}: \Re^P\f$ and
   * \f$Y: \Re^P \times \Re^K\f$ be temporaries.
   */
  BOOST_AUTO(T, temp_matrix<M1>(P,K));
  BOOST_AUTO(U, temp_matrix<M1>(P,K));
  BOOST_AUTO(a, temp_vector<V1>(P));
  BOOST_AUTO(b, temp_vector<V1>(P));
  BOOST_AUTO(c, temp_vector<V1>(K));
  BOOST_AUTO(Z, temp_matrix<M1>(P,this->N));
  BOOST_AUTO(M, temp_matrix<M1>(this->N,K));

  bool converged = K == 1;
  real l1 = 0.0, l2;
  int steps = 0;
  while (!converged) {
    /**
     * \f[T_{p,k} = p(\mathbf{X}_{p,*}\,|\,c_k)\f]
     */
    for (k = 0; k < K; ++k) {
      BOOST_AUTO(col, column(*T,k));
      this->get(k).densities(X, col);
    }

    /**
     * Weight densities by normalised mixture weights:
     *
     * \f[U = TD/W\f]
     *
     * where \f$D\f$ is diagonal matrix with diagonal \f$\mathbf{w}\f$.
     */
    gdmm(1.0/W, this->ws, *T, 0.0, *U, 'R');

    /**
     * Normalise rows of \f$U\f$:
     *
     * \f[\mathbf{a} = T\mathbf{w}/W\f]
     * \f[T = DU\f]
     *
     * where \f$D\f$ is a diagonal matrix with diagonal \f$1/\mathbf{a}\f$.
     */
    gemv(1.0/W, *T, this->ws, 0.0, *a);
    element_rcp(a->begin(), a->end(), b->begin());
    gdmm(1.0, *b, *U, 0.0, *T);
    #ifndef NDEBUG
    for (int i = 0; i < T->size1(); ++i) {
      assert(relErr(bi::sum(row(*T,i).begin(), row(*T,i).end(), 0.0), 1.0) < 1.0e-5);
    }
    #endif

    /**
     * Compute new weights:
     *
     * \f[\mathbf{w} = T^T\mathbf{y}\f]
     */
    gemv(1.0/Y, *T, y, 0.0, this->ws, 'T');
    thrust::inclusive_scan(this->ws.begin(), this->ws.end(), this->Ws.begin());
    assert(relErr(this->weight(), 1.0) < 1.0e-5);
    W = this->weight();

    /**
     * Normalise columns of \f$T\f$:
     *
     * \f[U = TD\f]
     *
     * where \f$D\f$ is a diagonal matrix with diagonal \f$1/\mathbf{w}\f$
     * (using new weights, which are column sums).
     */
    element_rcp(this->ws.begin(), this->ws.end(), c->begin());
    gdmm(1.0/Y, y, *T, 0.0, *U);
    gdmm(1.0, *c, *U, 0.0, *T, 'R');
    #ifndef NDEBUG
    for (j = 0; j < T->size2(); ++j) {
      assert(relErr(bi::sum(column(*T,j).begin(), column(*T,j).end(), 0.0), 1.0) < 1.0e-5);
    }
    #endif

    /**
     * Compute new mixture component means:
     *
     * \f{eqnarray*}[
     * M &=& X^TU \\
     * \boldsymbol{\mu}_k &=& M_{*,k}
     * \f}
     */
    gemm(1.0, X, *T, 0.0, *M, 'T');
    for (k = 0; k < K; ++k) {
      this->get(k).setMean(column(*M,k));
    }

    /**
     * Compute new mixture component covariances:
     *
     * \f[\Sigma_k = \sum_p U_{p,k} \mathbf{X}_{p,*}\mathbf{X_{p,*}^T}\,.\f]
     *
     * So let:
     *
     * \f[T_{p,k} = \sqrt{U_{p,k}}\,,\f]
     *
     * and:
     *
     * \f[Z = DX\f]
     *
     * where \f$D\f$ is a diagonal matrix with diagonal \f$T_{*,k}\f$. Then:
     *
     * \f[Sigma_k = Z^TZ - \boldsymbol{\mu}_k\boldsymbol{\mu}_k^T\,.\f]
     */
    for (k = 0; k < K; ++k) {
      element_sqrt(column(*T,k).begin(), column(*T,k).end(),
          column(*U,k).begin());
      gdmm(1.0, column(*U,k), X, 0.0, *Z);
      syrk(1.0, *Z, 0.0, this->get(k).cov(), 'L', 'T');
      syr(-1.0, this->get(k).mean(), this->get(k).cov());
    }

    /**
     * Compute (unnormalised) likelihood:
     *
     * \f{eqnarray*}
     * l &=& \prod_p a_p^{y_p} \\
     * \log l &=& \sum_p y_p \log a_p
     * \f}
     */
    element_log(a->begin(), a->end(), b->begin());
    l2 = dot(y, *b);
    assert (steps == 0 || l2 > l1); // likelihood should always increase

    /**
     * Verbose output.
     */
//    std::cerr << "STEP " << steps << std::endl;
//    std::cerr << "----------------------------------------------" << std::endl;
//    std::cerr << "l = " << l2 << std::endl;
//    for (k = 0; k < K; ++k) {
//      std::cerr << "Component " << k << ": " << std::endl;
//      for (i = 0; i < this->N; ++i) {
//        std::cerr << this->get(k).mean()(i) << " ";
//      }
//      std::cerr << std::endl;
//      for (i = 0; i < this->N; ++i) {
//        for (j = 0; j < this->N; ++j) {
//          std::cerr << this->get(k).cov()(i,j) << " ";
//        }
//        std::cerr << std::endl;
//      }
//    }

    /**
     * Consider converged if \f$(l_2 - l_1)/l_2 < \epsilon\f$, where
     * \f$l_2\f$ is likelihood of this iteration, and \f$l_1\f$ that of
     * previous.
     */
    converged = steps > 0 && expm1(fabs(l2 - l1)) < eps;
    l1 = l2;
    ++steps;

    for (k = 0; k < K; ++k) {
      this->get(k).init();
    }
  };

  synchronize();
  delete T;
  delete U;
  delete a;
  delete b;
  delete c;
  delete Z;
  delete M;

  return converged;
}

template<class V1, class M1>
bi::GaussianMixturePdf<V1,M1>::~GaussianMixturePdf() {
  //
}

#ifndef __CUDACC__
template<class V1, class M1>
template<class Archive>
void bi::GaussianMixturePdf<V1,M1>::serialize(Archive& ar,
    const unsigned version) {
  ar & boost::serialization::base_object<MixturePdf<GaussianPdf<V1,M1> > >(*this);
}
#endif

#endif
