/**
 * @file
 *
 * @author Lawrence Murray <murray@stats.ox.ac.uk>
 */
#include "GaussianAdapter.hpp"

#include "../mpi/mpi.hpp"
#include "../math/view.hpp"
#include "../math/operation.hpp"
#include "../primitive/vector_primitive.hpp"

bi::GaussianAdapter::GaussianAdapter(const bool local,
    const double scale) :
    local(local), scale(scale) {
  clear();
}

void bi::GaussianAdapter::adapt() throw (CholeskyException) {
  const int NP = smu.size();

  /* mean and covariance */
  mu.resize(NP);
  Sigma.resize(NP, NP);

  axpy(1.0/W, smu, mu, true);
  axpy(1.0/W, vec(sSigma), vec(Sigma), true);
  syr(-1.0, mu, Sigma);

  /* Cholesky factor of covariance */
  U.resize(NP, NP);
  chol(Sigma, U);

  /* scale for local moves */
  if (local) {
    matrix_scal(scale, U);
  }

  /* determinant */
  detU = prod_reduce(diagonal(U));
}

#ifdef ENABLE_MPI
void bi::GaussianAdapter::distributedAdapt() throw (CholeskyException) {
  const int NP = smu.size();

  boost::mpi::communicator world;
  const int size = world.size();

  double W1;
  typename temp_host_vector<real>::type smu1(NP);
  typename temp_host_matrix<real>::type sSigma1(NP, NP), Smu(NP, size), SSigma(NP*NP, size);

  W1 = boost::mpi::all_reduce(world, W, std::plus<double>());
  boost::mpi::all_gather(world, smu.buf(), NP, vec(Smu).buf());
  boost::mpi::all_gather(world, sSigma.buf(), NP*NP, vec(SSigma).buf());

  sum_columns(Smu, smu1);
  sum_columns(SSigma, vec(sSigma1));

  /* mean and covariance */
  mu.resize(NP);
  Sigma.resize(NP, NP);

  axpy(1.0/W1, smu1, mu, true);
  axpy(1.0/W1, vec(sSigma1), vec(Sigma), true);
  syr(-1.0, mu, Sigma);

  /* Cholesky factor of covariance */
  U.resize(NP, NP);
  chol(Sigma, U);

  /* scale for local moves */
  if (local) {
    matrix_scal(scale, U);
  }

  /* determinant */
  detU = prod_reduce(diagonal(U));
}
#endif

void bi::GaussianAdapter::clear() {
  smu.clear();
  sSigma.clear();
  W = 0.0;
}
