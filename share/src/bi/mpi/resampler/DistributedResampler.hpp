/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_METHOD_DISTRIBUTEDRESAMPLER_HPP
#define BI_MPI_METHOD_DISTRIBUTEDRESAMPLER_HPP

#include <vector>

namespace bi {
/**
 * Resampler for particle filter, distributed using MPI.
 *
 * @ingroup method_resampler
 *
 * @tparam R Resampler type.
 */
template<class R>
class DistributedResampler: public Resampler {
public:
  /**
   * Constructor.
   *
   * @param base Base resampler.
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   */
  DistributedResampler(R* base, const double essRel = 0.5);

  /**
   * @copydoc concept::Resampler::resample(Random&, V1, V2, O1&)
   */
  template<class V1, class V2, class O1>
  void resample(Random&, V1 lws, V2 as, O1& s)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc Resampler::isTriggered
   */
  template<class V1>
  bool isTriggered(const V1 lws) const
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc Resampler::ess
   */
  template<class V1>
  static typename V1::value_type ess(const V1 lws)
      throw (ParticleFilterDegeneratedException);

private:
  /**
   * Redistribute offspring around processes.
   *
   * @tparam M1 Matrix type.
   * @tparam O1
   *
   * @param[in,out] O Offspring matrix. Rows index particles, columns index
   * processes.
   * @param[in,out] X Matrix of particles in this process.
   */
  template<class M1, class O1>
  static void redistribute(M1 O, O1& s);

  /**
   * @name Timing
   */
  //@{
  /**
   * Report resample timings to stderr.
   */
  static void reportResample(int rank, long usecs);

  /**
   * Report redistribution timings to stderr.
   */
  static void reportRedistribute(int rank, long usecs);
  //@}

  /**
   * Select particle from copy() compatible object.
   *
   * @tparam B Model type.
   * @tparam L Location.
   *
   * @param s State.
   * @param p Index of particle to select.
   *
   * @return Particle.
   */
  template<class B, bi::Location L>
  static typename bi::State<B,L>::matrix_reference_type::vector_reference_type select(
      State<B,L>& s, const int p);

  /**
   * Receive particle.
   *
   * @tparam T1 Assignable type.
   *
   * @param s State.
   * @param p Index of particle to select.
   *
   * @return Particle.
   */
  template<class T1>
  static T1& select(std::vector<T1*>& s, const int p);

  /**
   * Base resampler.
   */
  R* base;
};
}

#include "../mpi.hpp"
#include "../../math/temp_vector.hpp"
#include "../../math/temp_matrix.hpp"
#include "../../math/view.hpp"

#include "boost/mpi/nonblocking.hpp"
#include "boost/mpi/collectives.hpp"

#include <list>

template<class R>
bi::DistributedResampler<R>::DistributedResampler(R* base,
    const double essRel) :
    Resampler(essRel), base(base) {
  //
}

template<class R>
template<class V1, class V2, class O1>
void bi::DistributedResampler<R>::resample(Random& rng, V1 lws, V2 as, O1& s)
    throw (ParticleFilterDegeneratedException) {
  typedef typename V1::value_type T1;

#ifdef ENABLE_TIMING
  synchronize();
  TicToc clock;
#endif

  boost::mpi::communicator world;
  const int rank = world.rank();
  const int size = world.size();
  const int P = lws.size();

  typename temp_host_matrix<real>::type Lws(P, size);
  typename temp_host_matrix<int>::type O(P, size);

  /* gather weights to root */
  if (V1::on_device) {
    /* gather takes raw pointer, so need to copy to host */
    typename temp_host_vector<real>::type lws1(P);
    lws1 = lws;
    synchronize();
    boost::mpi::gather(world, lws1.buf(), P, vec(Lws).buf(), 0);
  } else {
    /* already on host */
    boost::mpi::gather(world, lws.buf(), P, vec(Lws).buf(), 0);
  }

  /* compute offspring on root and broadcast */
  if (rank == 0) {
    base->offspring(rng, vec(Lws), vec(O), P * size);
  }
  boost::mpi::broadcast(world, O, 0);

#ifdef ENABLE_TIMING
  long usecs = clock.toc();
  reportResample(rank, usecs);
#endif

  redistribute(O, s);
  offspringToAncestors(column(O, rank), as);
  permute(as);
  copy(as, s);
  lws.clear();
}

template<class R>
void bi::DistributedResampler<R>::reportResample(int rank, long usecs) {
  std::cerr << "DistributedResampler::resample ";
  std::cerr << " process " << rank << " ";
  std::cerr << usecs << " us";
  std::cerr << std::endl;
}

template<class R>
template<class V1>
bool bi::DistributedResampler<R>::isTriggered(const V1 lws) const
    throw (ParticleFilterDegeneratedException) {
  boost::mpi::communicator world;
  const int size = world.size();
  const int P = lws.size();

  return essRel >= 1.0 || ess(lws) < essRel * size * P;
}

template<class R>
template<class V1>
typename V1::value_type bi::DistributedResampler<R>::ess(const V1 lws)
    throw (ParticleFilterDegeneratedException) {
  typedef typename V1::value_type T1;

  T1 mx, sum1, sum2, result;
  boost::mpi::communicator world;

  mx = max_reduce(lws);
  mx = boost::mpi::all_reduce(world, mx, boost::mpi::maximum<T1>());

  sum1 = op_reduce(lws, nan_minus_and_exp_functor<T1>(mx), 0.0,
      thrust::plus<T1>());
  sum1 = boost::mpi::all_reduce(world, sum1, std::plus<T1>());

  sum2 = op_reduce(lws, nan_minus_exp_and_square_functor<T1>(mx), 0.0,
      thrust::plus<T1>());
  sum2 = boost::mpi::all_reduce(world, sum2, std::plus<T1>());

  result = (sum1 * sum1) / sum2;

  if (result > 0.0) {
    return result;
  } else {
    throw ParticleFilterDegeneratedException();
  }
}

template<class R>
template<class M1, class O1>
void bi::DistributedResampler<R>::redistribute(M1 O, O1& s) {
  typedef typename temp_host_vector<int>::type int_vector_type;

#ifdef ENABLE_TIMING
  synchronize();
  TicToc clock;
#endif

  boost::mpi::communicator world;
  const int rank = world.rank();
  const int size = world.size();
  const int P = O.size1();

  int sendi, recvi, sendj, recvj, sendn, recvn, n, sendr, recvr, tag = 0;

  int_vector_type Ps(size);  // number of particles in each process
  int_vector_type ranks(size);  // ranks sorted by number of particles
  std::list<boost::mpi::request> reqs;

  sum_rows(O, Ps);
  seq_elements(ranks, 0);
  sort_by_key(Ps, ranks);

  /* redistribute offspring */
  sendj = size - 1;
  recvj = 0;
  sendi = 0;
  recvi = 0;

  while (Ps(sendj) > P) {
    /* ranks */
    sendr = ranks(sendj);
    recvr = ranks(recvj);

    /* advance to first nonzero of sender */
    while (O(sendi, sendr) == 0) {
      ++sendi;
    }

    /* advance to first zero of receiver */
    while (O(recvi, recvr) > 0) {
      ++recvi;
    }

    /* determine number of offspring to transfer */
    recvn = P - Ps(recvj);  // max to receive
    sendn = bi::min(Ps(sendj) - P, O(sendi, sendr));  // max to send
    n = bi::min(recvn, sendn);  // actual to transfer

    /* update offspring */
    O(sendi, sendr) -= n;
    O(recvi, recvr) += n;

    /* update particle counts */
    Ps(sendj) -= n;
    Ps(recvj) += n;
    BI_ASSERT(Ps(sendj) >= P);
    BI_ASSERT(Ps(recvj) <= P);

    /* transfer particle */
    if (rank == recvr) {
      reqs.push_back(world.irecv(sendr, tag, select(s, recvi)));
    } else if (rank == sendr) {
      reqs.push_back(world.isend(recvr, tag, select(s, sendi)));
    }
    ++tag;

    if (Ps(sendj) == P) {
      --sendj;
      sendi = 0;
    }
    if (Ps(recvj) == P) {
      ++recvj;
      recvi = 0;
    }
  }

  /* wait for all copies to complete */
  boost::mpi::wait_all(reqs.begin(), reqs.end());

#ifdef ENABLE_TIMING
  long usecs = clock.toc();
  reportRedistribute(rank, usecs);
#endif
}

template<class R>
void bi::DistributedResampler<R>::reportRedistribute(int rank, long usecs) {
  std::cerr << "DistributedResampler::redistribute ";
  std::cerr << " process " << rank << " ";
  std::cerr << usecs << " us";
  std::cerr << std::endl;
}

template<class R>
template<class B, bi::Location L>
typename bi::State<B,L>::matrix_reference_type::vector_reference_type bi::DistributedResampler<
    R>::select(State<B,L>& s, const int p) {
  return row(s.getDyn(), p);
}

template<class R>
template<class T1>
T1& bi::DistributedResampler<R>::select(std::vector<T1*>& s, const int p) {
  return *s[p];
}

#endif
