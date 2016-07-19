/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MPI_RESAMPLER_DISTRIBUTEDRESAMPLER_HPP
#define BI_MPI_RESAMPLER_DISTRIBUTEDRESAMPLER_HPP

#include "../../resampler/Resampler.hpp"

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
class DistributedResampler: public Resampler<R> {
public:
  /**
   * Constructor.
   *
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   * @param Use anytime mode? Triggers correction of marginal likelihood
   * estimates for the elimination of active particles.
   */
  DistributedResampler(const double essRel = 0.5, const bool anytime = false);

  /**
   * @copydoc Resampler::reduce(const V1, double*)
   */
  template<class V1>
  double reduce(const V1 lws, double* lW);

  /**
   * @copydoc Resampler::resample(Random&, V1, V2, O1&)
   */
  template<class S1>
  bool resample(Random& rng, const ScheduleElement now, S1& s)
      throw (ParticleFilterDegeneratedException);

private:
  /**
   * Redistribute offspring around processes so that all processes have same
   * number of particles.
   *
   * @tparam M1 Matrix type.
   * @tparam O1 State type.
   *
   * @param[in,out] O Offspring matrix. Rows index particles, columns index
   * processes.
   * @param[in,out] s State.
   */
  template<class M1, class S1>
  void redistribute(M1 O, S1& s);

  /**
   * Rotate particles around process so that all processes have a random
   * sample.
   *
   * @tparam S1 State type.
   *
   * @param[in,out] s State.
   */
  template<class S1>
  void rotate(S1& s);

  /**
   * @name Timing
   */
  //@{
  /**
   * Report resample timings to stderr.
   */
  static void reportResample(int timestep, int rank, long usecs);

  /**
   * Report redistribution timings to stderr.
   */
  static void reportRedistribute(int timestep, int rank, long usecs);
  //@}
};
}

#include "../mpi.hpp"
#include "../../math/temp_vector.hpp"
#include "../../math/temp_matrix.hpp"
#include "../../math/view.hpp"

template<class R>
bi::DistributedResampler<R>::DistributedResampler(const double essRel,
    const bool anytime) :
    Resampler<R>(essRel, anytime) {
  //
}

template<class R>
template<class V1>
double bi::DistributedResampler<R>::reduce(const V1 lws, double* lW) {
  typedef typename V1::value_type T1;

  boost::mpi::communicator world;
  const int size = world.size();
  T1 mx, sum1, sum2;
  int P;

  P = lws.size();
  P = boost::mpi::all_reduce(world, P, std::plus<int>());
  mx = max_reduce(lws);
  mx = boost::mpi::all_reduce(world, mx, boost::mpi::maximum<T1>());

  sum1 = op_reduce(lws, nan_minus_and_exp_functor<T1>(mx), 0.0,
      thrust::plus<T1>());
  sum1 = boost::mpi::all_reduce(world, sum1, std::plus<T1>());

  sum2 = op_reduce(lws, nan_minus_exp_and_square_functor<T1>(mx), 0.0,
      thrust::plus<T1>());
  sum2 = boost::mpi::all_reduce(world, sum2, std::plus<T1>());

  if (lW != NULL) {
    *lW = mx + bi::log(sum1);
    if (this->anytime) {
      *lW -= bi::log(double(size * (P - 1)));
    } else {
      *lW -= bi::log(double(size * P));
    }
  }
  return (sum1 * sum1) / sum2;
}

template<class R>
template<class S1>
bool bi::DistributedResampler<R>::resample(Random& rng,
    const ScheduleElement now, S1& s)
        throw (ParticleFilterDegeneratedException) {
  boost::mpi::communicator world;
  const int rank = world.rank();
  const int size = world.size();
  const int P = s.size();

  bool r = (now.isObserved() || now.hasBridge())
      && s.ess < this->essRel * size * P;
  if (r) {
#if ENABLE_DIAGNOSTICS == 2
    synchronize();
    TicToc clock;
#endif

    typename temp_host_matrix<real>::type Lws(P, size);
    typename temp_host_matrix<int>::type O(P, size);
    typename temp_host_vector<int>::type as1(P);

    /* gather weights to root */
    if (S1::on_device) {
      /* gather takes raw pointer, so need to copy to host */
      typename temp_host_vector<real>::type lws1(P);
      lws1 = s.logWeights();
      synchronize();
      boost::mpi::gather(world, lws1.buf(), P, vec(Lws).buf(), 0);
    } else {
      /* already on host */
      boost::mpi::gather(world, s.logWeights().buf(), P, vec(Lws).buf(), 0);
    }

    /* compute offspring on root and broadcast */
    if (rank == 0) {
      typename precompute_type<R,S1::temp_int_vector_type::location>::type pre;

      R::precompute(vec(Lws), pre);
      R::offspring(rng, vec(Lws), P * size, vec(O), pre);
    }
    boost::mpi::broadcast(world, O.buf(), P * size, 0);

#if ENABLE_DIAGNOSTICS == 2
    long usecs = clock.toc();
    const int timesteps = s.front()->getOutput().size() - 1;
    reportResample(timesteps, rank, usecs);
#endif
    redistribute(O, s);
    offspringToAncestors(column(O, rank), as1);
    permute(as1);
    s.gather(now, as1);
    set_elements(s.logWeights(), s.logLikelihood);
    this->shuffle(rng, s);
    rotate(s);
  } else if (now.hasOutput()) {
    seq_elements(s.ancestors(), 0);
  }
  return r;
}

template<class R>
void bi::DistributedResampler<R>::reportResample(int timestep, int rank,
    long usecs) {
  fprintf(stderr, "%d: DistributedResampler::resample proc %d %ld us\n",
      timestep, rank, usecs);
}

template<class R>
template<class M1, class S1>
void bi::DistributedResampler<R>::redistribute(M1 O, S1& s) {
  typedef typename temp_host_vector<int>::type int_vector_type;

#if ENABLE_DIAGNOSTICS == 2
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
  std::list < boost::mpi::request > reqs;

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

    /* advance to next nonzero of sender */
    while (O(sendi, sendr) == 0) {
      ++sendi;
    }

    /* advance to next zero of receiver */
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
      reqs.push_back(world.irecv(sendr, tag, *s.s1s[recvi]));
      reqs.push_back(world.irecv(sendr, tag + 1, *s.out1s[recvi]));
    } else if (rank == sendr) {
      reqs.push_back(world.isend(recvr, tag, *s.s1s[sendi]));
      reqs.push_back(world.isend(recvr, tag + 1, *s.out1s[sendi]));
    }
    tag += 2;

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

#if ENABLE_DIAGNOSTICS == 2
  long usecs = clock.toc();
  const int timesteps = s.front()->getOutput().size() - 1;
  reportRedistribute(timesteps, rank, usecs);
#endif
}

template<class R>
template<class S1>
void bi::DistributedResampler<R>::rotate(S1& s) {
  boost::mpi::communicator world;
  const int rank = world.rank();
  const int size = world.size();
  const int P = s.size();

  std::vector < boost::mpi::request > sends1(P), sends2(P);
  int p, sendr, recvr;

  /* pipeline first round of sends */
  const int buf = 8;
  for (p = 0; p < buf * size && p < P; ++p) {
    if (p % size > 0) {
      recvr = (rank + p) % size;
      sends1[p] = world.isend(recvr, 2 * rank * p, *s.s1s[p]);
      sends2[p] = world.isend(recvr, 2 * rank * p + 1, *s.out1s[p]);
    }
  }

  /* receive incoming one by one */
  for (p = 0; p < P; ++p) {
    if (p % size > 0) {
      /* receive new particle for this position */
      sendr = (rank + size - (p % size)) % size;
      world.recv(sendr, 2 * sendr * p, s.s2);
      world.recv(sendr, 2 * sendr * p + 1, s.out2);

      /* ensure old particle in this position has been sent */
      sends1[p].wait();
      sends2[p].wait();

      /* replace the old particle with the new particle */
      s.s2.swap(*s.s1s[p]);
      s.out2.swap(*s.out1s[p]);

      /* continue the pipeline */
      recvr = (rank + p) % size;
      if (p + buf * size < P) {
        sends1[p + buf * size] = world.isend(recvr,
            2 * rank * (p + buf * size), *s.s1s[p]);
        sends2[p + buf * size] = world.isend(recvr,
            2 * rank * (p + buf * size) + 1, *s.out1s[p]);
      }
    }
  }
}

template<class R>
void bi::DistributedResampler<R>::reportRedistribute(int timestep, int rank,
    long usecs) {
  fprintf(stderr, "%d: DistributedResampler::redistribute proc %d %ld us\n",
      timestep, rank, usecs);
}

#endif
