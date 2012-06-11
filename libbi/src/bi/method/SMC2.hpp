/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2597 $
 * $Date: 2012-05-24 14:33:24 +0000 (Thu, 24 May 2012) $
 */
#ifndef BI_METHOD_SMC2_HPP
#define BI_METHOD_SMC2_HPP

#include "misc.hpp"
#include "../state/State.hpp"
#include "../math/vector.hpp"
#include "../math/matrix.hpp"
#include "../math/view.hpp"
#include "../misc/location.hpp"
#include "../misc/exception.hpp"

#ifndef __CUDACC__
#include "boost/serialization/serialization.hpp"
#endif

using namespace std;

namespace bi {
/**
 * @internal
 *
 * %State of SMC2.
 */
class SMC2State {
public:
  typedef host_vector<real, pinned_allocator<real> > vector_type;
  typedef host_matrix<real, pinned_allocator<real> > matrix_type;
  /**
   * Constructor.
   *
   * @tparam B Model type.
   *
   * @param m Model.
   * @param Ntheta Number of theta-particles.
   * @param Nx Number of x-particles.
   */
  template<class B>
  SMC2State(B& m, const int Ntheta);

  /**
   * Copy constructor.
   */
  SMC2State(const SMC2State& o);

  /**
   * Assignment.
   */
  SMC2State& operator=(const SMC2State& o);

  /**
   * Swap.
   */
  void swap(SMC2State& o);

private:
  #ifndef __CUDACC__
  /**
   * Serialize.
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

template<class B>
inline bi::SMC2State::SMC2State(B& m, const int Ntheta)
{
	//
}

inline bi::SMC2State::SMC2State(const SMC2State& o)
{operator=(o); // ensures deep copy
}

inline bi::SMC2State& bi::SMC2State::operator=(
    const SMC2State& o) {
  return *this;
}

inline void bi::SMC2State::swap(SMC2State& o) {
}

#ifndef __CUDACC__
template<class Archive>
inline void bi::SMC2State::serialize(Archive& ar, const unsigned version) {
}
#endif

namespace bi {
/**
 * Particle Marginal Metropolis-Hastings (PMMH) sampler.
 *
 * See @ref Jones2010 "Jones, Parslow & Murray (2009)", @ref Andrieu2010
 * "Andrieu, Doucet \& Holenstein (2010)". Adaptation is supported according
 * to @ref Haario2001 "Haario, Saksman & Tamminen (2001)".
 *
 * @ingroup method
 *
 * @tparam B Model type
 * @tparam IO1 #concept::SMC2Buffer type.
 * @tparam CL Cache location.
 */
template<class B, class R, class IO1, Location CL = ON_HOST>
class SMC2 {
public:
  /**
   * State type.
   */
  typedef SMC2State state_type;
  typedef host_vector<real, pinned_allocator<real> > vector_type;
  typedef host_matrix<real, pinned_allocator<real> > matrix_type;

  /**
   * Constructor.
   *
   * @tparam IO2 #concept::SparseInputBuffer type.
   *
   * @param m Model.
   * @param out Output.
   * @param flag Indicates how initial conditions should be handled.
   *
   * @see ParticleFilter
   */
  SMC2(B& m, R* resam = NULL, IO1* out = NULL,
      const InitialConditionMode initial = EXCLUDE_INITIAL);

  /**
   * Get output buffer.
   */
  IO1* getOutput();

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  /**
   * Sample.
   *
   * @tparam L Location.
   * @tparam F #concept::Filter type.
   * @tparam IO2 #concept::SparseInputBuffer type.
   *
   * @param rng Random number generator.
   * @param T Length of time to sample over.
   * @param s State.
   * @param filter Filter.
   * @param inInit Initialisation file.
   * @param C Number of samples to draw.
   */
  template<Location L, class F, class IO2>
  void sample(Random& rng, const real T, State<B,L>& s, F* filter,
      IO2* inInit = NULL, const int C = 1);
  template<Location L, class F, class IO2>
    void importancesampling(Random& rng, const real T, State<B,L>& s, F* filter,
        IO2* inInit = NULL, const int C = 1);
private:
  /**
   * Model.
   */
  B& m;

  /**
   * Output.
   */
  IO1* out;

  /**
   * Size of MCMC state. Will include at least all p-vars, and potentially
   * the initial state of d- and c-vars, depending on settings.
   */
  int M;

  /**
   * Initial condition handling mode.
   */
  InitialConditionMode initial;

  /**
   * Resampler for the theta-particles
   */
  R* resam;
  /* net sizes, for convenience */
  static const int ND = net_size<typename B::DTypeList>::value;
  static const int NR = net_size<typename B::RTypeList>::value;
  static const int NP = net_size<typename B::PTypeList>::value;
};

/**
 * Factory for creating SMC2 objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see SMC2
 */
template<Location CL = ON_HOST>
struct SMC2Factory {
  /**
   * Create particle MCMC sampler.
   *
   * @return SMC2 object. Caller has ownership.
   *
   * @see SMC2::SMC2()
   */
  template<class B, class R, class IO1>
  static SMC2<B,R,IO1,CL>* create(B& m, R* resam = NULL,
      IO1* out = NULL, const InitialConditionMode initial = EXCLUDE_INITIAL) {
    return new SMC2<B,R,IO1,CL>(m, resam, out, initial);
  }
};
}

#include "../math/misc.hpp"
#include "../math/io.hpp"
#include <netcdfcpp.h>


template<class B, class R, class IO1, bi::Location CL>
bi::SMC2<B,R,IO1,CL>::SMC2(
    B& m, R*resam, IO1* out, const InitialConditionMode initial) :
    m(m),
    resam(resam),
    out(out),
    M(NP + ((initial == INCLUDE_INITIAL) ? ND : 0)),
    initial(initial)
    {
  std::cerr << "Creating an SMC^2 object..."<< std::endl;
}

template<class B, class R, class IO1, bi::Location CL>
template<bi::Location L, class F, class IO2>
void bi::SMC2<B,R,IO1,CL>::importancesampling(Random& rng,
    const real T, State<B,L>& s, F* filter, IO2* inInit, const int C) {
  int c;
  std::cerr << "Importance sampling..."<< std::endl;
  std::cerr << "# observations : " << T << std::endl;
  int Nx = s.size();
  std::cerr << "# x-particles : " << Nx << std::endl;
  int Ntheta = 10;
  std::cerr << "# thetaparticles : " << Ntheta << std::endl;
  /*
   * Init theta particles from the prior;
   * We could initialize from another distribution
   * and use importance sampling to retrieve a sample
   * from the posterior distribution, as long as
   * the prior density is evaluable.
   * So we could separate the initial distribution
   * from the prior distribution.
   */
  matrix_type thetaparticles(Ntheta, NP);

  vector_type thetaweights(Ntheta);
  for (int indexrow = 0; indexrow < Ntheta; indexrow ++){
	  if (indexrow % 10 == 0){
		  m.parameterSamples(rng, s);
		  row(thetaparticles, indexrow) = row(s.get(P_VAR), 0);
		  row(thetaparticles, indexrow)[0] = 0.1;
	  } else {
		  row(thetaparticles, indexrow) = row(thetaparticles, indexrow-1);
	  }
	  thetaweights(indexrow) = 0.;
  }
  /*
   * Run a filter from each theta particles,
   * get the associated likelihood and put in the theta-particle weights
   */
  for (int indexrow = 0; indexrow < Ntheta; indexrow ++){
	  s.resizeMax(Nx, true);
	  if (initial == INCLUDE_INITIAL) {
		  set_rows(s.get(D_VAR), subrange(row(thetaparticles, indexrow), NP, ND));
	  } else {
		  m.initialSamples(rng, s);
	  }
	  // Copy and pasting the content of the filter() method of ParticleFilter

	  filter->reset();
	  assert (T >= filter->getTime());
      assert (filter->getEssRel() >= 0.0 && filter->getEssRel() <= 1.0);
      // Here I use Nx for P (I guess) const int P = s.size();
      int n = 0, r = 0;
      typename loc_temp_vector<L,real>::type lws(Nx);
      typename loc_temp_vector<L,int>::type as(Nx);
      real ll = 0.0;
      filter->init(rng, row(thetaparticles, indexrow), s, lws, as);
      while (filter->getTime() < T) {
    	  r = n > 0 && filter->resample(rng, s, lws, as);
    	  filter->predict(rng, T, s);
    	  filter->correct(s, lws);
    	  ll += logsumexp_reduce(lws) - std::log(Nx);
    	  ++n;
      }
	  thetaweights(indexrow) = ll;
  }

  NcFile ncfile("results/smc2.nc", NcFile::Replace);
  NcDim* xDim  = ncfile.add_dim("Ntheta", Ntheta);
  NcDim* NPDim  = ncfile.add_dim("ParameterDim", NP);
  NcVar* weightdata = ncfile.add_var("logweights", ncDouble, xDim);
  NcVar* thetadata = ncfile.add_var("thetaparticles", ncDouble, NPDim, xDim);
  weightdata->put(thetaweights.buf(), Ntheta);
  thetadata->put(thetaparticles.buf(), NP, Ntheta);
}


template<class B, class R, class IO1, bi::Location CL>
template<bi::Location L, class F, class IO2>
void bi::SMC2<B,R,IO1,CL>::sample(Random& rng,
    const real T, State<B,L>& s, F* filter, IO2* inInit, const int C) {
  int c;
  std::cerr << "Importance sampling..."<< std::endl;
  std::cerr << "# observations : " << T + 1 << std::endl;
  int Nx = s.size();
  std::cerr << "# x-particles : " << Nx << std::endl;
  int Ntheta = 10;
  std::cerr << "# thetaparticles : " << Ntheta << std::endl;
  // thetaparticles contain the values of the parameters at the current time step
  matrix_type thetaparticles(Ntheta, NP);
  // log weights associated to each parameter value at the current time step
  vector_type thetaweights(Ntheta);
  // log weights history: all time steps
  std::vector<vector_type* > thetaweightsHistory;

  // vector of States associated to each theta-particle,
  // at current time step
  std::vector<State <B, L>* > allStates;
  // log weights of the x-particles associated to each theta-particle
  // at current time step
  std::vector<typename loc_temp_vector<L,real>::type *> allXLogWeights;
  // ancestors of the x-particles associated to each theta-particle
  // at current time step
  std::vector<typename loc_temp_vector<L,int>::type *> allXAncestors;
  // log likelihood associated to each theta-particles at current
  // time step
  vector_type allLogLikelihoods(Ntheta);

  //
  /*
   * Init theta particles from the prior;
   * We could initialize from another distribution
   * and use importance sampling to retrieve a sample
   * from the posterior distribution, as long as
   * the prior density is evaluable.
   * So we could separate the initial distribution
   * from the prior distribution.
   */
  for (int indextheta = 0; indextheta < Ntheta; indextheta ++){
	  m.parameterSamples(rng, s);
	  row(thetaparticles, indextheta) = row(s.get(P_VAR), 0);
	  thetaweights(indextheta) = 0.0;
  }

  real currentSMC2time = filter->getTime();
  /*
   * create objects of appropriate size in the various vectors
   */
  for (int indextheta = 0; indextheta < Ntheta; indextheta++){
	  State <B, L>* pointerToState = new State<B, L>(s);
	  allStates.push_back(pointerToState);
	  typename loc_temp_vector<L,real>::type* pointerToLws =
			  new typename loc_temp_vector<L,real>::type(Nx);
	  typename loc_temp_vector<L,int>::type* pointerToAncs =
			  new typename loc_temp_vector<L,int>::type(Nx);
	  allXLogWeights.push_back(pointerToLws);
	  allXAncestors.push_back(pointerToAncs);
	  allLogLikelihoods[indextheta] = 0.0;
  }
  filter->reset();
  int n = 0, r = 0;
  /*
   * init all the associated x-particles
   */
  for (int indextheta = 0; indextheta < Ntheta; indextheta ++){
	  if (indextheta != Ntheta - 1) {filter->mark();}
	  filter->init(rng, row(thetaparticles, indextheta), *allStates[indextheta],
			  *allXLogWeights[indextheta], *allXAncestors[indextheta]);
	  if (indextheta != Ntheta - 1) {filter->restore();}
  }
  cerr << "starting loop over time" << endl;
  while (filter->getTime() < T) {
	  for (int indextheta = 0; indextheta < Ntheta; indextheta ++){
		  // predict step for the theta-particles:
		  // empty [no dynamic on theta]
		  // correction step for the theta-particles:
		  // update each x-filter, and get the associated log likelihood
		  if (indextheta != Ntheta - 1){ filter->mark();}
		  r = n > 0 && filter->resample(rng, *allStates[indextheta],
				  *allXLogWeights[indextheta], *allXAncestors[indextheta]);
		  filter->predict(rng, T, *allStates[indextheta]);
		  filter->correct(*allStates[indextheta], *allXLogWeights[indextheta]);
		  real incrLL = logsumexp_reduce(*allXLogWeights[indextheta]) - std::log(Nx);
		  allLogLikelihoods[indextheta] += incrLL;
		  thetaweights[indextheta] += incrLL;
		  if (indextheta != Ntheta - 1){ filter->restore();}
	  }
	  vector_type * pointerToThetaWeights = new vector_type(Ntheta);
	  (*pointerToThetaWeights) = thetaweights;
	  thetaweightsHistory.push_back(pointerToThetaWeights);
	  ++n;
  }

  cerr << thetaweights << endl;
  vector_type thetaancestors(Ntheta);
  resam->ancestors(rng, thetaweights, thetaancestors);
  cerr << thetaancestors << endl;

  /*
   * Convert std::vector of vector_type into matrix_type, for writing
   * into a NetCDF file
   */
  matrix_type allthetaw(Ntheta, n);
  for (int indexcol = 0; indexcol < n; indexcol++ ){
	  for (int indexrow = 0; indexrow < Ntheta; indexrow ++ ){
		  row(allthetaw, indexrow)[indexcol] = (*thetaweightsHistory[indexcol])[indexrow];
	  }
	  // we can then delete these:
	  delete thetaweightsHistory[indexcol];
  }
  /*
   * Writing into NetCDF file
   */
  NcFile ncfile("results/smc2.nc", NcFile::Replace);
  NcDim* xDim  = ncfile.add_dim("Ntheta", Ntheta);
  NcDim* NPDim  = ncfile.add_dim("ParameterDim", NP);
  NcDim* TDim  = ncfile.add_dim("TimeDim", n);
  NcVar* weightdata = ncfile.add_var("logweights", ncDouble, xDim);
  NcVar* thetadata = ncfile.add_var("thetaparticles", ncDouble, NPDim, xDim);
  weightdata->put(thetaweights.buf(), Ntheta);
  thetadata->put(thetaparticles.buf(), NP, Ntheta);
  NcVar* allthetawdata = ncfile.add_var("weighthistory", ncDouble, TDim, xDim);
  allthetawdata->put(allthetaw.buf(), n, Ntheta);
  /*
   * delete remaining stuff
   */
  for (int indextheta = 0; indextheta < Ntheta; indextheta++){
	  delete allStates[indextheta];
	  delete allXLogWeights[indextheta];
	  delete allXAncestors[indextheta];
  }
}

#endif
