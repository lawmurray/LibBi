/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2424 $
 * $Date: 2012-03-23 17:40:36 +0800 (Fri, 23 Mar 2012) $
 */
#ifndef BI_METHOD_NELDERMEADOPTIMISER_HPP
#define BI_METHOD_NELDERMEADOPTIMISER_HPP

#include "../state/State.hpp"
#include "../math/gsl.hpp"

#include <gsl/gsl_multimin.h>

namespace bi {
/**
 * @internal
 *
 * %State of NelderMeadOptimiser.
 */
struct NelderMeadOptimiserState {
  /**
   * Constructor.
   *
   * @param m Model.
   */
  NelderMeadOptimiserState(const int M);

  /**
   * Destructor.
   */
  ~NelderMeadOptimiserState();

  /**
   * State.
   */
  gsl_vector* x;

  /**
   * Step size.
   */
  gsl_vector* step;

  /**
   * Minimizer.
   */
  gsl_multimin_fminimizer* minimizer;

  /**
   * Size.
   */
  double size;
};
}

inline bi::NelderMeadOptimiserState::NelderMeadOptimiserState(const int M) {
  x = gsl_vector_alloc(M);
  step = gsl_vector_alloc(M);
  minimizer = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2, M);
}

inline bi::NelderMeadOptimiserState::~NelderMeadOptimiserState() {
  gsl_vector_free(x);
  gsl_vector_free(step);
  gsl_multimin_fminimizer_free(minimizer);
}

namespace bi {
/**
 * Parameter structure passed to function to optimise.
 */
template<class B, Location L, class F>
struct NelderMeadOptimiserParams {
  B* m;
  Random* rng;
  State<B,L>* s;
  F* filter;
  real T;
};

/**
 * Nelder-Mead simplex optimisation.
 *
 * @ingroup method
 *
 * @tparam B Model type
 * @tparam IO1 #concept::OptimiserBuffer type.
 * @tparam CL Cache location.
 */
template<class B, class IO1, Location CL = ON_HOST>
class NelderMeadOptimiser {
public:
  /**
   * State type.
   */
  typedef NelderMeadOptimiserState state_type;

  /**
   * Constructor.
   *
   * @tparam IO1 #concept::OptimiserBuffer type.
   *
   * @param m Model.
   * @param out Output.
   * @param mode Mode of operation.
   *
   * @see ParticleFilter
   */
  NelderMeadOptimiser(B& m, IO1* out = NULL,
      const OptimiserMode mode = MAXIMUM_LIKELIHOOD);

  /**
   * Get output buffer.
   */
  IO1* getOutput();

  /**
   * @name High-level interface.
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Optimise.
   *
   * @tparam L Location.
   * @tparam F #concept::Filter type.
   * @tparam IO2 #concept::SparseInputBuffer type.
   *
   * @param rng Random number generator.
   * @param T Length of time to optimise over.
   * @param[in,out] s State.
   * @param filter Filter.
   * @param inInit Initialisation file.
   * @param stopSteps Maximum number of steps to take.
   * @param stopSize Size for stopping criterion.
   *
   * Note that @p s should be initialised with a starting state.
   */
  template<Location L, class F, class IO2>
  void optimise(Random& rng, const real T, State<B,L>& s, F* filter,
      IO2* inInit = NULL, const int stopSteps = 100,
      const real stopSize = 1.0e-4);
  //@}

  /**
   * @name Low-level interface.
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Initialise.
   *
   * @tparam L Location.
   * @tparam F #concept::Filter type.
   * @tparam IO2 #concept::SparseInputBuffer type.
   *
   * @param rng Random number generator.
   * @param T Length of time to optimise over.
   * @param s State.
   * @param filter Filter.
   * @param inInit Initialisation file.
   */
  template<Location L, class F, class IO2>
  void init(Random& rng, const real T, State<B,L>& s, F* filter, IO2* inInit);

  /**
   * Perform one iteration step of optimiser.
   */
  void step();

  /**
   * Has optimiser converged?
   *
   * @param stopSize Size for stopping criterion.
   */
  bool hasConverged(const real stopSize = 1.0e-4);

  /**
   * Output current state.
   *
   * @tparam L Location.
   *
   * @param k Index in output file.
   * @param s State.
   */
  template<Location L>
  void output(const int k, const State<B,L>& s);

  /**
   * Report progress on stderr.
   *
   * @param k Number of steps taken.
   */
  void report(const int k);

  /**
   * Terminate.
   */
  void term();
  //@}

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
   * Optimisation mode.
   */
  OptimiserMode mode;

  /**
   * Current state.
   */
  NelderMeadOptimiserState state;

  /**
   * Cost function for maximum likelihood.
   */
  template<Location L, class F>
  static double ml(const gsl_vector* x, void* params);

  /**
   * Cost function for maximum a posteriori.
   */
  template<Location L, class F>
  static double map(const gsl_vector* x, void* params);

  /* net sizes, for convenience */
  static const int ND = net_size<typename B::DTypeList>::value;
  static const int NP = net_size<typename B::PTypeList>::value;
};

/**
 * Factory for creating NelderMeadOptimiser objects.
 *
 * @ingroup method
 *
 * @tparam CL Cache location.
 *
 * @see NelderMeadOptimiser
 */
template<Location CL = ON_HOST>
struct NelderMeadOptimiserFactory {
  /**
   * Create Nelder-Mead optimiser.
   *
   * @return NelderMeadOptimiser object. Caller has ownership.
   *
   * @see NelderMeadOptimiser::NelderMeadOptimiser()
   */
  template<class B, class IO1>
  static NelderMeadOptimiser<B,IO1,CL>* create(B& m, IO1* out = NULL,
      const OptimiserMode mode = MAXIMUM_LIKELIHOOD) {
    return new NelderMeadOptimiser<B,IO1,CL>(m, out, mode);
  }
};
}

#include "Resampler.hpp"
#include "../math/misc.hpp"
#include "../math/view.hpp"
#include "../math/temp_vector.hpp"
#include "../misc/exception.hpp"

template<class B, class IO1, bi::Location CL>
bi::NelderMeadOptimiser<B,IO1,CL>::NelderMeadOptimiser(B& m, IO1* out,
    const OptimiserMode mode) :
    m(m),
    out(out),
    mode(mode),
    state(NP) {
  //
}

template<class B, class IO1, bi::Location CL>
IO1* bi::NelderMeadOptimiser<B,IO1,CL>::getOutput() {
  return out;
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L, class F, class IO2>
void bi::NelderMeadOptimiser<B,IO1,CL>::optimise(Random& rng, const real T,
    State<B,L>& s, F* filter, IO2* inInit, const int stopSteps,
    const real stopSize) {
  int k = 0;
  init(rng, T, s, filter, inInit);
  while (k < stopSteps && !hasConverged(stopSize)) {
    step();
    report(k);
    output(k, s);
    ++k;
  }
  term();
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L, class F, class IO2>
void bi::NelderMeadOptimiser<B,IO1,CL>::init(Random& rng, const real T,
    State<B,L>& s, F* filter, IO2* inInit) {
  /* first run of filter to get everything initialised properly */
  filter->rewind();
  filter->filter(rng, 0.0, s, inInit);

  /* initialise state vector */
  BOOST_AUTO(x, gsl_vector_reference(state.x));
  x = vec(s.get(P_VAR));
  set_elements(gsl_vector_reference(state.step), 0.01);

  /* parameters */
  NelderMeadOptimiserParams<B,L,F>* params =
      new NelderMeadOptimiserParams<B,L,F>(); ///@todo Leaks
  params->m = &m;
  params->rng = &rng;
  params->s = &s;
  params->filter = filter;
  params->T = T;

  /* function */
  gsl_multimin_function* f = new gsl_multimin_function(); ///@todo Leaks
  if (mode == MAXIMUM_A_POSTERIORI) {
    f->f = NelderMeadOptimiser<B,IO1,CL>::template map<L,F>;
  } else {
    f->f = NelderMeadOptimiser<B,IO1,CL>::template ml<L,F>;
  }
  f->n = M;
  f->params = params;

  gsl_multimin_fminimizer_set(state.minimizer, f, state.x, state.step);
}

template<class B, class IO1, bi::Location CL>
void bi::NelderMeadOptimiser<B,IO1,CL>::step() {
  int status = gsl_multimin_fminimizer_iterate(state.minimizer);
  BI_ERROR(status == GSL_SUCCESS, "iteration failed");
}

template<class B, class IO1, bi::Location CL>
bool bi::NelderMeadOptimiser<B,IO1,CL>::hasConverged(const real stopSize) {
  state.size = gsl_multimin_fminimizer_size(state.minimizer);
  return gsl_multimin_test_size(state.size, stopSize) == GSL_SUCCESS;
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L>
void bi::NelderMeadOptimiser<B,IO1,CL>::output(const int k, const State<B,L>& s) {
  if (out != NULL) {
    out->writeState(P_VAR, k, vec(s.get(P_VAR)));
    out->writeState(D_VAR, k, row(s.get(D_VAR), 0));
    out->writeValue(k, -state.minimizer->fval);
    out->writeSize(k, state.size);
  }
}

template<class B, class IO1, bi::Location CL>
void bi::NelderMeadOptimiser<B,IO1,CL>::report(const int k) {
  std::cerr << k << ":\t";
  std::cerr << "value=" << -state.minimizer->fval;
  std::cerr << '\t';
  std::cerr << "size=" << state.size;
  std::cerr << std::endl;
}

template<class B, class IO1, bi::Location CL>
void bi::NelderMeadOptimiser<B,IO1,CL>::term() {
  //
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L, class F>
double bi::NelderMeadOptimiser<B,IO1,CL>::ml(const gsl_vector* x,
    void* params) {
  typedef NelderMeadOptimiserParams<B,L,F> param_type;
  param_type* p = reinterpret_cast<param_type*>(params);

  /* evaluate */
  try {
    p->filter->rewind();
    real ll = p->filter->filter(*p->rng, p->T, gsl_vector_reference(x), *p->s);
    return -ll;
  } catch (CholeskyException e) {
    return GSL_NAN;
  } catch (ParticleFilterDegeneratedException e) {
    return GSL_NAN;
  }
}

template<class B, class IO1, bi::Location CL>
template<bi::Location L, class F>
double bi::NelderMeadOptimiser<B,IO1,CL>::map(const gsl_vector* x,
    void* params) {
  typedef NelderMeadOptimiserParams<B,L,F> param_type;
  param_type* p = reinterpret_cast<param_type*>(params);
  typename temp_host_vector<real>::type lp(1);
  lp.clear();

  int P = p->s->size();
  p->s->resize(1, true);

  /* initialise */
  vec(p->s->getAlt(P_VAR)) = gsl_vector_reference(x);
  p->m->parameterLogDensities(*p->s, lp);
  p->m->parameterPostLogDensities(*p->s, lp);
  p->s->resize(P, true);

  /* evaluate */
  try {
    p->filter->rewind();
    real ll = p->filter->filter(*p->rng, p->T, gsl_vector_reference(x), *p->s);
    return -(ll + lp(0));
  } catch (CholeskyException e) {
    return GSL_NAN;
  }
}

#endif
