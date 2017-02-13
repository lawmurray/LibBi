/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_OPTIMISER_NELDERMEADOPTIMISER_HPP
#define BI_OPTIMISER_NELDERMEADOPTIMISER_HPP

#include "../state/Schedule.hpp"
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
  minimizer = gsl_multimin_fminimizer_alloc(
      gsl_multimin_fminimizer_nmsimplex2, M);
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
template<class B, class F, class S, class IO1, class IO2>
struct NelderMeadOptimiserParams {
  B* m;
  Random* rng;
  S* s;
  F* filter;
  IO1* out;
  IO2* in;
  ScheduleIterator first, last;
};

/**
 * Nelder-Mead simplex optimisation.
 *
 * @ingroup method_optimiser
 *
 * @tparam B Model type
 * @tparam F #concept::Filter type.
 */
template<class B, class F>
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
   * @param filter Filter.
   * @param out Output.
   * @param mode Mode of operation.
   *
   * @see BootstrapPF
   */
  NelderMeadOptimiser(B& m, F& filter, const OptimiserMode mode = MAXIMUM_LIKELIHOOD);

  /**
   * @name High-level interface
   *
   * An easier interface for common usage.
   */
  //@{
  /**
   * Optimise.
   *
   * @tparam S State type.
   * @tparam IO1 Output type.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param[in,out] s State.
   * @param out Output buffer.
   * @param inInit Initialisation file.
   * @param simplexSizeRel Size of simplex relative to each dimension.
   * @param stopSteps Maximum number of steps to take.
   * @param stopSize Size for stopping criterion.
   *
   * Note that @p s should be initialised with a starting state.
   */
  template<class S, class IO1, class IO2>
  void optimise(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S& s, IO1& out, IO2& inInit,
      const real simplexSizeRel = 0.1, const int stopSteps = 100,
      const real stopSize = 1.0e-4);
  //@}

  /**
   * @name Low-level interface
   *
   * Largely used by other features of the library or for finer control over
   * performance and behaviour.
   */
  //@{
  /**
   * Initialise.
   *
   * @tparam S State type.
   * @tparam IO1 Output type.
   * @tparam IO2 Input type.
   *
   * @param[in,out] rng Random number generator.
   * @param first Start of time schedule.
   * @param last End of time schedule.
   * @param s State.
   * @param[in,out] out Output buffer;
   * @param inInit Initialisation file.
   * @param simplexSizeRel Size of simplex relative to each dimension.
   */
  template<class S, class IO1, class IO2>
  void init(Random& rng, const ScheduleIterator first,
      const ScheduleIterator last, S& s, IO1& out, IO2& inInit,
      const real simplexSizeRel = 0.1);

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
   * @tparam S State type.
   * @tparam IO1 Output type.
   *
   * @param k Index in output file.
   * @param s State.
   * @param[in,out] out Output buffer.
   */
  template<class S, class IO1>
  void output(const int k, const S& s, IO1& out);

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
   * Filter.
   */
  F& filter;

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
  template <class S, class IO1, class IO2>
  static double ml(const gsl_vector* x, void* params);

  /**
   * Cost function for maximum a posteriori.
   */
  template <class S, class IO1, class IO2>
  static double map(const gsl_vector* x, void* params);
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
  template<class B, class F>
  static NelderMeadOptimiser<B,F>* create(B& m, F& filter,
      const OptimiserMode mode = MAXIMUM_LIKELIHOOD) {
    return new NelderMeadOptimiser<B,F>(m, filter, mode);
  }
};
}

#include "../math/misc.hpp"
#include "../math/view.hpp"
#include "../math/temp_vector.hpp"
#include "../misc/exception.hpp"

#include "../misc/TicToc.hpp"

template<class B, class F>
bi::NelderMeadOptimiser<B,F>::NelderMeadOptimiser(B& m, F& filter,
    const OptimiserMode mode) :
    m(m), filter(filter), mode(mode), state(B::NP) {
  //
}

template<class B, class F>
template<class S, class IO1, class IO2>
void bi::NelderMeadOptimiser<B,F>::optimise(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, S& s,
    IO1& out, IO2& inInit, const real simplexSizeRel, const int stopSteps,
    const real stopSize) {
  TicToc clock;
  int k = 0;
  init(rng, first, last, s.s, s.out, inInit, simplexSizeRel);
  while (k < stopSteps && !hasConverged(stopSize)) {
    step();
    report(k);
    output(k, s.s, out);
    ++k;
  }
  s.clock = clock.toc();
  out.writeClock(s.clock);
  term();
}

template<class B, class F>
template<class S, class IO1, class IO2>
void bi::NelderMeadOptimiser<B,F>::init(Random& rng,
    const ScheduleIterator first, const ScheduleIterator last, S& s,
    IO1& out, IO2& inInit, const real simplexSizeRel) {
  filter.init(rng, *first, s, out, inInit);
  filter.filter(rng, first, last, s, out);

  /* initialise state vector */
  BOOST_AUTO(x, gsl_vector_reference(state.x));
  x = vec(s.get(P_VAR));
  gsl_vector_reference(state.step) = x;
  mulscal_elements(gsl_vector_reference(state.step), simplexSizeRel,
      gsl_vector_reference(state.step));

  /* parameters */
  NelderMeadOptimiserParams<B,F,S,IO1,IO2>* params = new NelderMeadOptimiserParams<B,F,S,IO1,IO2>();  ///@todo Leaks
  params->m = &m;
  params->rng = &rng;
  params->s = &s;
  params->filter = &filter;
  params->out = &out;
  params->in = &inInit;
  params->first = first;
  params->last = last;

  /* function */
  gsl_multimin_function* f = new gsl_multimin_function();  ///@todo Leaks
  if (mode == MAXIMUM_A_POSTERIORI) {
    f->f = NelderMeadOptimiser<B,F>::template map<S,IO1,IO2>;
  } else {
    f->f = NelderMeadOptimiser<B,F>::template ml<S,IO1,IO2>;
  }
  f->n = B::NP;
  f->params = params;

  gsl_multimin_fminimizer_set(state.minimizer, f, state.x, state.step);
}

template<class B, class F>
void bi::NelderMeadOptimiser<B,F>::step() {
  int status = gsl_multimin_fminimizer_iterate(state.minimizer);
  BI_ERROR_MSG(status == GSL_SUCCESS, "iteration failed");
}

template<class B, class F>
bool bi::NelderMeadOptimiser<B,F>::hasConverged(const real stopSize) {
  state.size = gsl_multimin_fminimizer_size(state.minimizer);
  return gsl_multimin_test_size(state.size, stopSize) == GSL_SUCCESS;
}

template<class B, class F>
template<class S, class IO1>
void bi::NelderMeadOptimiser<B,F>::output(const int k,
    const S& s, IO1& out) {
  out.writeParameters(k, s.get(P_VAR));
  out.writeValue(k, -state.minimizer->fval);
  out.writeSize(k, state.size);
}

template<class B, class F>
void bi::NelderMeadOptimiser<B,F>::report(const int k) {
  std::cerr << k << ":\t";
  std::cerr << "value=" << -state.minimizer->fval;
  std::cerr << '\t';
  std::cerr << "size=" << state.size;
  std::cerr << std::endl;
}

template<class B, class F>
void bi::NelderMeadOptimiser<B,F>::term() {
  //
}

template<class B, class F>
template<class S, class IO1, class IO2>
double bi::NelderMeadOptimiser<B,F>::ml(const gsl_vector* x,
    void* params) {
  typedef NelderMeadOptimiserParams<B,F,S,IO1,IO2> param_type;
  param_type* p = reinterpret_cast<param_type*>(params);

  /* evaluate */
  try {
    p->filter->init(*p->rng, *(p->first), *p->s, *p->out, *p->in);
    p->filter->filter(*p->rng, p->first, p->last, *p->s, *p->out);
    real ll = (*p->s).logLikelihood;
    return -ll;
  } catch (CholeskyException e) {
    return GSL_NAN;
  } catch (ParticleFilterDegeneratedException e) {
    return GSL_NAN;
  }
}

template<class B, class F>
template<class S, class IO1, class IO2>
double bi::NelderMeadOptimiser<B,F>::map(const gsl_vector* x,
    void* params) {
  typedef NelderMeadOptimiserParams<B,F,S,IO1,IO2> param_type;
  param_type* p = reinterpret_cast<param_type*>(params);

  int P = p->s->size();
  p->s->resizeMax(1, true);

  /* initialise */
  vec(p->s->get(PY_VAR)) = gsl_vector_reference(x);
  real lp = p->m->parameterLogDensity(*p->s);
  p->s->resizeMax(P, true);

  /* evaluate */
  if (bi::is_finite(lp)) {
    try {
      p->filter->init(*p->rng, *(p->first), *p->s, *p->out, *p->in);
      p->filter->filter(*p->rng, p->first, p->last, *p->s, *p->out);
      real ll = (*p->s).logLikelihood;
      return -(ll + lp);
    } catch (CholeskyException e) {
      return GSL_NAN;
    }
  } else {
    return GSL_NAN;
  }
}

#endif
